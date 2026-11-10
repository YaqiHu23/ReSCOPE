import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scorer import ROIScorer
from .config import positive_int


def gumbel_softmax_sample(logits, tau=1.0, hard=False, eps=1e-10):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y = F.softmax((logits + g) / tau, dim=-1)

    if not hard:
        return y

    idx = y.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
    return y_hard - y.detach() + y


class PartTileROIDictionary(nn.Module):
    def __init__(
        self,
        in_ch,
        roi_tiles,
        stride_tiles,
        tau_select=1.0,
        tau_match=0.05,
        recon_percent=0.25,
        max_samples=None,
        hard_select_train=False,
        eps=1e-6,
        roi_feat_chunk=256,
        normalize_select_logits=True,
        select_logit_clip=8.0,
        select_entropy_weight=1e-3,
        roi_valid_ratio=0.7,
        avoid_roi_overlap=True,
        selection_mode="combinatorial_thretopk",
        selection_noise="gumbel",
        sample_in_eval=False,
        thretopk_eps=1e-6,
        reconstruction_mode="roi_prototypes",
        prototype_grid_size=8,
        prototype_use_straight_through=True,
        target_outside_selected_roi=True,
    ):
        super().__init__()
        self.scorer = ROIScorer(in_ch, roi_tiles, stride_tiles)
        self.roi_tiles = int(roi_tiles)
        self.stride_tiles = int(stride_tiles)
        self.tau_select = float(tau_select)
        self.tau_match = float(tau_match)
        self.recon_percent = float(recon_percent)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.hard_select_train = bool(hard_select_train)
        self.eps = float(eps)
        self.roi_feat_chunk = int(roi_feat_chunk)
        self.normalize_select_logits = bool(normalize_select_logits)
        self.select_logit_clip = float(select_logit_clip)
        self.select_entropy_weight = float(select_entropy_weight)
        self.roi_valid_ratio = float(roi_valid_ratio)
        self.avoid_roi_overlap = bool(avoid_roi_overlap)
        self.selection_mode = str(selection_mode)
        self.selection_noise = str(selection_noise).lower()
        self.sample_in_eval = bool(sample_in_eval)
        self.thretopk_eps = float(thretopk_eps)
        self.reconstruction_mode = str(reconstruction_mode)
        self.prototype_grid_size = int(prototype_grid_size)
        self.prototype_use_straight_through = bool(prototype_use_straight_through)
        self.target_outside_selected_roi = bool(target_outside_selected_roi)
        if self.prototype_grid_size < 1:
            raise ValueError(f"prototype_grid_size must be >= 1, got {prototype_grid_size}")

    def _stable_selection_logits(self, logits, candidate_mask):
        logits32 = logits.float()
        valid_logits = logits32[candidate_mask]
        if valid_logits.numel() == 0:
            raise ValueError("No valid ROI candidates after masking")

        if self.normalize_select_logits:
            mu = valid_logits.mean()
            sigma = valid_logits.std(unbiased=False)
            logits32 = (logits32 - mu) / (sigma + self.eps)
        if self.select_logit_clip > 0:
            logits32 = logits32.clamp(-self.select_logit_clip, self.select_logit_clip)
        logits32 = logits32.masked_fill(~candidate_mask, -1e9)
        return logits32

    def compute_roi_candidate_mask(self, valid_mask):
        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.ndim != 3:
            raise ValueError(f"valid_mask must be [H,W] or [B,H,W], got {valid_mask.shape}")

        pooled = F.avg_pool2d(
            valid_mask.float().unsqueeze(1),
            kernel_size=self.roi_tiles,
            stride=self.stride_tiles,
        )
        return pooled.squeeze(1) >= self.roi_valid_ratio

    def _mark_overlapping_rois(self, used_mask, idx, nh, nw):
        if not self.avoid_roi_overlap:
            used_mask[idx] = True
            return used_mask

        row = int(idx) // int(nw)
        col = int(idx) % int(nw)
        radius = max(0, int(math.ceil(float(self.roi_tiles) / float(self.stride_tiles))) - 1)

        row0 = max(0, row - radius)
        row1 = min(int(nh), row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(int(nw), col + radius + 1)

        used_grid = used_mask.view(int(nh), int(nw))
        used_grid[row0:row1, col0:col1] = True
        return used_mask

    def _sample_selection_logits(self, logits):
        if self.selection_noise != "gumbel":
            return logits
        if not (self.training or self.sample_in_eval):
            return logits

        u = torch.rand_like(logits).clamp_(self.thretopk_eps, 1.0 - self.thretopk_eps)
        g = -torch.log(-torch.log(u))
        return logits + g

    def _smooth_laplace_cdf(self, x):
        return 0.5 + 0.5 * torch.sign(x) * (1.0 - torch.exp(-torch.abs(x)))

    def _smooth_laplace_pdf(self, x):
        return 0.5 * torch.exp(-torch.abs(x))

    def _thretopk_laplace_valid(self, scores, k):
        n = int(scores.numel())
        if n == 0:
            raise ValueError("No valid scores passed to ThreTopK")
        if k <= 0:
            return torch.zeros_like(scores)
        if k >= n:
            return torch.ones_like(scores)

        x_sort, _ = torch.sort(scores)
        lse_lower = torch.logcumsumexp(x_sort, dim=0)
        lse_upper_inclusive = torch.logcumsumexp((-x_sort).flip(0), dim=0).flip(0)
        lse_upper = torch.empty_like(x_sort)
        lse_upper[:-1] = lse_upper_inclusive[1:]
        lse_upper[-1] = -torch.inf

        m = torch.arange(n - 1, -1, -1, device=scores.device, dtype=scores.dtype)
        k_tensor = torch.as_tensor(float(k), device=scores.device, dtype=scores.dtype)
        km = k_tensor - m
        term = torch.sqrt(km.square() + torch.exp(lse_lower + lse_upper)) + km
        term = term.clamp_min(self.thretopk_eps)
        lambdas = lse_lower - torch.log(term)

        upper_bound = torch.empty_like(x_sort)
        upper_bound[:-1] = x_sort[1:]
        upper_bound[-1] = torch.inf
        valid = (
            torch.isfinite(lambdas)
            & (lambdas >= x_sort - self.thretopk_eps)
            & (lambdas <= upper_bound + self.thretopk_eps)
        )
        if bool(valid.any().item()):
            lamb = lambdas[valid][-1]
        else:
            lamb = self._thretopk_laplace_bisect(scores, k)

        z_detached_lambda = scores - lamb
        dlambda = self._smooth_laplace_pdf(z_detached_lambda)
        dlambda = dlambda / dlambda.sum().clamp_min(self.thretopk_eps)
        custom_lamb = (scores * dlambda.detach()).sum()
        lamb = custom_lamb + (lamb - custom_lamb).detach()
        return self._smooth_laplace_cdf(scores - lamb).clamp(0.0, 1.0)

    def _thretopk_laplace_bisect(self, scores, k, steps=64):
        low = scores.min() - 32.0
        high = scores.max() + 32.0
        target = float(k)

        for _ in range(int(steps)):
            mid = 0.5 * (low + high)
            mass = self._smooth_laplace_cdf(scores - mid).sum()
            if float(mass.detach().item()) < target:
                high = mid
            else:
                low = mid

        return 0.5 * (low + high)

    def _thretopk_laplace(self, logits, k, candidate_mask):
        valid_scores = logits[candidate_mask]
        valid_soft = self._thretopk_laplace_valid(valid_scores, int(k))
        soft = torch.zeros_like(logits)
        soft = soft.masked_scatter(candidate_mask, valid_soft)
        return soft

    def _decode_nonoverlap_topk(self, scores, candidate_mask, requested, nh, nw):
        ranking_scores = scores.masked_fill(~candidate_mask, -torch.inf)
        order = torch.argsort(ranking_scores, descending=True)
        used_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        indices = []

        for idx_tensor in order:
            idx = int(idx_tensor.item())
            if not bool(candidate_mask[idx].item()):
                break
            if bool(used_mask[idx].item()):
                continue

            indices.append(idx)
            used_mask = self._mark_overlapping_rois(used_mask.clone(), idx, nh, nw)
            if len(indices) >= int(requested):
                break

        return indices

    def select_combinatorial_rois_soft(self, logits, num_rois, candidate_mask, nh, nw):
        requested = positive_int(num_rois)

        available = int(candidate_mask.sum().item())
        if available == 0:
            raise ValueError("No ROI candidates available for selection")

        if available < requested:
            raise ValueError(f"Requested {requested} ROIs but only {available} valid candidates exist")
        picks = requested
        sampled_logits = self._sample_selection_logits(logits)
        scaled_logits = sampled_logits / max(self.tau_select, self.thretopk_eps)
        soft_multi_hot = self._thretopk_laplace(
            logits=scaled_logits,
            k=picks,
            candidate_mask=candidate_mask,
        )

        roi_indices = self._decode_nonoverlap_topk(
            scores=sampled_logits,
            candidate_mask=candidate_mask,
            requested=picks,
            nh=nh,
            nw=nw,
        )
        if len(roi_indices) != requested:
            raise ValueError(
                f"Non-overlap decoder selected {len(roi_indices)} of {requested} requested ROIs. "
                "Reduce the ROI count/size, adjust stride, or expand the tissue region."
            )

        grad_mass = soft_multi_hot / float(max(len(roi_indices), 1))
        probs_list = []
        for roi_index in roi_indices:
            hard_probs = torch.zeros_like(logits).unsqueeze(0)
            hard_probs[:, int(roi_index)] = 1.0
            soft_probs = grad_mass.unsqueeze(0)
            probs_list.append(hard_probs + (soft_probs - soft_probs.detach()))

        normalized_mass = soft_multi_hot / soft_multi_hot.sum().clamp_min(self.thretopk_eps)
        selection_entropy = -(
            normalized_mass[candidate_mask]
            * normalized_mass[candidate_mask].clamp_min(self.thretopk_eps).log()
        ).sum()
        return probs_list, roi_indices, selection_entropy

    def select_multiple_rois_soft(self, logits, num_rois, candidate_mask, nh, nw):
        requested = positive_int(num_rois)

        available = int(candidate_mask.sum().item())
        if available == 0:
            raise ValueError("No ROI candidates available for selection")

        if available < requested:
            raise ValueError(f"Requested {requested} ROIs but only {available} valid candidates exist")
        picks = requested
        probs_list = []
        indices = []
        used_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)

        for _ in range(picks):
            remaining_mask = candidate_mask & ~used_mask
            if not bool(remaining_mask.any().item()):
                break

            masked_logits = logits.clone()
            masked_logits[~remaining_mask] = -1e9

            probs = gumbel_softmax_sample(
                masked_logits.unsqueeze(0),
                tau=self.tau_select,
                hard=self.hard_select_train,
            )
            idx = int(probs.detach().argmax(dim=-1).item())
            probs_list.append(probs)
            indices.append(idx)
            used_mask = used_mask.clone()
            used_mask = self._mark_overlapping_rois(used_mask, idx, nh, nw)

        if len(indices) != requested:
            raise ValueError(f"Non-overlap decoder selected {len(indices)} of {requested} requested ROIs")
        entropy = 0.0
        for probs in probs_list:
            p = probs.clamp_min(self.eps)
            entropy = entropy + (-(p * p.log()).sum(dim=-1).mean())
        entropy = entropy / max(len(probs_list), 1)

        return probs_list, indices, entropy

    def _selected_roi_tile_mask(self, roi_indices, height, width, nw, valid_mask):
        selected_mask = torch.zeros((height, width), device=valid_mask.device, dtype=torch.bool)

        for roi_index in roi_indices:
            row = int(roi_index) // int(nw)
            col = int(roi_index) % int(nw)
            top = row * self.stride_tiles
            left = col * self.stride_tiles
            bottom = min(top + self.roi_tiles, int(height))
            right = min(left + self.roi_tiles, int(width))
            selected_mask[top:bottom, left:right] = True

        return selected_mask & valid_mask

    def _sample_target_tiles(self, target_y, target_x, sample_targets):
        if not sample_targets:
            return target_y, target_x

        num_targets = int(target_y.numel())
        if num_targets == 0:
            return target_y, target_x

        sample_count = int(num_targets * self.recon_percent)
        if self.max_samples is not None:
            sample_count = min(sample_count, self.max_samples)
        sample_count = max(1, min(sample_count, num_targets))
        if sample_count >= num_targets:
            return target_y, target_x

        perm = torch.randperm(num_targets, device=target_y.device)[:sample_count]
        return target_y[perm], target_x[perm]

    def _roi_probs_to_tile_weight(self, roi_probs_list, roi_indices, height, width, nh, nw):
        tile_weight = None
        kernel = torch.ones(
            (1, 1, self.roi_tiles, self.roi_tiles),
            device=roi_probs_list[0].device,
            dtype=roi_probs_list[0].dtype,
        )

        for probs, roi_index in zip(roi_probs_list, roi_indices):
            hard_probs = torch.zeros_like(probs)
            hard_probs[:, int(roi_index)] = 1.0
            probs_st = hard_probs - probs.detach() + probs
            roi_grid = probs_st.view(1, 1, int(nh), int(nw))
            coverage = F.conv_transpose2d(
                roi_grid,
                kernel,
                stride=self.stride_tiles,
            ).squeeze(0).squeeze(0)

            full_coverage = torch.zeros(
                (height, width),
                device=coverage.device,
                dtype=coverage.dtype,
            )
            cov_h = min(int(height), int(coverage.shape[0]))
            cov_w = min(int(width), int(coverage.shape[1]))
            full_coverage[:cov_h, :cov_w] = coverage[:cov_h, :cov_w]
            tile_weight = full_coverage if tile_weight is None else tile_weight + full_coverage

        if tile_weight is None:
            raise ValueError("No selected ROI probabilities available for reconstruction")
        return tile_weight

    def compute_tile_nn_reconstruction_loss(
        self,
        x,
        valid_mask,
        roi_indices,
        nw,
        chunk_s=512,
        sample_targets=True,
    ):
        batch, channels, height, width = x.shape
        if batch != 1:
            raise ValueError("Only batch size 1 is supported")
        if not roi_indices:
            raise ValueError("No selected ROI indices available for reconstruction")

        valid_grid = valid_mask[0].to(device=x.device, dtype=torch.bool)
        selected_tile_mask = self._selected_roi_tile_mask(
            roi_indices=roi_indices,
            height=height,
            width=width,
            nw=nw,
            valid_mask=valid_grid,
        )
        target_tile_mask = valid_grid & ~selected_tile_mask

        memory_y, memory_x = torch.nonzero(selected_tile_mask, as_tuple=True)
        target_y, target_x = torch.nonzero(target_tile_mask, as_tuple=True)
        if memory_y.numel() == 0:
            raise ValueError("No valid tiles found inside selected ROI(s)")
        if target_y.numel() == 0:
            raise ValueError("No valid tiles found outside selected ROI(s)")

        target_y, target_x = self._sample_target_tiles(
            target_y=target_y,
            target_x=target_x,
            sample_targets=sample_targets,
        )
        memory = x[0, :, memory_y, memory_x].t().contiguous()
        targets = x[0, :, target_y, target_x].t().contiguous()
        memory_norm = F.normalize(memory.float(), dim=-1)

        total_sqerr = torch.zeros((), device=x.device, dtype=torch.float32)
        num_targets = int(targets.shape[0])
        chunk_s = max(1, int(chunk_s))

        for s0 in range(0, num_targets, chunk_s):
            s1 = min(s0 + chunk_s, num_targets)
            target_chunk = targets[s0:s1].float()
            target_norm = F.normalize(target_chunk, dim=-1)
            sim = target_norm @ memory_norm.t()
            nearest_idx = sim.argmax(dim=-1)
            reconstructed = memory[nearest_idx].float()
            total_sqerr = total_sqerr + F.mse_loss(
                reconstructed,
                target_chunk,
                reduction="sum",
            )

        return total_sqerr / float(num_targets * channels)

    def compute_tile_soft_nn_reconstruction_loss(
        self,
        x,
        valid_mask,
        roi_probs_list,
        roi_indices,
        nh,
        nw,
        chunk_s=512,
        sample_targets=True,
    ):
        batch, channels, height, width = x.shape
        if batch != 1:
            raise ValueError("Only batch size 1 is supported")
        if not roi_indices:
            raise ValueError("No selected ROI indices available for reconstruction")

        valid_grid = valid_mask[0].to(device=x.device, dtype=torch.bool)
        selected_tile_weight = self._roi_probs_to_tile_weight(
            roi_probs_list=roi_probs_list,
            roi_indices=roi_indices,
            height=height,
            width=width,
            nh=nh,
            nw=nw,
        )
        selected_tile_weight = selected_tile_weight * valid_grid.to(selected_tile_weight.dtype)

        selected_tile_mask = (selected_tile_weight.detach() > 0.5) & valid_grid
        target_tile_mask = valid_grid & ~selected_tile_mask

        memory_y, memory_x = torch.nonzero(selected_tile_mask, as_tuple=True)
        target_y, target_x = torch.nonzero(target_tile_mask, as_tuple=True)
        if memory_y.numel() == 0:
            raise ValueError("No valid tiles found inside selected ROI(s)")
        if target_y.numel() == 0:
            raise ValueError("No valid tiles found outside selected ROI(s)")

        target_y, target_x = self._sample_target_tiles(
            target_y=target_y,
            target_x=target_x,
            sample_targets=sample_targets,
        )
        memory = x[0, :, memory_y, memory_x].t().contiguous()
        targets = x[0, :, target_y, target_x].t().contiguous()
        memory_prior = selected_tile_weight[memory_y, memory_x].float().clamp_min(self.eps)
        memory_log_prior = memory_prior.log().unsqueeze(0)
        memory_float = memory.float()
        memory_norm = F.normalize(memory_float, dim=-1)

        total_sqerr = torch.zeros((), device=x.device, dtype=torch.float32)
        num_targets = int(targets.shape[0])
        chunk_s = max(1, int(chunk_s))

        for s0 in range(0, num_targets, chunk_s):
            s1 = min(s0 + chunk_s, num_targets)
            target_chunk = targets[s0:s1].float()
            target_norm = F.normalize(target_chunk, dim=-1)
            sim = target_norm @ memory_norm.t()
            weights = F.softmax((sim / self.tau_match) + memory_log_prior, dim=-1)
            reconstructed = weights @ memory_float
            total_sqerr = total_sqerr + F.mse_loss(
                reconstructed,
                target_chunk,
                reduction="sum",
            )

        return total_sqerr / float(num_targets * channels)

    def _prototype_bins(self):
        bins = []
        for idx in range(self.prototype_grid_size):
            start = int(math.floor(float(idx) * float(self.roi_tiles) / float(self.prototype_grid_size)))
            end = int(math.floor(float(idx + 1) * float(self.roi_tiles) / float(self.prototype_grid_size)))
            if idx == self.prototype_grid_size - 1:
                end = self.roi_tiles
            if end <= start:
                raise ValueError(
                    f"prototype_grid_size={self.prototype_grid_size} is too large for roi_tiles={self.roi_tiles}"
                )
            bins.append((start, end))
        return bins

    def _prototype_selection_probs(self, probs, roi_index):
        if not self.prototype_use_straight_through:
            return probs

        hard_probs = torch.zeros_like(probs)
        hard_probs[:, int(roi_index)] = 1.0
        return hard_probs - probs.detach() + probs

    def compute_selected_roi_prototypes(
        self,
        x,
        valid_mask,
        roi_probs_list,
        roi_indices,
        nh,
        nw,
    ):
        if not roi_probs_list:
            raise ValueError("No selected ROI probabilities available for prototype construction")

        _, channels, height, width = x.shape
        valid_grid = valid_mask.to(device=x.device, dtype=torch.bool)
        if valid_grid.ndim == 2:
            valid_grid = valid_grid.unsqueeze(0)
        valid_float = valid_grid[:, None].float()
        x_float = x.float()

        probs_grids = []
        for probs, roi_index in zip(roi_probs_list, roi_indices):
            probs_for_proto = self._prototype_selection_probs(probs, roi_index)
            probs_grids.append(probs_for_proto.view(1, 1, int(nh), int(nw)).float())

        row_bins = self._prototype_bins()
        col_bins = self._prototype_bins()
        prototype_lists = [[] for _ in probs_grids]

        for row0, row1 in row_bins:
            block_h = row1 - row0
            y_stop = row0 + (int(nh) - 1) * self.stride_tiles + block_h
            if y_stop > height:
                raise ValueError(f"Prototype row window exceeds input height: {y_stop} > {height}")

            for col0, col1 in col_bins:
                block_w = col1 - col0
                x_stop = col0 + (int(nw) - 1) * self.stride_tiles + block_w
                if x_stop > width:
                    raise ValueError(f"Prototype column window exceeds input width: {x_stop} > {width}")

                feat_sum = F.avg_pool2d(
                    x_float[:, :, row0:y_stop, col0:x_stop],
                    kernel_size=(block_h, block_w),
                    stride=self.stride_tiles,
                ) * float(block_h * block_w)
                valid_count = F.avg_pool2d(
                    valid_float[:, :, row0:y_stop, col0:x_stop],
                    kernel_size=(block_h, block_w),
                    stride=self.stride_tiles,
                ) * float(block_h * block_w)
                block_mean = feat_sum / valid_count.clamp_min(self.eps)
                block_mean = block_mean * (valid_count > 0).to(block_mean.dtype)

                for list_idx, probs_grid in enumerate(probs_grids):
                    prototype = (block_mean * probs_grid).sum(dim=(2, 3)).view(channels)
                    prototype_lists[list_idx].append(prototype)

        prototypes = [prototype for prototype_list in prototype_lists for prototype in prototype_list]
        return torch.stack(prototypes, dim=0)

    def _sample_prototype_targets(
        self,
        valid_mask,
        roi_indices,
        height,
        width,
        nw,
        sample_targets,
    ):
        valid_grid = valid_mask[0].to(dtype=torch.bool)

        if self.target_outside_selected_roi:
            selected_tile_mask = self._selected_roi_tile_mask(
                roi_indices=roi_indices,
                height=height,
                width=width,
                nw=nw,
                valid_mask=valid_grid,
            )
            target_tile_mask = valid_grid & ~selected_tile_mask
            if not bool(target_tile_mask.any().item()):
                raise ValueError("Selected ROIs cover every valid tile; no outside reconstruction targets remain")
        else:
            target_tile_mask = valid_grid

        target_y, target_x = torch.nonzero(target_tile_mask, as_tuple=True)
        if target_y.numel() == 0:
            raise ValueError("No valid target tiles available for prototype reconstruction")

        return self._sample_target_tiles(
            target_y=target_y,
            target_x=target_x,
            sample_targets=sample_targets,
        )

    def compute_roi_prototype_reconstruction_loss(
        self,
        x,
        valid_mask,
        roi_probs_list,
        roi_indices,
        nh,
        nw,
        chunk_s=512,
        sample_targets=True,
    ):
        batch, channels, height, width = x.shape
        if batch != 1:
            raise ValueError("Only batch size 1 is supported")
        if not roi_indices:
            raise ValueError("No selected ROI indices available for reconstruction")

        prototypes = self.compute_selected_roi_prototypes(
            x=x,
            valid_mask=valid_mask,
            roi_probs_list=roi_probs_list,
            roi_indices=roi_indices,
            nh=nh,
            nw=nw,
        )
        if prototypes.numel() == 0:
            raise ValueError("No ROI prototypes were constructed")

        target_y, target_x = self._sample_prototype_targets(
            valid_mask=valid_mask,
            roi_indices=roi_indices,
            height=height,
            width=width,
            nw=nw,
            sample_targets=sample_targets,
        )
        targets = x[0, :, target_y, target_x].t().contiguous()

        prototypes_float = prototypes.float()
        prototypes_norm = F.normalize(prototypes_float, dim=-1)
        total_sqerr = torch.zeros((), device=x.device, dtype=torch.float32)
        num_targets = int(targets.shape[0])
        chunk_s = max(1, int(chunk_s))

        for s0 in range(0, num_targets, chunk_s):
            s1 = min(s0 + chunk_s, num_targets)
            target_chunk = targets[s0:s1].float()
            target_norm = F.normalize(target_chunk, dim=-1)
            sim = target_norm @ prototypes_norm.t()
            weights = F.softmax(sim / self.tau_match, dim=-1)
            reconstructed = weights @ prototypes_float
            total_sqerr = total_sqerr + F.mse_loss(
                reconstructed,
                target_chunk,
                reduction="sum",
            )

        return total_sqerr / float(num_targets * channels)

    def forward(
        self,
        x,
        valid_mask=None,
        roi_candidate_mask=None,
        num_rois=1,
        chunk=256,
        reconstruction_mode=None,
        sample_targets=True,
        include_entropy=True,
    ):
        batch, _, height, width = x.shape
        if batch != 1:
            raise ValueError("Only batch size 1 is supported")
        if height < self.roi_tiles or width < self.roi_tiles:
            raise ValueError(
                f"Input region {(height, width)} is smaller than roi_tiles={self.roi_tiles}. "
                "Reduce model.base_roi_tiles or adjust the Visium region."
            )

        if valid_mask is None:
            valid_mask = torch.ones((batch, height, width), device=x.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(0)

        logits_map = self.scorer(x)
        nh, nw = logits_map.shape[2], logits_map.shape[3]

        if roi_candidate_mask is None:
            roi_candidate_mask = self.compute_roi_candidate_mask(valid_mask)
        else:
            roi_candidate_mask = roi_candidate_mask.to(device=x.device, dtype=torch.bool)
            if roi_candidate_mask.ndim == 2:
                roi_candidate_mask = roi_candidate_mask.unsqueeze(0)

        if roi_candidate_mask.shape != (batch, nh, nw):
            raise ValueError(
                f"roi_candidate_mask shape {roi_candidate_mask.shape} does not match expected {(batch, nh, nw)}"
            )

        flat_candidate_mask = roi_candidate_mask.flatten()
        logits = self._stable_selection_logits(logits_map.flatten(), flat_candidate_mask)

        if self.selection_mode == "sequential_gumbel":
            roi_probs_list, roi_indices_debug, selection_entropy = self.select_multiple_rois_soft(
                logits=logits,
                num_rois=num_rois,
                candidate_mask=flat_candidate_mask,
                nh=nh,
                nw=nw,
            )
        elif self.selection_mode == "combinatorial_thretopk":
            roi_probs_list, roi_indices_debug, selection_entropy = self.select_combinatorial_rois_soft(
                logits=logits,
                num_rois=num_rois,
                candidate_mask=flat_candidate_mask,
                nh=nh,
                nw=nw,
            )
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        mode = self.reconstruction_mode if reconstruction_mode is None else str(reconstruction_mode)
        if mode == "hard_nn":
            recon_loss = self.compute_tile_nn_reconstruction_loss(
                x=x,
                valid_mask=valid_mask,
                roi_indices=roi_indices_debug,
                nw=nw,
                chunk_s=chunk,
                sample_targets=sample_targets,
            )
        elif mode == "soft_nn":
            recon_loss = self.compute_tile_soft_nn_reconstruction_loss(
                x=x,
                valid_mask=valid_mask,
                roi_probs_list=roi_probs_list,
                roi_indices=roi_indices_debug,
                nh=nh,
                nw=nw,
                chunk_s=chunk,
                sample_targets=sample_targets,
            )
        elif mode == "roi_prototypes":
            recon_loss = self.compute_roi_prototype_reconstruction_loss(
                x=x,
                valid_mask=valid_mask,
                roi_probs_list=roi_probs_list,
                roi_indices=roi_indices_debug,
                nh=nh,
                nw=nw,
                chunk_s=chunk,
                sample_targets=sample_targets,
            )
        else:
            raise ValueError(f"Unknown reconstruction_mode: {mode}")

        if include_entropy:
            loss = recon_loss - self.select_entropy_weight * selection_entropy + (0.0 * logits.sum())
        else:
            loss = recon_loss
        return loss, roi_indices_debug

    def forward_with_seed(
        self,
        x,
        seed,
        valid_mask=None,
        roi_candidate_mask=None,
        num_rois=1,
        chunk=256,
        reconstruction_mode=None,
        sample_targets=True,
        include_entropy=True,
    ):
        device_index = x.device.index
        devices = [device_index] if x.device.type == "cuda" and device_index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if x.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            return self.forward(
                x=x,
                valid_mask=valid_mask,
                roi_candidate_mask=roi_candidate_mask,
                num_rois=num_rois,
                chunk=chunk,
                reconstruction_mode=reconstruction_mode,
                sample_targets=sample_targets,
                include_entropy=include_entropy,
            )
