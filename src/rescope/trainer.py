import torch

try:
    from torch.amp import autocast as _autocast
except ImportError:
    from torch.cuda.amp import autocast as _autocast

try:
    from torch.amp import GradScaler as _GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler


def _make_grad_scaler(enabled):
    try:
        return _GradScaler(device="cuda", enabled=enabled)
    except TypeError:
        return _GradScaler(enabled=enabled)


def _cuda_autocast(enabled):
    try:
        return _autocast(device_type="cuda", enabled=enabled)
    except TypeError:
        return _autocast(enabled=enabled)


class Trainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.opt = optimizer
        self.cfg = config
        amp_on = bool(config["train"].get("amp", False))
        self.scaler = _make_grad_scaler(enabled=amp_on)

    def train_epoch(self, loader, device):
        self.model.train()
        total_loss = 0.0
        amp_on = bool(self.cfg["train"].get("amp", False))

        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)

            with _cuda_autocast(enabled=amp_on):
                loss, _ = self.model(
                    x,
                    valid_mask=valid_mask,
                    num_rois=self.cfg["train"].get("num_rois", 1),
                    chunk=self.cfg["train"]["chunk_size"],
                    sample_targets=True,
                    include_entropy=True,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            total_loss += float(loss.detach().item())

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def eval_epoch(self, loader, device, seed):
        self.model.eval()
        total_loss = 0.0
        amp_on = bool(self.cfg["train"].get("amp", False))
        eval_cfg = self.cfg.get("eval", {})
        reconstruction_mode = eval_cfg.get("reconstruction_mode", "roi_prototypes")
        sample_targets = bool(eval_cfg.get("sample_targets", False))

        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)

            with _cuda_autocast(enabled=amp_on):
                loss, _ = self.model.forward_with_seed(
                    x,
                    seed=seed,
                    valid_mask=valid_mask,
                    num_rois=self.cfg["train"].get("num_rois", 1),
                    chunk=self.cfg["train"]["chunk_size"],
                    reconstruction_mode=reconstruction_mode,
                    sample_targets=sample_targets,
                    include_entropy=False,
                )

            total_loss += float(loss.detach().item())

        return total_loss / max(len(loader), 1)
