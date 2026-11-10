# BLEEP and iSTAR downstream prediction

Run this after ReSCOPE has produced `results/roi_1`, `roi_2`, and `roi_3`. Each budget defines a separate reference set: the union of its selected ROIs. BLEEP and iSTAR each train on that reference expression and predict the remaining cells. This is the paper's cell-level adaptation of the two methods, using the upstream model implementations.

## 1. Set up code and environments

From the ReSCOPE repository root, on Linux with Python 3.9:

```bash
PYTHON_BIN=python3.9 bash scripts/setup_downstream.sh
```

This creates `.venv-bleep` and `.venv-istar`, downloads the official repositories into `third_party/`, and applies the supplied iSTAR training patch. iSTAR uses its original PyTorch/Lightning environment, separate from ReSCOPE and BLEEP. Setup downloads **code and dependencies, not model checkpoints**. With existing environments, install the matching `requirements-bleep.txt` / `requirements-istar.txt`, run `python scripts/setup_downstream_repos.py`, and set their Python paths in the config.

## 2. Obtain official pretrained weights

| Component | Official source | How it is used |
|---|---|---|
| BLEEP | [bowang-lab/BLEEP](https://github.com/bowang-lab/BLEEP) | Its ResNet50 encoder loads public ImageNet weights through [timm](https://github.com/huggingface/pytorch-image-models). The contrastive model is then trained on your ROI image/expression pairs. |
| iSTAR | [daviddaiweizhang/istar](https://github.com/daviddaiweizhang/istar), [official checkpoint download script](https://github.com/daviddaiweizhang/istar/blob/3cb0e5352a86df337f41c5841d0808da0003457b/download_checkpoints.sh) | HIPT ViT-256 and ViT-4K image encoders; the expression predictor is trained on your ROI counts. |

Run the following explicit download commands once, with internet access:

```bash
# Download/cache the public ResNet50 weights used by BLEEP.
.venv-bleep/bin/python -c "import timm; timm.create_model('resnet50', pretrained=True)"

# Download the two public HIPT checkpoints from the locations specified by iSTAR.
(cd third_party/istar && bash download_checkpoints.sh)
```

The second command requires `wget` and creates `third_party/istar/checkpoints/vit256_small_dino.pth` and `vit4k_xs_dino.pth`. Keep those names and locations. BLEEP's weights use the normal timm/Hugging Face cache; download and run under the same account/cache location. For offline machines, copy the public cache and HIPT files from a machine where these commands succeeded.

**No internal checkpoints are required or provided.** BLEEP's `best.pt` and iSTAR's regression checkpoints are outputs of training on your data, not a universal pretrained model to download. Do not substitute a model trained with a different gene panel. UNI2-h and CellViT downloads remain documented in the main README. All model-weight and generated-output directories are excluded from Git; upstream code and weight licenses apply separately.

## 3. Prepare inputs and configure paths

Edit `examples/downstream.yaml`. Paths are relative to the config file.

| Config field | Input |
|---|---|
| `histology_image` | The same full-resolution H&E image used for ROI selection; TIFF/BigTIFF is recommended for these adapters. |
| `roi_results_dir` | Directory containing `roi_1`, `roi_2`, `roi_3`, each with `visium_hd_region_eval_metadata.json`. |
| `single_cell_dir` | `cell_gene_counts.npz` (sparse cells × genes), `cell_ids.tsv.gz`, `features.tsv.gz` (gene ID, unique gene name, feature type), and `cell_metadata.csv.gz`. |
| `gene_list` | A predefined panel with one unique gene name per line, matching `features.tsv.gz`. Use the same panel for both methods and all budgets; the paper experiments used a fixed 2,000-gene panel. |
| `work_dir` | A new directory for downstream outputs. |

The preprocessing pipeline produces all four cell files when `hd_square_002um_dir` is set. With H&E-only inputs it produces annotations only, so you must supply matched measured expression before running downstream training/evaluation. The benchmark needs counts for outside-ROI cells for evaluation; these counts are not passed to either training process. Select the gene panel in advance or from an independent reference, rather than choosing genes based on held-out prediction performance.

Matrix rows must match `cell_ids.tsv.gz`; columns must match `features.tsv.gz`. Metadata needs unique `cell_id` values and `centroid_x,centroid_y` in full-resolution x,y pixels. iSTAR also needs `polygon_area_fullres` to estimate a cell radius, or an explicit `istar.radius_raw` in pixels. The adapter uses the supplied axes as written. The `pixel_size_raw` / `pixel_size_target` ratio controls iSTAR rescaling; the example retains the paper wrapper's ratio of 1, with automatic downscaling above 8192 pixels.

## 4. Run

After configuring inputs and downloading weights:

```bash
bash run.sh downstream --config examples/downstream.yaml --stage check
bash run.sh downstream --config examples/downstream.yaml
```

The full command prepares reference/query splits, trains and predicts with BLEEP and iSTAR separately for 1/2/3 ROIs, and writes comparison tables. For one method or budget, append `--method bleep` or `--method istar`, and/or `--num-rois 1` (also supports 2 and 3).

Alternatively, run each stage **once, in order**, using a fresh `work_dir`:

| Stage | Command suffix after `bash run.sh downstream --config examples/downstream.yaml` | Reads → writes |
|---|---|---|
| Split cells | `--stage prepare` | Selected ROI rectangles + cell counts → inside reference / outside query files |
| BLEEP | `--stage bleep` | Reference counts + image + ResNet50 weights → newly trained BLEEP model, outside-cell predictions, metrics |
| iSTAR | `--stage istar` | Reference counts + image + HIPT weights → newly trained iSTAR models, spatial expression maps, outside-cell predictions, metrics |
| Compare | `--stage summarize` | Completed predictions → per-run and common-query comparison tables |

`--dry-run` prints the planned commands without running or writing outputs. Training can require substantial RAM, disk and GPU time for full slides. BLEEP uses 20 epochs and iSTAR 50 epochs by default, matching the paper workflow. Use `bleep.num_workers: 0` if worker multiprocessing is unavailable. A failed stage is recorded with its error; inspect its log and use a new `work_dir` for a full rerun.

## 5. Read outputs

```text
downstream/
├── roi_1/                  # roi_2 and roi_3 have the same structure
│   ├── prepared/roi_1/     # reference/query count tables and ROI manifest
│   ├── bleep/roi_1/        # bleep_run/, bleep_prediction/, bleep_eval/
│   ├── istar/roi_1/        # istar_run/, istar_eval/
│   ├── bleep.log
│   └── istar.log
├── summary.csv             # each run's own outside-ROI query set
└── common_query_summary.csv # the same query cells across all requested runs
```

Each `*_eval/` contains `predictions_and_truth.npz`, `metrics.json`, and `per_gene_mse.csv`. NPZ arrays are `predicted`, `truth`, `query_cell_ids`, and `gene_names`. Metrics are raw-count MSE and log1p-count MSE. iSTAR writes full spatial maps under `istar_run/cnts-super-all/`. Cells outside its output grid or without finite predictions are excluded and counted in its metrics. Common-query scores use the intersection of retained query cells and the same genes; an empty intersection is reported explicitly.

BLEEP selects its checkpoint using a validation split within the reference ROI cells. iSTAR uses the supplied paper-workflow patch to retain the lowest training-loss checkpoint. Outside query expression is used only for evaluation. Source provenance and the iSTAR patch/license are listed in [THIRD_PARTY.md](../THIRD_PARTY.md).
