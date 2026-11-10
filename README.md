# ReSCOPE

ReSCOPE selects representative tissue regions using image embeddings and cell annotations. This repository includes preprocessing, **independent training for 1, 2, and 3 ROIs**, and downstream **BLEEP/iSTAR expression prediction**. Each ROI budget gets its own newly initialized model and matching checkpoint.

```text
H&E image ── UNI2-h ──────────────── embeddings.npy + meta.csv ─┐
          └─ CellViT ── cells.json ── cell_metadata.csv.gz ──────┤
                                                             └─ train/select 1, 2, 3 ROIs
                                                                  └─ BLEEP / iSTAR + measured ROI expression
```

## 1. Set up the environments

For the full pipeline, use **Linux, Python 3.9, Git, and an NVIDIA CUDA GPU**. Run these commands from the repository root:

```bash
PYTHON_BIN=python3.9 bash scripts/setup_preprocessing.sh
```

This creates `.venv-rescope` for UNI2-h/preprocessing/ReSCOPE and `.venv-cellvit` for CellViT, and downloads a pinned CellViT checkout into `third_party/CellViT`. The two environments keep their different NumPy/SciPy requirements separate. Setup requires internet access; it does not download model weights.

To reuse existing environments, install `requirements-preprocess.txt` in the ReSCOPE environment and `requirements-cellvit.txt` in a separate CellViT environment. Set `PYTHON_BIN` when calling `run.sh`, and set `cellvit_python` in the pipeline config.

## 2. Prepare the image and model weights

You need an H&E image (`.tif`, `.tiff`, `.btf`, or an OpenSlide-supported WSI), its **actual microns per pixel (`mpp`)**, and two pretrained models:

| Input | Default location | Source |
|---|---|---|
| Tissue image | `data/my_sample/image.btf` | Your tissue image |
| UNI2-h checkpoint | `weights/uni2-h/pytorch_model.bin` | [UNI2-h model page](https://huggingface.co/MahmoodLab/UNI2-h) |
| CellViT-SAM-H x40 checkpoint | `weights/cellvit/CellViT-SAM-H-x40.pth` | [CellViT checkpoints](https://github.com/TIO-IKIM/CellViT#inference) |

Request access to UNI2-h on its model page and accept its terms. Once approved:

```bash
.venv-rescope/bin/huggingface-cli login
.venv-rescope/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('MahmoodLab/UNI2-h', 'pytorch_model.bin', local_dir='weights/uni2-h')"
.venv-rescope/bin/gdown 1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV -O weights/cellvit/CellViT-SAM-H-x40.pth
```

The CellViT checkpoint must match the image magnification. For x20 images, obtain the x20 checkpoint from the same official page and change the checkpoint path and `magnification` below. Weights and full-slide datasets are downloaded separately and are excluded from Git.

## 3. Configure one sample

Edit `examples/pipeline.yaml`. Set `wsi_path`, `mpp`, `magnification`, checkpoint locations, and `work_dir`. All paths are relative to this config file. Adjust `base_roi_tiles` if you need a different ROI size; the default is 164 embedding tiles per side.

For the Visium HD cell population used by the paper workflow, also set `hd_square_002um_dir` to the matched `square_002um` directory containing `filtered_feature_bc_matrix.h5` and `spatial/tissue_positions.parquet`. The cell step then assigns 2 µm bins to nuclei and retains cells with assigned bins. Leave it `null` for H&E-only analysis, which uses all detected nuclei inside the image. These are different input cell populations.

## 4. Run all steps with one command

```bash
bash run.sh pipeline --config examples/pipeline.yaml
```

The pipeline extracts embeddings, prepares CellViT patches, detects and classifies cells, writes the cell table and ReSCOPE config, then trains and selects **1, 2, and 3 ROIs separately**. Embedding and segmentation progress is recorded in `embed.log` and `segment.log` inside `work_dir`.

To inspect inputs first:

```bash
bash run.sh pipeline --config examples/pipeline.yaml --stage check
```

To run one step at a time, use the following commands **in order**:

| Step | Command | Reads → writes |
|---|---|---|
| Extract UNI2-h features | `bash run.sh pipeline --config examples/pipeline.yaml --stage embed` | Image + UNI2-h weights → `embeddings/` |
| Prepare CellViT image patches | `bash run.sh pipeline --config examples/pipeline.yaml --stage patches` | Image → `patches/` |
| Segment and classify nuclei | `bash run.sh pipeline --config examples/pipeline.yaml --stage segment` | Patches + CellViT weights → `patches/<image-stem>/cell_detection/rescope/cells.json` |
| Prepare the cell table | `bash run.sh pipeline --config examples/pipeline.yaml --stage cells` | `cells.json` (+ optional HD bins) → `cells/cell_metadata.csv.gz` |
| Prepare model inputs | `bash run.sh pipeline --config examples/pipeline.yaml --stage config` | Embedding manifest + cell table → `rescope.yaml` |
| Train and select ROIs | `bash run.sh pipeline --config examples/pipeline.yaml --stage train` | `rescope.yaml` → `results/roi_1`, `roi_2`, `roi_3` |

Step implementations are in `src/rescope/preprocessing/`: `uni.py`, `patches.py`, `cellvit.py`, `cells.py`, and `pipeline.py`; optional HD aggregation is in `hd_cells.py`. Each step records completion or failure. Use a fresh `work_dir` for a full rerun; existing outputs are protected. Whole-slide memory and disk needs depend on image size; lower the batch sizes if needed.

## 5. Find the results

```text
runs/my_sample/
├── embeddings/             # embeddings.npy, meta.csv, manifest.json
├── patches/                # CellViT input patches and cells.json
├── cells/                  # cell_metadata.csv.gz; HD count files when requested
├── rescope.yaml            # automatically generated training configuration
└── results/
    ├── roi_1/              # independent 1-ROI model and selection
    ├── roi_2/              # independent 2-ROI model and selection
    ├── roi_3/              # independent 3-ROI model and selection
    └── summary.json
```

Each ROI directory contains `best_model.pt`, `training_history.csv`, `rois.csv`, `rois.geojson`, and configuration/provenance JSON files. ROI and cell coordinates are full-resolution **x,y pixels**. Generated configs use the extractor's actual integer pixel spacing and full grid dimensions, including filtered background tiles.

## 6. Run downstream BLEEP and iSTAR

These workflows train on measured expression from cells **inside the selected ROI union**, predict cells outside it, and evaluate against their held-out counts. Cell-type annotations alone are insufficient: use the four count/metadata files produced by the optional Visium HD aggregation step, or equivalent aligned cell-level expression data.

```bash
PYTHON_BIN=python3.9 bash scripts/setup_downstream.sh
# Follow docs/downstream.md to obtain official weights and configure examples/downstream.yaml.
bash run.sh downstream --config examples/downstream.yaml
```

This runs both methods independently for all three ROI budgets. For one method/budget, append `--method bleep --num-rois 1`. The [downstream guide](docs/downstream.md) gives the official repository and weight links, required inputs, step-by-step commands, and output locations. Internal/data-trained checkpoints are not distributed; each user trains those models on their own ROI data.

To include downstream prediction in the original full-pipeline command, configure both files and set `downstream_config: downstream.yaml` in `examples/pipeline.yaml`. Then `bash run.sh pipeline --config examples/pipeline.yaml` runs from the image through ROI selection and both downstream methods. Measured cell counts and the predefined gene panel are required.

## Existing embeddings or a quick demo

If you already have embeddings and a cell table, edit `examples/config.yaml` and run:

```bash
bash run.sh --config examples/config.yaml --output-dir ./runs/precomputed
```

`embeddings.npy` must be a float array `[tiles, features]`; `meta.csv` must contain matching zero-based `row,col` coordinates. `cell_metadata.csv.gz` requires `centroid_x,centroid_y,cell_type` with CellViT codes (0 unknown, 1 tumor, 2 inflammatory, 3 connective, 4 dead, 5 epithelial). When available, set `data.tile_px` and `data.grid_shape` from the extraction manifest. Append `--num-rois 1`, `2`, or `3` to run one budget.

Without real data or weights, check ROI training/selection on small synthetic CPU inputs:

```bash
bash run.sh demo --output-dir ./runs/demo
```

The demo checks ROI selection only and is not a biological benchmark. New training records its seeds; it does not recreate archived checkpoint weights. The ROI scorer's best checkpoint uses training loss, and its evaluation loss is a same-sample diagnostic.

Third-party sources and licenses: [THIRD_PARTY.md](THIRD_PARTY.md). Developer tests after preprocessing setup: `PYTHONPATH=src .venv-rescope/bin/python -m unittest discover -s tests -v`.
