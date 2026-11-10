"""Public, non-interactive Python API.

Inputs may be YAML/JSON paths or Python mappings. These functions never parse
sys.argv, prompt for input, invoke shell commands, or install dependencies.
They raise normal Python exceptions, and write only to a fresh output directory.
Model computation currently prints progress to stdout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

Config = Union[str, PathLike, Mapping[str, Any]]


@dataclass(frozen=True)
class ROI:
    """One selected square in full-resolution image pixels (x right, y down)."""

    name: str
    index: int
    left_px: float
    top_px: float
    right_px: float
    bottom_px: float


@dataclass(frozen=True)
class RunResult:
    """Outputs shared by train, select and run_pipeline.

    rois is empty for train; metadata/csv/geojson paths are None for train.
    checkpoint points at the trained or explicitly supplied model.
    history_path is None for select, which does not train a model.
    """

    output_dir: Path
    checkpoint: Path
    num_rois: int
    rois: Tuple[ROI, ...]
    manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    metadata: Optional[Mapping[str, Any]]
    csv_path: Optional[Path]
    geojson_path: Optional[Path]
    history_path: Optional[Path]


def _read_result(output_dir: Path) -> RunResult:
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    config = json.loads((output_dir / "resolved_config.json").read_text())
    metadata_path = output_dir / "visium_hd_region_eval_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    rois = () if metadata is None else tuple(
        ROI(name=box["roi_name"], index=index, left_px=box["left_px"], top_px=box["top_px"],
            right_px=box["right_px"], bottom_px=box["bottom_px"])
        for index, box in zip(metadata["model_roi_indices"], metadata["model_rois"])
    )
    return RunResult(
        output_dir=output_dir, checkpoint=Path(manifest["checkpoint"]["path"]),
        num_rois=manifest["num_rois"], rois=rois, manifest=manifest, config=config,
        metadata=metadata, csv_path=output_dir / "rois.csv" if metadata else None,
        geojson_path=output_dir / "rois.geojson" if metadata else None,
        history_path=output_dir / "training_history.csv" if (output_dir / "training_history.csv").is_file() else None,
    )


def train(config: Config, *, num_rois: Optional[int] = None, output_dir=None,
          device: Optional[str] = None, epochs: Optional[int] = None, base_dir=None) -> RunResult:
    """Train with N ROIs and return the checkpoint and training artifacts.

    A cell table is not required. num_rois overrides train.num_rois in the input.
    Mapping-relative paths use base_dir (or cwd); file-relative paths use the
    config file's directory. Existing output directories are never overwritten.
    """
    from .workflow import execute
    return _read_result(execute(config, command="train", num_rois=num_rois,
                                output_dir=output_dir, device=device, epochs=epochs, base_dir=base_dir))


def select(config: Config, *, checkpoint: Union[str, PathLike], num_rois: Optional[int] = None,
           output_dir=None, device: Optional[str] = None, base_dir=None) -> RunResult:
    """Select N ROIs using a checkpoint; never retrain or auto-select a weight file.

    Requires embeddings and annotated cell metadata. A different checkpoint
    training budget raises ValueError.
    Explicit checkpoint/output_dir paths resolve from cwd.
    """
    from .workflow import execute
    return _read_result(execute(config, command="select", checkpoint=checkpoint,
                                num_rois=num_rois, output_dir=output_dir, device=device, base_dir=base_dir))


def run_pipeline(config: Config, *, num_rois: Optional[int] = None, output_dir=None,
                 device: Optional[str] = None, epochs: Optional[int] = None, base_dir=None) -> RunResult:
    """Train with N ROIs, then select N ROIs using that freshly trained checkpoint."""
    from .workflow import execute
    return _read_result(execute(config, command="run", num_rois=num_rois, output_dir=output_dir,
                                device=device, epochs=epochs, base_dir=base_dir))


def validate(config: Config, *, num_rois: Optional[int] = None, device: Optional[str] = None,
             base_dir=None, stage: str = "run") -> dict:
    """Check config/data without writing outputs. stage is train, select or run.

    stage='train' skips cell metadata requirements. This checks candidate count
    and area, not final combination feasibility or checkpoint compatibility.
    """
    from .workflow import execute
    return execute(config, command="validate", num_rois=num_rois, device=device,
                   base_dir=base_dir, validate_for=stage)
