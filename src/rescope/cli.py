from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import __version__


def positive_count(text):
    try:
        value = int(text)
        if value < 1:
            raise ValueError()
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a positive integer") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description="ReSCOPE: independently train and select 1, 2, and 3 ROIs")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "train", "select", "validate"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--num-rois", type=int, choices=(1, 2, 3), required=command == "select",
                       help="one budget; omitted = independently run all three budgets")
        p.add_argument("--output-dir", type=Path)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"))
        if command in {"run", "train"}:
            p.add_argument("--epochs", type=positive_count)
        if command == "select":
            p.add_argument("--checkpoint", type=Path, required=True)
    demo = sub.add_parser("demo", help="run all three budgets on small synthetic CPU inputs")
    demo.add_argument("--output-dir", type=Path)
    pipeline = sub.add_parser("pipeline", help="WSI -> embeddings + cells -> independent ROI training")
    pipeline.add_argument("--config", type=Path, required=True)
    pipeline.add_argument("--stage", choices=("all", "check", "embed", "patches", "segment", "cells", "config", "train", "downstream"), default="all")
    pipeline.add_argument("--dry-run", action="store_true")
    downstream = sub.add_parser("downstream", help="selected ROIs -> BLEEP/iSTAR expression prediction")
    downstream.add_argument("--config", type=Path, required=True)
    downstream.add_argument("--stage", choices=("all", "check", "prepare", "bleep", "istar", "summarize"), default="all")
    downstream.add_argument("--num-rois", type=int, choices=(1, 2, 3))
    downstream.add_argument("--method", choices=("both", "bleep", "istar"), default="both")
    downstream.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        from .batch import run_budgets
        if args.command == "downstream":
            from .downstream.pipeline import run_downstream
            result = run_downstream(args.config, stage=args.stage,
                                    budgets=(args.num_rois,) if args.num_rois else (1, 2, 3),
                                    methods=("bleep", "istar") if args.method == "both" else (args.method,),
                                    dry_run=args.dry_run)
        elif args.command == "pipeline":
            from .preprocessing.pipeline import run_pipeline as preprocess
            result = preprocess(args.config, args.stage, args.dry_run)
        elif args.command == "demo":
            from .demo import create_demo
            scratch = Path(os.environ.get("ISCOUT_SCRATCH_ROOT", Path.cwd()))
            directory = args.output_dir or scratch / "runs" / f"demo_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
            project = os.environ.pop("ISCOUT_PROJECT_ROOT", None)
            try:
                # Synthetic test fixtures can live in the temporary work directory.
                config = create_demo(directory)
                result = run_budgets(config, output_dir=directory / "results", device="cpu")
            finally:
                if project is not None:
                    os.environ["ISCOUT_PROJECT_ROOT"] = project
        elif args.command == "select":
            from .workflow import execute
            result = execute(args.config, command="select", num_rois=args.num_rois,
                             checkpoint=args.checkpoint, output_dir=args.output_dir, device=args.device)
        else:
            result = run_budgets(args.config, command=args.command,
                                 budgets=(args.num_rois,) if args.num_rois else (1, 2, 3),
                                 output_dir=args.output_dir, device=args.device,
                                 epochs=getattr(args, "epochs", None))
        print(json.dumps(result, indent=2) if isinstance(result, list) else f"Completed: {result}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"rescope: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("rescope: interrupted", file=sys.stderr)
        return 130
