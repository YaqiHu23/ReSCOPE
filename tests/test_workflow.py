import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rescope.config import load_config
from rescope.demo import create_demo
from rescope.model import PartTileROIDictionary
from rescope.workflow import execute


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="iscout_test_")
        self.root = Path(self.temp.name)
        self.config = create_demo(self.root / "input with spaces")
        self.env = patch.dict(os.environ, {key: value for key, value in os.environ.items()
                                         if key != "ISCOUT_PROJECT_ROOT"}, clear=True)
        self.env.start()
        torch.set_num_threads(1)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_train_and_select_budgets(self):
        for k in (1, 2, 3):
            with self.subTest(num_rois=k), contextlib.redirect_stdout(io.StringIO()):
                output = execute(self.config, num_rois=k, output_dir=self.root / f"run{k}")
                checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["num_rois"], k)
                self.assertEqual(checkpoint["config"]["train"]["num_rois"], k)
                self.assertTrue(np.isfinite(checkpoint["loss"]))
                self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
                metadata = json.loads((output / "visium_hd_region_eval_metadata.json").read_text())
                self.assertEqual(len(metadata["model_rois"]), k)
                self.assertEqual(len(set(metadata["model_roi_indices"])), k)
                self.assertEqual(metadata["checkpoint_provenance"]["train_num_rois"], k)
                for i, a in enumerate(metadata["model_rois"]):
                    for b in metadata["model_rois"][i + 1:]:
                        self.assertTrue(a["right_px"] <= b["left_px"] or b["right_px"] <= a["left_px"] or
                                        a["bottom_px"] <= b["top_px"] or b["bottom_px"] <= a["top_px"])
                self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"], "completed")
                self.assertEqual(len(json.loads((output / "rois.geojson").read_text())["features"]), k)
                with (output / "rois.csv").open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), k)

    def test_invalid_counts_and_override(self):
        for k in (None, 0, -1, 1.5, True, "3", 4, 5):
            with self.subTest(count=k), self.assertRaises(ValueError):
                load_config(self.config, num_rois=k)
        cfg = json.loads(self.config.read_text())
        cfg["train"]["num_rois"] = 2
        self.config.write_text(json.dumps(cfg))
        self.assertEqual(load_config(self.config)["train"]["num_rois"], 2)
        self.assertEqual(load_config(self.config, num_rois=3)["train"]["num_rois"], 3)

    def test_no_silent_underselection(self):
        model = PartTileROIDictionary(8, roi_tiles=2, stride_tiles=1)
        logits = torch.tensor([0., 0., 0., 0., 10., 0., 0., 0., 0.])
        mask = torch.ones(9, dtype=torch.bool)
        model.eval()
        with self.assertRaisesRegex(ValueError, "selected 1 of 2"):
            model.select_combinatorial_rois_soft(logits, 2, mask, 3, 3)
        with self.assertRaisesRegex(ValueError, "only 9 valid"):
            model.select_combinatorial_rois_soft(logits, 10, mask, 3, 3)

    def test_budget_mismatch_and_no_overwrite(self):
        with contextlib.redirect_stdout(io.StringIO()):
            trained = execute(self.config, command="train", num_rois=2, output_dir=self.root / "trained")
            with self.assertRaisesRegex(ValueError, "Checkpoint trained"):
                execute(self.config, command="select", num_rois=3, checkpoint=trained / "best_model.pt",
                        output_dir=self.root / "mismatch")
            self.assertFalse((self.root / "mismatch").exists())
            with self.assertRaises(FileExistsError):
                execute(self.config, command="train", num_rois=2, output_dir=trained)

    def test_invalid_data_and_budget_fail_before_output(self):
        with self.assertRaisesRegex(ValueError, "supports num_rois"):
            execute(self.config, num_rois=99, output_dir=self.root / "bad")
        self.assertFalse((self.root / "bad").exists())
        features = self.config.parent / "embeddings/embeddings.npy"
        x = np.load(features)
        x[0, 0] = np.nan
        np.save(features, x)
        with self.assertRaisesRegex(ValueError, "NaN"):
            execute(self.config, num_rois=1, output_dir=self.root / "bad")

    def test_validation_is_read_only(self):
        output = self.root / "unused"
        result = execute(self.config, command="validate", num_rois=3, output_dir=output)
        self.assertEqual(result["solver"], "exact")
        self.assertFalse(output.exists())

    def test_metadata_requires_cell_types(self):
        import pandas as pd
        cell_path = self.config.parent / "cells/cell_metadata.csv.gz"
        data = pd.read_csv(cell_path).drop(columns="cell_type")
        data.to_csv(cell_path, index=False)
        with self.assertRaisesRegex(KeyError, "cell_type"):
            execute(self.config, command="validate", num_rois=1)

    def test_relative_paths_and_cli_from_another_cwd(self):
        cfg = load_config(self.config, num_rois=1)
        self.assertEqual(cfg["data"]["embedding_dir"], str(self.config.parent / "embeddings"))
        from rescope.cli import main
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = main(["validate", "--config", str(self.config), "--num-rois", "3"])
            self.assertEqual(result, 0, stderr.getvalue())
            self.assertEqual(json.loads(stdout.getvalue())[0]["num_rois"], 3)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["train", "--config", str(self.config), "--num-rois", "-2"])
            self.assertEqual(error.exception.code, 2)
        finally:
            os.chdir(previous)



if __name__ == "__main__":
    unittest.main()
