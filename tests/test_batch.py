import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rescope.batch import run_budgets
from rescope.demo import create_demo


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = create_demo(self.root / "inputs")
        self.env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                         if k != "ISCOUT_PROJECT_ROOT"}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_all_budgets_have_distinct_trained_checkpoints(self):
        output = self.root / "all"
        with contextlib.redirect_stdout(io.StringIO()):
            run_budgets(self.config, output_dir=output)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["budgets"], [1, 2, 3])
        weight_hashes = []
        for k in (1, 2, 3):
            run = output / f"roi_{k}"
            ckpt = torch.load(run / "best_model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(ckpt["num_rois"], k)
            self.assertEqual(ckpt["config"]["train"]["num_rois"], k)
            self.assertEqual(ckpt["train_seed"], 42)
            params = b"".join(t.cpu().numpy().tobytes() for t in ckpt["model_state_dict"].values())
            weight_hashes.append(hashlib.sha256(params).hexdigest())
            meta = json.loads((run / "visium_hd_region_eval_metadata.json").read_text())
            self.assertEqual(meta["num_rois_selected"], k)
            self.assertEqual(meta["checkpoint_provenance"]["train_num_rois"], k)
            self.assertEqual(Path(meta["checkpoint_provenance"]["path"]), run / "best_model.pt")
        self.assertEqual(len(set(weight_hashes)), 3)
        before = (output / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_budgets(self.config, output_dir=output)
        self.assertEqual((output / "summary.json").read_bytes(), before)

    def test_failed_batch_records_error(self):
        data = json.loads(self.config.read_text())
        data["data"]["embedding_dir"] = "missing"
        self.config.write_text(json.dumps(data))
        output = self.root / "failed"
        with self.assertRaises((FileNotFoundError, ValueError)):
            run_budgets(self.config, output_dir=output)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIn("error", summary)
        self.assertFalse((output / "roi_2").exists())

    def test_batch_validation_is_read_only(self):
        output = self.root / "unused"
        report = run_budgets(self.config, command="validate", output_dir=output)
        self.assertEqual([r["num_rois"] for r in report], [1, 2, 3])
        self.assertFalse(output.exists())
