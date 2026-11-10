import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from rescope import ROI, RunResult, run_pipeline, select, train, validate
from rescope.config import load_config
from rescope.demo import create_demo


class PythonApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="iscout_api_test_")
        self.root = Path(self.temp.name)
        self.path = create_demo(self.root / "sample with spaces")
        self.config = json.loads(self.path.read_text())
        self.env = patch.dict(os.environ, {key: value for key, value in os.environ.items()
                                         if key != "ISCOUT_PROJECT_ROOT"}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_mapping_pipeline_returns_results_without_cli_or_mutating_input(self):
        self.config["data"]["embedding_dir"] = Path("embeddings")
        self.config["selection"]["reference_fraction_retry_multipliers"] = [1.0, 1.2, 1.5, 0.0]
        before = copy.deepcopy(self.config)
        with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", side_effect=AssertionError("API prompted")), \
                patch.object(sys, "argv", ["caller.py", "--unrelated-option"]):
            result = run_pipeline(self.config, num_rois=3, base_dir=self.path.parent,
                                  output_dir=self.root / "pipeline")
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.num_rois, 3)
        self.assertEqual(len(result.rois), 3)
        self.assertTrue(all(isinstance(roi, ROI) and roi.left_px < roi.right_px for roi in result.rois))
        self.assertTrue(result.checkpoint.is_file())
        self.assertTrue(result.csv_path.is_file())
        self.assertTrue(result.geojson_path.is_file())
        self.assertTrue(result.history_path.is_file())
        self.assertEqual(result.manifest["config_source"], "mapping")
        self.assertIsNone(result.manifest["source_config"])
        self.assertEqual(self.config, before)
        result.config["selection"]["reference_fraction_retry_multipliers"].append(99)
        self.assertEqual(self.config, before)

    def test_train_then_select_reuses_explicit_checkpoint(self):
        training_config = copy.deepcopy(self.config)
        del training_config["data"]["single_cell_dir"]
        with contextlib.redirect_stdout(io.StringIO()):
            trained = train(training_config, num_rois=3, epochs=1, base_dir=self.path.parent,
                            output_dir=self.root / "trained")
            before = trained.checkpoint.stat().st_mtime_ns
            selected = select(self.config, num_rois=3, checkpoint=trained.checkpoint,
                              base_dir=self.path.parent, output_dir=self.root / "selected")
        self.assertEqual(trained.rois, ())
        self.assertIsNone(trained.metadata)
        self.assertIsNone(trained.csv_path)
        self.assertIsNone(trained.geojson_path)
        self.assertEqual(selected.checkpoint, trained.checkpoint)
        self.assertEqual(before, selected.checkpoint.stat().st_mtime_ns)
        self.assertEqual(len(selected.rois), 3)
        self.assertIsNone(selected.history_path)
        with self.assertRaisesRegex(ValueError, "Checkpoint trained"):
            select(self.config, num_rois=2, checkpoint=trained.checkpoint,
                   base_dir=self.path.parent, output_dir=self.root / "bad-budget")
        self.assertFalse((self.root / "bad-budget").exists())

    def test_validate_file_mapping_and_train_only(self):
        file_report = validate(self.path, num_rois=3)
        mapping_report = validate(MappingProxyType(self.config), num_rois=3, base_dir=self.path.parent)
        self.assertEqual(file_report, mapping_report)
        del self.config["data"]["single_cell_dir"]
        report = validate(self.config, num_rois=3, stage="train", base_dir=self.path.parent)
        self.assertEqual(report["stage"], "train")
        with self.assertRaisesRegex(ValueError, "single_cell_dir"):
            validate(self.config, num_rois=3, stage="select", base_dir=self.path.parent)
        with self.assertRaisesRegex(ValueError, "validate_for"):
            validate(self.config, num_rois=3, stage="typo", base_dir=self.path.parent)
        with self.assertRaisesRegex(ValueError, "base_dir"):
            validate(self.path, num_rois=3, base_dir=self.path.parent)

    def test_exceptions_are_python_exceptions_and_no_input(self):
        with patch("builtins.input", side_effect=AssertionError("API prompted")):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                train(self.config, num_rois=0, base_dir=self.path.parent, output_dir=self.root / "bad")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run_pipeline(self.config, base_dir=self.path.parent, output_dir=self.root / "bad")
            with self.assertRaises(FileNotFoundError):
                select(self.path, num_rois=1, checkpoint=self.root / "missing.pt", output_dir=self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_import_does_not_load_model_dependencies(self):
        package_root = Path(__file__).resolve().parents[1] / "src"
        code = f"import sys; sys.path.insert(0, {str(package_root)!r}); import rescope; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        proc = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
