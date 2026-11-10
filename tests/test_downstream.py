import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
import yaml

from rescope.downstream.bleep_expression import aggregate_predictions, rebuild_reference_matrix
from rescope.downstream.common import compute_metrics, load_single_cell_inputs, load_truth
from rescope.downstream.istar import compute_query_locs
from rescope.downstream.pipeline import command, load_downstream, run_downstream, summarize
from rescope.downstream.prepare import prepare, read_rois, roi_mask, save_subset


class DownstreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rescope_downstream_")
        self.root = Path(self.temp.name)
        self.ids = [f"{i:03d}" for i in range(20)]
        self.counts = np.arange(60).reshape(20, 3).astype(np.float32)
        self.features = pd.DataFrame({"gene_id": ["id1", "id2", "id3"],
                                      "gene_name": ["G1", "G2", "G3"], "feature_type": ["Gene Expression"] * 3})
        self.cells = pd.DataFrame({"cell_id": self.ids, "centroid_x": np.arange(1, 21),
                                   "centroid_y": [2] * 20, "polygon_area_fullres": [10] * 20})
        save_subset(self.root / "cells", sparse.csr_matrix(self.counts), self.ids, self.features, self.cells)
        self.rois = [{"left_px": left, "right_px": right, "top_px": 0, "bottom_px": 4}
                     for left, right in [(0, 4), (8, 12), (12, 16)]]
        for k in (1, 2, 3):
            path = self.root / f"results/roi_{k}/visium_hd_region_eval_metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"model_rois": self.rois[:k], "checkpoint_provenance": {"train_num_rois": k}}))
        (self.root / "image.tiff").touch()
        (self.root / "genes.txt").write_text("G3\nG1\n")
        self.config = self.root / "downstream.yaml"
        self.config.write_text(yaml.safe_dump({
            "histology_image": "image.tiff", "single_cell_dir": "cells", "roi_results_dir": "results",
            "gene_list": "genes.txt", "work_dir": "outputs", "bleep_repo": "BLEEP", "istar_repo": "istar",
            "bleep_python": "env-bleep/bin/python", "istar_python": "env-istar/bin/python", "device": "cpu"}))
        self.cfg = load_downstream(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_all_budgets_uses_roi_union_not_bounding_box(self):
        run_downstream(self.config, stage="prepare")
        for k in (1, 2, 3):
            root = self.root / f"outputs/roi_{k}/prepared/roi_{k}"
            ref, ref_ids, _, ref_meta = load_single_cell_inputs(root / "reference_single_cell")
            query, query_ids, _, _ = load_single_cell_inputs(root / "query_single_cell")
            self.assertFalse(set(ref_ids) & set(query_ids))
            self.assertEqual(set(ref_ids) | set(query_ids), set(self.ids))
            expected = roi_mask(self.rois[:k], self.cells[["centroid_x", "centroid_y"]].to_numpy())
            np.testing.assert_array_equal(ref.toarray(), self.counts[expected])
            np.testing.assert_array_equal(query.toarray(), self.counts[~expected])
            self.assertNotIn("003", ref_ids)  # x=4 is the exclusive edge, not inside the gap.
            self.assertEqual(ref_meta.cell_id.tolist(), ref_ids)
        with self.assertRaises(FileExistsError):
            run_downstream(self.config, stage="prepare")

    def test_string_ids_and_shuffled_metadata_align_to_matrix(self):
        self.cells.iloc[::-1].to_csv(self.root / "cells/cell_metadata.csv.gz", index=False)
        _, ids, _, meta = load_single_cell_inputs(self.root / "cells")
        self.assertEqual(ids[0], "000")
        self.assertEqual(meta.centroid_x.tolist(), list(range(1, 21)))
        truth = load_truth(self.root / "cells", ["019", "000"], ["G3", "G1"])
        np.testing.assert_array_equal(truth, self.counts[[19, 0]][:, [2, 0]])

    def test_invalid_counts_and_duplicate_gene_names_are_rejected(self):
        bad = sparse.csr_matrix(self.counts)
        bad[0, 1] = -1
        sparse.save_npz(self.root / "cells/cell_gene_counts.npz", bad)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_single_cell_inputs(self.root / "cells")
        sparse.save_npz(self.root / "cells/cell_gene_counts.npz", sparse.csr_matrix(self.counts))
        self.features.loc[1, "gene_name"] = "G1"
        self.features.to_csv(self.root / "cells/features.tsv.gz", sep="\t", index=False, header=False)
        with self.assertRaisesRegex(ValueError, "unique"):
            load_single_cell_inputs(self.root / "cells")

    def test_missing_measured_expression_fails_before_writing(self):
        (self.root / "cells/cell_gene_counts.npz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "measured gene counts"):
            run_downstream(self.config, stage="prepare")
        self.assertFalse((self.root / "outputs").exists())

    def test_roi_budget_mismatch_and_invalid_geometry(self):
        path = self.root / "bad.json"
        path.write_text(json.dumps({"model_rois": self.rois[:1], "checkpoint_provenance": {"train_num_rois": 3}}))
        with self.assertRaisesRegex(ValueError, "training checkpoint"):
            read_rois(path, 1)
        path.write_text(json.dumps({"model_rois": [{**self.rois[0], "right_px": 0}]}))
        with self.assertRaisesRegex(ValueError, "bounds"):
            read_rois(path, 1)

    def test_dry_run_and_commands_use_selected_budgets_and_separate_environments(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run_downstream(self.config, budgets=(2,), methods=("bleep",), dry_run=True)
        self.assertIn("rescope.downstream.bleep", stream.getvalue())
        self.assertNotIn("rescope.downstream.istar", stream.getvalue())
        self.assertFalse((self.root / "outputs").exists())
        for method in ("bleep", "istar"):
            cmd = command(self.cfg, 2, method)
            self.assertEqual(cmd[0], self.cfg[method + "_python"])
            self.assertEqual(cmd[cmd.index("--roi-names") + 1], "roi_2")
            self.assertEqual(cmd[cmd.index("--gene-list") + 1], self.cfg["gene_list"])

    def test_bad_gene_panel_and_configuration_do_not_start_training(self):
        (self.root / "genes.txt").write_text("G1\nMISSING\n")
        with self.assertRaisesRegex(ValueError, "absent"):
            prepare(self.cfg, 1)
        self.assertFalse((self.root / "outputs").exists())
        data = yaml.safe_load(self.config.read_text())
        data["bleep"] = {"val_fraction": 1.0}
        self.config.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(ValueError, "val_fraction"):
            load_downstream(self.config)

    def test_bleep_neighbor_aggregation_matches_manual_result(self):
        reference = sparse.csr_matrix([[1, 2], [3, 6], [8, 9]], dtype=np.float32)
        indices = np.tile([[2, 0]], (70, 1))  # cross the query-chunk boundary
        scores = np.tile([[1001., 1000.]], (70, 1)).astype(np.float32)
        expected = (np.array([8, 9]) * np.e + np.array([1, 2])) / (np.e + 1)
        for method, target in [("simple", [8, 9]), ("average", [4.5, 5.5]), ("weighted_average", expected)]:
            actual = aggregate_predictions(reference, indices, scores, np.array([0, 1]), method)
            np.testing.assert_allclose(actual, np.tile(target, (70, 1)), rtol=1e-6)

    def test_reference_reconstruction_preserves_selected_cells_and_gene_order(self):
        result = rebuild_reference_matrix({"single_cell_dir": str(self.root / "cells")},
                                           pd.DataFrame({"cell_id": ["007", "002"]}), ["G3", "G1"])
        np.testing.assert_array_equal(result.toarray(), self.counts[[7, 2]][:, [2, 0]])

    def test_istar_query_coordinates_follow_image_scale_and_xy_order(self):
        from PIL import Image
        root = self.root / "scaled"
        root.mkdir()
        (root / "pixel-size-raw.txt").write_text("1")
        (root / "pixel-size.txt").write_text("2")
        Image.new("RGB", (160, 80)).save(root / "he.jpg")
        points = pd.DataFrame({"centroid_x": [64], "centroid_y": [32]})
        np.testing.assert_array_equal(compute_query_locs(root, points, (5, 10)), [[1, 2]])

    def test_common_query_summary_aligns_ids_across_budgets(self):
        for k, ids in [(1, ["a", "b", "c"]), (2, ["c", "b", "d"])]:
            directory = self.root / f"outputs/roi_{k}/bleep/roi_{k}/bleep_eval"
            directory.mkdir(parents=True)
            pred = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
            truth = np.zeros_like(pred)
            np.savez_compressed(directory / "predictions_and_truth.npz", predicted=pred, truth=truth,
                                query_cell_ids=np.array(ids), gene_names=np.array(["G3", "G1"]))
            metrics, _ = compute_metrics(pred, truth, ["G3", "G1"])
            (directory / "metrics.json").write_text(json.dumps(metrics))
        summarize(self.cfg, (1, 2), ("bleep",))
        result = pd.read_csv(self.root / "outputs/common_query_summary.csv")
        self.assertEqual(result.n_query_cells.tolist(), [2, 2])
        np.testing.assert_allclose(result.mse, [6.5, 2.5])

    def test_nonfinite_metrics_are_not_reported_as_success(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            compute_metrics(np.array([[np.nan, 1]]), np.ones((1, 2)), ["G1", "G2"])

    def test_raw_pipeline_dispatches_optional_downstream_stage(self):
        from rescope.preprocessing.pipeline import run_pipeline
        raw_config = {"sample": "test", "wsi_path": "image.tiff", "mpp": 0.25,
                      "uni_checkpoint": "uni.bin", "cellvit_repo": "CellViT",
                      "cellvit_checkpoint": "cellvit.pth", "cellvit_python": "cellvit/bin/python",
                      "work_dir": "raw_outputs", "downstream_config": "downstream.yaml"}
        path = self.root / "pipeline.yaml"
        path.write_text(yaml.safe_dump(raw_config))
        with patch("rescope.preprocessing.pipeline.preflight"), patch("rescope.downstream.pipeline.run_downstream") as run:
            run_pipeline(path, stage="downstream")
        run.assert_called_once_with(str(self.config))
        status = json.loads((self.root / "raw_outputs/stage_downstream.json").read_text())
        self.assertEqual(status["status"], "completed")

    def test_raw_pipeline_keeps_downstream_optional(self):
        from rescope.preprocessing.pipeline import run_pipeline
        raw_config = {"sample": "test", "wsi_path": "image.tiff", "mpp": 0.25,
                      "uni_checkpoint": "uni.bin", "cellvit_repo": "CellViT",
                      "cellvit_checkpoint": "cellvit.pth", "cellvit_python": "cellvit/bin/python",
                      "work_dir": "raw_outputs"}
        path = self.root / "pipeline.yaml"
        path.write_text(yaml.safe_dump(raw_config))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_pipeline(path, dry_run=True)
        self.assertNotIn("downstream:", output.getvalue())
        with self.assertRaisesRegex(ValueError, "downstream_config"):
            run_pipeline(path, stage="downstream")
        self.assertFalse((self.root / "raw_outputs").exists())

    def test_python_environment_symlinks_are_preserved(self):
        import sys
        from rescope.config import resolve_path
        interpreter = self.root / "env-bleep/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        cfg = load_downstream(self.config)
        self.assertEqual(cfg["bleep_python"], str(interpreter))
        self.assertNotEqual(cfg["bleep_python"], str(interpreter.resolve()))
        from rescope.preprocessing.pipeline import load_pipeline
        raw = {"sample": "test", "wsi_path": "image.tiff", "mpp": 0.25,
               "uni_checkpoint": "uni.bin", "cellvit_repo": "CellViT",
               "cellvit_checkpoint": "cellvit.pth", "cellvit_python": "env-bleep/bin/python"}
        path = self.root / "pipeline.yaml"
        path.write_text(yaml.safe_dump(raw))
        self.assertEqual(load_pipeline(path)["cellvit_python"], str(interpreter))
        self.assertEqual(resolve_path(interpreter, self.root), interpreter.resolve())


if __name__ == "__main__":
    unittest.main()
