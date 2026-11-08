"""Regression: the paired exporter must preserve the scorer's Pearson rules."""
from pathlib import Path
import csv
import json
import tempfile
import unittest

import h5py
import numpy as np

from strsc.export import export_analysis
from strsc.hest import InputError
from strsc.hest_core import _corr_cols


class ConstantPredictionExport(unittest.TestCase):
    def test_constant_prediction_and_undefined_truth_preserve_saved_scores(self):
        # 20 sites and 50 genes satisfy the real input parser. These generated
        # inputs only exercise export; fixed spatial rows are not rescored.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            genes = np.asarray([f"G{i}" for i in range(50)])
            barcodes = np.asarray([f"spot{i}" for i in range(20)])
            counts = np.tile(np.arange(20)[:, None] + 1, (1, 50))
            counts[:, 1] = 3
            xy = np.column_stack((np.arange(20), np.arange(20) % 4))
            np.savez(root / "counts.npz", counts=counts, xy=xy,
                     genes=genes, barcodes=barcodes)
            truth = np.log1p(counts)
            prediction = truth.copy()
            prediction[:, 0] = 1.0  # Variable truth; constant prediction => 0.
            prediction[:, 1] = np.arange(20)  # Constant truth stays undefined.
            prediction[:, 2] = -truth[:, 2]  # Signed Pearson must stay -1.
            saved = _corr_cols(prediction, truth)
            self.assertEqual(saved[0], 0.0)
            self.assertTrue(np.isnan(saved[1]))
            self.assertAlmostEqual(saved[2], -1.0)

            arms = ("baseline", "intervention")
            for arm in arms:
                with h5py.File(root / f"{arm}.h5", "w") as handle:
                    handle["prediction"] = prediction
                    handle["barcodes"] = barcodes.astype("S")
                    handle["genes"] = genes.astype("S")
            manifest = {
                "task": "LUNG", "specimen": "TENX118", "model": "synthetic",
                "prediction_transform": "log1p_raw_integer_counts_no_library_normalization",
                "cohort": {"TENX118": "counts.npz"},
                "predictions": {arm: f"{arm}.h5" for arm in arms},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            record = {
                "status": "COMPLETE", "protocol": "HEST_G0047B",
                "task": "LUNG", "specimen": "TENX118",
                "patch_common_spots": 20, "graph_retained_spots": 20,
                "models": [{"model": arm, "fine": 0.2, "coarse": 0.4,
                            "ordinary_PCC_graph_support": float(np.nanmean(saved))}
                           for arm in arms],
                "band_scores": [{"model": arm, "band": f"B{i}", "rsc": 0.2,
                                 "rsc_status": "OK", "V_RNA": 1.0,
                                 "V_model": 1.0, "C_model": 0.2}
                                for arm in arms for i in range(1, 6)],
            }
            (run / "analysis.json").write_text(json.dumps(record), encoding="utf-8")
            with (run / "gene_scores.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "specimen", "gene",
                    "pearson_graph_support", "spatial_eligible", "rna_weight"], delimiter="\t")
                writer.writeheader()
                for arm in arms:
                    for gene, score in zip(genes, saved):
                        writer.writerow({"model": arm, "specimen": "TENX118", "gene": gene,
                            "pearson_graph_support": score, "spatial_eligible": True,
                            "rna_weight": 1.0})

            output = root / "export.json"
            receipt = export_analysis(root / "manifest.json", run, output)
            self.assertFalse(receipt["spatial_scoring_rerun"])
            bundle = json.loads(output.read_text(encoding="utf-8"))
            for arm in arms:
                scores = bundle["specimens"][0][arm]["gene_scores"]
                self.assertEqual(scores[0]["pearson"], 0.0)
                self.assertIsNone(scores[1]["pearson"])
                self.assertAlmostEqual(scores[2]["pearson"], -1.0)
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))

            # Reusing the scorer must not weaken the changed-prediction guard.
            with h5py.File(root / "baseline.h5", "r+") as handle:
                changed = handle["prediction"][:]
                changed[:, 2] = truth[:, 2]
                handle["prediction"][:] = changed
            with self.assertRaisesRegex(InputError, "Prediction and saved Pearson disagree: baseline, G2"):
                export_analysis(root / "manifest.json", run, root / "must_not_export.json")


if __name__ == "__main__":
    unittest.main()
