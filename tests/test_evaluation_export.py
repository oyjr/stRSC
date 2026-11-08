import json
from pathlib import Path

import pytest

from strsc.evaluation_export import export_results


def hest_run(root: Path, unit="S1"):
    root.mkdir()
    record = {"protocol": "HEST_G0047B", "status": "COMPLETE", "task": "LUNG", "specimen": unit,
              "models": [{"model": "one_prediction", "ordinary_PCC_graph_support": 0.6,
                          "coarse": 0.8, "fine": -0.1}],
              "band_scores": [{"model": "one_prediction", "band": "B1", "rsc": None,
                               "rsc_status": "ZERO_MODEL_VARIANCE", "V_RNA": 1, "V_model": 0, "C_model": 0}]}
    (root / "analysis.json").write_text(json.dumps(record))
    return root


def test_single_model_and_signed_undefined_scores(tmp_path):
    run = hest_run(tmp_path / "run")
    bundle = export_results([run], tmp_path / "bundle.json")
    assert {row["model"] for row in bundle["rows"]} == {"one_prediction"}
    assert next(row for row in bundle["rows"] if row["metric"] == "rsc_fine")["value"] == -0.1
    assert bundle["rows"][-1]["value"] is None
    assert bundle["rows"][-1]["status"] == "ZERO_MODEL_VARIANCE"
    assert json.loads((tmp_path / "bundle.json").read_text())["schema"] == "strsc-evaluation-bundle-v1"


def test_two_runs_and_duplicate_identity_rejected(tmp_path):
    a, b = hest_run(tmp_path / "a", "S1"), hest_run(tmp_path / "b", "S2")
    merged = export_results([a, b], tmp_path / "merged.json")
    assert {row["unit_id"] for row in merged["rows"]} == {"S1", "S2"}
    with pytest.raises(ValueError, match="Duplicate result identity"):
        export_results([a, a], tmp_path / "duplicate.json")
    assert not (tmp_path / "duplicate.json").exists()


def test_fourier_requires_explicit_unit_and_preserves_partial_status(tmp_path):
    table = tmp_path / "s.tsv"
    table.write_text("model\tscale_um\twavelength_band\tv_rna\trsc\tnyquist_status\tstatus\n"
                     "one\t128\t0_128\t0\tnan\tPARTIAL_HIGH_FREQUENCY_WINDOW\tNO_FOURIER_MODES\n")
    with pytest.raises(ValueError, match="provide --unit"):
        export_results([table], tmp_path / "no-unit.json")
    bundle = export_results([table], tmp_path / "with-unit.json", units=["S1"])
    assert bundle["rows"][0]["value"] is None
    assert bundle["rows"][0]["details"]["nyquist_status"] == "PARTIAL_HIGH_FREQUENCY_WINDOW"
    assert bundle["rows"][0]["status"] == "NO_FOURIER_MODES"
