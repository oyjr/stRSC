import json

import numpy as np
import pandas as pd
import pytest

import strsc
from strsc import fourier, rsc_cli


def spot_table(path, sections=("S1", "S2"), genes=6, seed=7):
    """Counts on a hexagonal lattice of spots 100 apart, with a broad and a fine pattern, and two predictions."""
    rng = np.random.default_rng(seed)
    rows, truth, broad = [], [], []
    for s in sections:
        for i in range(12):
            for j in range(12):
                x, y = 100.0 * j + 50.0 * (i % 2), 86.6 * i
                rows.append((s, f"{s}_{i:02d}_{j:02d}", x, y))
    xy = np.asarray([[r[2], r[3]] for r in rows])
    phase = rng.uniform(0, 2 * np.pi, genes)
    slow = np.sin(xy[:, :1] / 400.0 + phase)
    fast = rng.normal(0.0, 1.0, size=(len(rows), genes))          # spot-to-spot pattern
    rate = np.exp(3.0 + 0.6 * slow + 0.4 * fast)
    counts = rng.poisson(rate)
    names = [f"g{k}" for k in range(genes)]
    frame = pd.DataFrame(rows, columns=["sample_id", "unit_id", "x", "y"])
    table = pd.concat([frame, pd.DataFrame(counts, columns=names)], axis=1)
    table.to_csv(path / "counts.csv", index=False)
    ids = frame[["sample_id", "unit_id"]]
    pd.concat([ids, pd.DataFrame(np.log1p(rate), columns=names)], axis=1).to_csv(path / "true_rates.csv", index=False)
    pd.concat([ids, pd.DataFrame(np.log1p(np.exp(3.0 + 0.6 * slow)), columns=names)], axis=1).to_csv(path / "broad_only.csv", index=False)
    return path / "counts.csv"


def test_public_api():
    assert strsc.__version__ == "0.4.0"
    for name in ("rsc", "read_counts", "regions", "ideal_check", "Result", "Counts", "Prediction"):
        assert hasattr(strsc, name)


def test_spots_cli_matches_python(tmp_path):
    counts_path = spot_table(tmp_path)
    counts = strsc.read_counts(counts_path)
    assert counts.section_names() == ["S1", "S2"] and len(counts.genes) == 6
    result = strsc.rsc(counts, {"true_rates": tmp_path / "true_rates.csv", "broad_only": tmp_path / "broad_only.csv"}, layout="spots")
    assert set(result.scales.section) == {"S1", "S2", "All sections"}
    assert list(result.scales.scale.unique()) == ["Broadest", "Broad", "Mid", "Fine", "Finest"]
    per = result.summary.set_index(["model", "section"])
    # the prediction with the fine pattern scores higher at fine scales than the one without it
    assert per.loc[("true_rates", "All sections"), "fine_rsc"] > per.loc[("broad_only", "All sections"), "fine_rsc"] + 0.3
    out = tmp_path / "cli-output"
    code = rsc_cli.main([str(counts_path), "-p", f"true_rates={tmp_path / 'true_rates.csv'}",
                         "-p", f"broad_only={tmp_path / 'broad_only.csv'}", "--layout", "spots", "--out", str(out), "--quiet"])
    assert code == 0
    saved = pd.read_csv(out / "scales.tsv", sep="\t", float_precision="round_trip")
    np.testing.assert_array_equal(saved.rsc.to_numpy(), result.scales.rsc.to_numpy())
    bundle = json.loads((out / "strsc-evaluation.json").read_text())
    assert bundle["schema"] == "strsc-evaluation-bundle-v1"
    assert {p.name for p in out.iterdir()} == {"scales.tsv", "summary.tsv", "genes.tsv", "settings.json", "strsc-evaluation.json"}


def test_grid_equals_fourier_score():
    rng = np.random.default_rng(3)
    y, x = np.indices((16, 16))
    coords = np.column_stack([x.ravel(), y.ravel()]) * 32.0
    rates = np.exp(2.5 + 0.5 * np.sin(2 * np.pi * x.ravel() / 8)[:, None] + rng.normal(0, 0.3, size=(256, 5)))
    counts = rng.poisson(rates).astype(np.int64)
    genes = [f"g{k}" for k in range(5)]
    reference = fourier.score(counts, rates, coords, np.full(256, 32.0**2), spacing_um=32, methods=["m"])
    data = strsc.Counts(strsc.files.sparse.csr_matrix(counts), genes, np.asarray([f"u{i}" for i in range(256)]),
                        np.full(256, "G"), coords, "um")
    result = strsc.rsc(data, {"m": rates}, layout="grid", spacing_um=32)
    ours = result.scales.set_index("band").rsc
    theirs = reference.set_index("wavelength_band").rsc
    for band in ours.index:
        assert ours[band] == pytest.approx(theirs[band], rel=1e-9, abs=1e-12)


def test_regions_and_ideal_check(tmp_path):
    counts_path = spot_table(tmp_path, sections=("S1",))
    counts = strsc.read_counts(counts_path)
    xy = counts.xy
    region = np.where(xy[:, 0] < np.median(xy[:, 0]), "left", "right").astype(object)
    region[((xy - xy.mean(axis=0)) ** 2).sum(axis=1) < 250**2] = "core"
    pd.DataFrame({"unit_id": counts.ids, "region": region}).to_csv(tmp_path / "regions.csv", index=False)
    counts = counts.with_labels(tmp_path / "regions.csv")
    result = strsc.regions(counts, {"true_rates": tmp_path / "true_rates.csv"}, focus="core", splits=4, resamples=100)
    assert set(result.scales.term) == {"whole", "between", "within", "focus", "focus_between", "focus_within"}
    assert {"rsc_ci_low", "rsc_ci_high", "testable"} <= set(result.scales.columns)
    table = strsc.ideal_check(counts, "spots", draws=1, splits=2)
    assert list(table.columns) == ["section", "scale", "scale_order", "rsc_mean", "rsc_min", "rsc_max", "draws"]
    assert len(table) == 5


def test_counts_must_be_whole_numbers(tmp_path):
    pd.DataFrame({"unit_id": ["a", "b"], "x": [0, 1], "y": [0, 0], "g": [1.5, 2]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(strsc.InputError, match="raw counts"):
        strsc.read_counts(tmp_path / "bad.csv")


def test_space_ranger_folder(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(1)
    barcodes = [f"AAAC{i:04d}-1" for i in range(30)]
    genes = ["GENE1", "GENE2", "GENE3"]
    dense = rng.poisson(3, size=(len(genes), len(barcodes)))
    csc = strsc.files.sparse.csc_matrix(dense)
    outs = tmp_path / "sample_A" / "outs"
    (outs / "spatial").mkdir(parents=True)
    with h5py.File(outs / "filtered_feature_bc_matrix.h5", "w") as handle:
        group = handle.create_group("matrix")
        # 10x stores the matrix as features x barcodes in CSC form
        group["data"], group["indices"], group["indptr"] = csc.data, csc.indices, csc.indptr
        group["shape"] = np.asarray(dense.shape)
        group["barcodes"] = np.asarray(barcodes, dtype="S")
        features = group.create_group("features")
        features["name"] = np.asarray(genes, dtype="S")
        features["feature_type"] = np.asarray(["Gene Expression"] * 3, dtype="S")
    positions = pd.DataFrame({"barcode": barcodes, "in_tissue": 1, "array_row": np.arange(30) // 6,
                              "array_col": 2 * (np.arange(30) % 6), "pxl_row_in_fullres": 100.0 * (np.arange(30) // 6),
                              "pxl_col_in_fullres": 100.0 * (np.arange(30) % 6)})
    positions.to_parquet(outs / "spatial" / "tissue_positions.parquet")
    (outs / "spatial" / "scalefactors_json.json").write_text(json.dumps({"spot_diameter_fullres": 110.0}))
    counts = strsc.read_counts(tmp_path / "sample_A")
    assert counts.section_names() == ["sample_A"] and counts.genes == genes and counts.unit == "um"
    np.testing.assert_allclose(counts.xy[1], [50.0, 0.0])          # 100 pixels x 55 um / 110 pixels
    np.testing.assert_array_equal(counts.counts.toarray(), dense.T)
