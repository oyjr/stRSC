import numpy as np
import pandas as pd
import pytest

from strsc import fourier


def grid_example(genes=40):
    rng = np.random.default_rng(42)
    y, x = np.indices((16, 16))
    coords = np.column_stack([x.ravel(), y.ravel()]) * 32.0
    signal = np.sin(2 * np.pi * x.ravel() / 8)
    rates = np.exp(3.5 + 0.5 * signal[:, None] + rng.normal(0, 0.2, size=(256, genes)))
    counts = rng.poisson(rates).astype(np.int64)
    prediction = np.stack([rates, -rates], axis=2)
    area = np.full(256, 32.0**2)
    return counts, prediction, coords, area


def test_physical_band_and_nonconstant_partition():
    counts, _, coords, area = grid_example(1)
    prediction = np.sin(2 * np.pi * coords[:, 0] / 256)[:, None]
    scores = fourier.score(
        counts, prediction, coords, area, spacing_um=32, seeds=[2026073101]
    )
    bands = scores.loc[scores.wavelength_band.ne("ALL_NON_DC")]
    assert bands.fourier_modes.sum() == 255
    target = bands.loc[bands.wavelength_band.eq("256_512")].iloc[0]
    assert target.v_model == pytest.approx(1, abs=1e-6)
    assert target.nyquist_status == "FULL_PHYSICAL_WINDOW"
    assert bands.v_model.sum() == pytest.approx(1, abs=1e-6)


def test_signed_scores_and_sufficient_statistics():
    scores = fourier.score(
        *grid_example(), spacing_um=32, methods=["positive", "negative"]
    )
    a = scores.loc[scores.model.eq("positive")].reset_index(drop=True)
    b = scores.loc[scores.model.eq("negative")].reset_index(drop=True)
    np.testing.assert_allclose(a.v_rna, b.v_rna)
    np.testing.assert_allclose(a.v_model, b.v_model)
    np.testing.assert_allclose(a.c_model, -b.c_model)
    np.testing.assert_allclose(a.rsc, -b.rsc, equal_nan=True)
    total = a.loc[a.wavelength_band.eq("ALL_NON_DC")].iloc[0]
    bands = a.loc[a.wavelength_band.ne("ALL_NON_DC")]
    for key in ("v_rna", "v_model", "c_model"):
        assert total[key] == pytest.approx(bands[key].sum())
    assert total.rsc == pytest.approx(
        total.c_model / np.sqrt(total.v_rna * total.v_model)
    )


def test_invalid_support_and_cli(tmp_path):
    counts, prediction, coords, area = grid_example(3)
    zero = fourier.score(
        counts,
        np.zeros_like(prediction[:, :, 0]),
        coords,
        area,
        spacing_um=32,
        seeds=[2026073101],
    )
    assert zero.rsc.isna().all()
    assert zero.loc[zero.fourier_modes.eq(0), "status"].eq("NO_FOURIER_MODES").all()
    with pytest.raises(ValueError, match="regular grid"):
        fourier.score(
            counts,
            prediction,
            coords + np.arange(len(coords))[:, None] * 0.001,
            area,
            spacing_um=32,
        )
    path = tmp_path / "grid.npz"
    np.savez(path, counts=counts, prediction=prediction, coords=coords, area=area)
    output = tmp_path / "scores.tsv"
    assert (
        fourier.main(
            [
                "--input",
                str(path),
                "--spacing-um",
                "32",
                "--seeds",
                "2026073101",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert len(pd.read_csv(output, sep="\t")) == 16
