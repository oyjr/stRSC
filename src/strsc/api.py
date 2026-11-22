"""Reproducible-signal correlation (RSC) in one call.

    import strsc
    counts = strsc.read_counts("sample.h5ad")
    result = strsc.rsc(counts, {"my_model": "predictions.csv"}, layout="spots")
    print(result.scales)

Each layout is scored as in the paper (Methods, "Reproducible-signal correlation" and
"Scales and count transforms"), with the toolkit's own functions:

``"grid"``
    Bins on a regular grid, such as Visium HD. A two-dimensional Fourier basis divides the
    variation into seven physical scales, from below 128 µm to 4,096 µm and above. Half-counts
    are transformed as sqrt(N_half / (0.5 a)) with a the bin area; five count splits. These are the
    steps of :func:`strsc.fourier.score`, which gives the same RSC.
``"spots"``
    Spots on a neighbor graph, such as Visium or HEST. The eigenvectors of the graph Laplacian
    give five relative scales, Broadest to Finest. Half-counts are transformed as log(1 + 2 N_half);
    twenty count splits. With two or more sections, each gene is weighted by the agreement of its
    two halves in the other sections, as for the HEST samples of the paper. A single section has
    no other sections, so each gene's two halves are divided by a common factor, as for the
    colorectal bins and the breast cancer cells of the paper; the values are then not directly
    comparable with those of the HEST weighting.
``"cells"``
    Single cells, such as Xenium. Lazy diffusion operators on the graph of the eight nearest
    neighbors within 40 µm give five relative scales. Half-counts are transformed as
    log(1 + 2 N_half); five count splits. ``within`` scores the cells of one class on their own
    graph, as for the malignant cells of the paper.

:func:`regions` scores agreement between and within annotated regions, as for the TLS sections,
and :func:`ideal_check` repeats the paper's test with the expected expression as the prediction.

The workflows of the paper (``strsc hest``, ``strsc fourier``, ``strsc.cell_score``) are unchanged.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

if sys.platform == "emscripten":  # pragma: no cover - only in the browser
    # WebAssembly cannot start threads, and fourier.fft_fields asks scipy for two FFT workers.
    # The number of workers changes how the work is shared, not the result.
    import scipy.fft

    if not getattr(scipy.fft.fft2, "_single_worker", False):
        _fft2 = scipy.fft.fft2

        def _fft2_single(*args, **kwargs):
            return _fft2(*args, **{**kwargs, "workers": 1})

        _fft2_single._single_worker = True
        scipy.fft.fft2 = _fft2_single
    # numpy in the browser multiplies matrices without BLAS; SciPy's BLAS does the spot projections
    # about 15 times faster, in the same single precision.
    from scipy.linalg import blas as _browser_blas

from . import __version__
from . import cell_core, evaluation_export, fourier
from . import hest_core as cal
from .files import Counts, InputError, Prediction, read_counts, read_predictions

LAYOUTS = ("spots", "grid", "cells")
RELATIVE = ("Broadest", "Broad", "Mid", "Fine", "Finest")
SPOT_SEEDS = tuple(cal.THINNING_SEEDS)           # 20 count splits, as for the HEST spots
GRID_SEEDS = tuple(fourier.SPLIT_SEEDS)          # 5, as for the colorectal bins
CELL_SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)  # 5, as for the breast cancer cells
REGION_SPLITS = 20                               # as for the TLS sections
CELL_GRAPH = dict(neighbors=8, max_edge_um=40.0, minimum_component_cells=20)
CELL_GENE_BLOCK = 64
GRID_LABELS = {"LT128": "<128 µm", "128_256": "128–256 µm", "256_512": "256–512 µm", "512_1024": "512–1,024 µm",
               "1024_2048": "1,024–2,048 µm", "2048_4096": "2,048–4,096 µm", "GE4096": "≥4,096 µm"}
TOL = 1e-12

Progress = Callable[[dict], None] | None


# ---------------------------------------------------------------- results

@dataclass
class Result:
    """Scores of one call. ``scales`` has one row per model, section and scale (or region term);
    ``summary`` one row per model and section; ``genes`` the same scores gene by gene."""

    layout: str
    scales: pd.DataFrame
    summary: pd.DataFrame
    genes: pd.DataFrame
    settings: dict
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        shown = self.summary.drop(columns=[c for c in self.summary.columns if c.startswith("n_")], errors="ignore")
        return f"RSC, {self.layout}: {len(self.scales)} rows\n" + shown.to_string(index=False, float_format=lambda v: f"{v:.3f}")

    def to_bundle(self, title: str = "RSC computed with strsc", synthetic: bool = False) -> dict:
        """The scores as an evaluation bundle, which the Computed results tab of strsc.org opens."""
        return _bundle(self, title, synthetic)

    def save(self, folder: str | Path, title: str = "RSC computed with strsc") -> list[Path]:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        for name, frame in (("scales.tsv", self.scales), ("summary.tsv", self.summary), ("genes.tsv", self.genes)):
            frame.to_csv(folder / name, sep="\t", index=False)
            written.append(folder / name)
        (folder / "settings.json").write_text(json.dumps(evaluation_export.finite_json(self.settings), indent=2, ensure_ascii=False))
        (folder / "strsc-evaluation.json").write_text(json.dumps(self.to_bundle(title)))
        return written + [folder / "settings.json", folder / "strsc-evaluation.json"]


def _ratio(c: float, vrna: float, vmodel: float) -> tuple[float, str]:
    if not (vmodel > TOL):
        return math.nan, "NO_PREDICTED_VARIANCE"
    if not (vrna > 0):
        return math.nan, "RNA_UNRESOLVED"
    return float(c / math.sqrt(vrna * vmodel)), "ESTIMABLE"


def _record(model, section, scale, order, c, vmodel, vrna, va, vb, seeds_c=None, seeds_vrna=None, **extra) -> dict:
    """One row: RSC, the Pearson r of the prediction with the mean of the two halves, and the split-half reliability."""
    rsc, status = _ratio(c, vrna, vmodel)
    mean_var = 0.25 * (va + vb + 2.0 * vrna)
    pearson = float(c / math.sqrt(vmodel * mean_var)) if vmodel > TOL and mean_var > TOL else math.nan
    reliability = float(vrna / math.sqrt(va * vb)) if va > TOL and vb > TOL else math.nan
    lo = hi = math.nan
    if seeds_c is not None and seeds_vrna is not None and vmodel > TOL:
        per = [c_ / math.sqrt(v_ * vmodel) for c_, v_ in zip(seeds_c, seeds_vrna) if v_ > 0]
        if per:
            lo, hi = float(min(per)), float(max(per))
    return dict(model=model, section=section, scale=scale, scale_order=order, rsc=rsc, pearson_r=pearson,
                split_half_reliability=reliability, C=float(c), V_model=float(vmodel), V_RNA=float(vrna),
                V_A=float(va), V_B=float(vb), rsc_split_min=lo, rsc_split_max=hi, status=status, **extra)


def _gene_rows(model, section, genes, labels, c, vmodel, vrna, va, vb, eligible=None) -> list[dict]:
    """Per-gene scores: c, vmodel, ... are (scales, genes) arrays."""
    rows = []
    for g, gene in enumerate(genes):
        if eligible is not None and not eligible[g]:
            continue
        for k, label in enumerate(labels):
            rsc, _ = _ratio(c[k, g], vrna[k, g], vmodel[k, g])
            rel = vrna[k, g] / math.sqrt(va[k, g] * vb[k, g]) if va[k, g] > TOL and vb[k, g] > TOL else math.nan
            rows.append(dict(model=model, section=section, gene=gene, scale=label, scale_order=k + 1, rsc=rsc,
                             split_half_reliability=float(rel), C=float(c[k, g]), V_model=float(vmodel[k, g]), V_RNA=float(vrna[k, g])))
    return rows


# ---------------------------------------------------------------- inputs

def _models(predictions, counts: Counts) -> list[Prediction]:
    if isinstance(predictions, Prediction):
        return [predictions]
    if isinstance(predictions, (str, Path)):
        return [read_predictions(predictions, counts)]
    if isinstance(predictions, Mapping):
        out = []
        for name, value in predictions.items():
            if isinstance(value, Prediction):
                out.append(Prediction(str(name), value.values, value.genes, value.source))
            elif isinstance(value, (str, Path)):
                out.append(read_predictions(value, counts, str(name)))
            else:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (len(counts), len(counts.genes)):
                    raise InputError(f"{name}: an array prediction needs the shape of the counts, {len(counts):,} × {len(counts.genes):,}.")
                out.append(Prediction(str(name), array, list(counts.genes), "array"))
        return out
    if isinstance(predictions, Sequence):
        return [p if isinstance(p, Prediction) else read_predictions(p, counts) for p in predictions]
    raise InputError("Give the predictions as a file, a Prediction, or a dict of model name to file or array.")


def _common(counts: Counts, models: list[Prediction], genes: Iterable[str] | None):
    """Genes predicted by every model, and the locations that have a value for each of them."""
    names = [m.name for m in models]
    if not models:
        raise InputError("Give at least one prediction.")
    if len(set(names)) != len(names):
        raise InputError("Two predictions have the same name. Name each model once.")
    wanted = list(counts.genes) if genes is None else [g for g in genes]
    if genes is not None:
        missing = [g for g in wanted if g not in set(counts.genes)]
        if missing:
            raise InputError(f"{len(missing)} of the chosen genes are not in the counts, for example {missing[0]}.")
    shared = [g for g in wanted if all(g in set(m.genes) for m in models)]
    if len(shared) < 1:
        raise InputError("No gene is in the counts and in every prediction. Use the same gene names.")
    columns = np.asarray([counts.genes.index(g) for g in shared])
    complete = np.ones(len(counts), dtype=bool)
    for m in models:
        complete &= np.isfinite(m.values[:, columns]).all(axis=1)
    notes = []
    dropped_genes = len(wanted) - len(shared)
    if dropped_genes:
        notes.append(f"{dropped_genes:,} gene{'s' if dropped_genes != 1 else ''} of the counts {'are' if dropped_genes != 1 else 'is'} not predicted by every model and {'were' if dropped_genes != 1 else 'was'} left out.")
    outside = int((~complete).sum())
    if outside:
        notes.append(f"{outside:,} location{'s' if outside != 1 else ''} without a prediction for every gene from every model {'were' if outside != 1 else 'was'} left out.")
    if not complete.any():
        raise InputError("No spot, bin or cell has a prediction for every gene from every model.")
    return shared, columns, complete, notes


def _settings(layout, counts, genes, models, seeds, **extra) -> dict:
    return dict(toolkit=f"strsc {__version__}", layout=layout, sections=counts.section_names(), genes=len(genes),
                models=[m.name for m in models], count_splits=len(seeds), split_seeds=list(map(int, seeds)), **extra)


# ---------------------------------------------------------------- spots

def _molecule_moments(counts: np.ndarray, specimen: str, task: str, seeds) -> dict:
    """As in strsc hest: per gene, the mean variance of a half-count, the mean covariance of the halves
    and their inner product, averaged over count splits."""
    d2 = np.zeros(counts.shape[1])
    covariance = np.zeros_like(d2)
    ell2 = np.zeros_like(d2)
    for seed in seeds:
        a, b = cal._halves(counts, task, specimen, seed)
        d2 += 0.5 * (np.mean(a * a, axis=0) + np.mean(b * b, axis=0))
        covariance += np.mean(a * b, axis=0)
        ell2 += np.sum(a * b, axis=0)
    scale = 1.0 / len(seeds)
    return {"d2": d2 * scale, "covariance": covariance * scale, "ell2": ell2 * scale}


def spot_bands(xy: np.ndarray, progress=None):
    """The five equal-rank groups of nonconstant Laplacian eigenmodes of each connected component,
    as in ``hest_core.build_basis`` (combinatorial Laplacian, constant mode dropped, stable order),
    with numpy's dense eigensolver."""
    adjacency, keep, labels, d0, radius = cal._graph(xy)
    kept_n = int(keep.sum())
    components = sorted(np.unique(labels).tolist())
    chunks_by_component, totals = {}, np.zeros(cal.BANDS, dtype=np.int64)
    for component in components:
        size = int(np.sum(labels == component))
        chunks = np.array_split(np.arange(max(size - 1, 0)), cal.BANDS)
        chunks_by_component[component] = chunks
        totals += np.asarray([len(chunk) for chunk in chunks])
    bands = [np.zeros((kept_n, int(totals[k])), dtype=np.float64) for k in range(cal.BANDS)]
    offsets = np.zeros(cal.BANDS, dtype=np.int64)
    for component in components:
        local = np.flatnonzero(labels == component)
        block = adjacency[local][:, local]
        laplacian = -block.toarray().astype(np.float64)
        degree = np.asarray(block.sum(axis=1)).reshape(-1).astype(np.float64)
        laplacian[np.diag_indices_from(laplacian)] = degree
        values, vectors = np.linalg.eigh(laplacian)
        order = np.argsort(values, kind="stable")
        vectors = vectors[:, order][:, 1:]
        for k, indices in enumerate(chunks_by_component[component]):
            start, end = int(offsets[k]), int(offsets[k]) + len(indices)
            if len(indices):
                bands[k][local, start:end] = vectors[:, indices]
            offsets[k] = end
        if progress:
            progress()
    return keep, bands, dict(spots_retained=kept_n, components=len(components), d0=d0, edge_radius=radius, band_modes=totals.tolist())


def _coefficients(bands, x):
    """Band projections in single precision, as ``hest_core._coefficients``."""
    x32 = np.asarray(x, dtype=np.float32)
    if sys.platform == "emscripten":  # pragma: no cover - only in the browser
        return [np.asarray(_browser_blas.sgemm(1.0, np.asarray(u, dtype=np.float32), x32, trans_a=True), dtype=np.float64, order="C")
                for u in bands]
    return [(np.asarray(u, dtype=np.float32).T @ x32).astype(np.float64) for u in bands]


def _task_label(names, default: str = "Sections") -> str:
    """The HEST task when the sections are exactly the samples of one, so that the count splits are
    those of ``strsc hest``; otherwise a plain label. The label seeds every count split."""
    return next((task for task, samples in cal.TASKS.items() if set(samples) == set(names)), default)


def score_spot_sections(sections: list[dict], genes: list[str], seeds=SPOT_SEEDS, step=None, max_spots: int | None = None,
                        task: str | None = None) -> list[dict]:
    """Relative scales on spots. ``sections``: dicts with name, counts (int, spots x genes), xy and
    predictions {model: spots x genes}. With two or more sections the gene weights come from the
    other sections, as ``strsc hest``; one section uses the common factor of each gene's halves."""
    task = task or _task_label([s["name"] for s in sections])
    weighted = len(sections) >= 2
    retained, moments = {}, {}
    for s in sections:
        _, keep, _, _, _ = cal._graph(s["xy"])
        if not keep.any():
            raise InputError(f"Section {s['name']}: no connected group of at least 20 spots.")
        if max_spots and int(keep.sum()) > max_spots:
            raise InputError(f"Section {s['name']} has {int(keep.sum()):,} connected spots. Here the limit is {max_spots:,}. Run the toolkit on your own machine for larger sections, without cropping them.")
        retained[s["name"]] = keep
        if weighted:
            moments[s["name"]] = _molecule_moments(s["counts"][keep], s["name"], task, seeds)
    records = []
    for s in sections:
        name, keep = s["name"], retained[s["name"]]
        counts = s["counts"][keep]
        if weighted:
            others = [o["name"] for o in sections if o["name"] != name]
            d2 = np.mean(np.stack([moments[o]["d2"] for o in others]), axis=0)
            cov = np.mean(np.stack([moments[o]["covariance"] for o in others]), axis=0)
            h = np.divide(cov, d2, out=np.zeros_like(d2), where=d2 > cal.NUMERIC_TOL)
            w, ell2 = np.maximum(h, 0.0), moments[name]["ell2"]
            eligible = (d2 > cal.NUMERIC_TOL) & (w > 0) & (ell2 > 0)
            d = np.sqrt(d2)
            rna_scale = np.divide(np.sqrt(w), d, out=np.zeros_like(w), where=d > cal.NUMERIC_TOL)
            rna_scale[~eligible] = 0.0
            p_scale = np.divide(np.sqrt(w) * np.sqrt(np.maximum(ell2, 0)), d, out=np.zeros_like(w), where=d > cal.NUMERIC_TOL)
            p_scale[~eligible] = 0.0
        else:
            others = []
            eligible = np.asarray(counts.sum(axis=0)).reshape(-1) > 0
        if step:
            step(name, "graph")
        keep2, bands, graph = spot_bands(s["xy"])
        assert np.array_equal(keep2, keep)
        prepared = {}
        for model, prediction_full in s["predictions"].items():
            prediction = prediction_full[keep]
            centered = prediction - prediction.mean(axis=0, keepdims=True)
            pnorm = np.sqrt(np.sum(centered * centered, axis=0))
            p = np.divide(centered, pnorm[None, :], out=np.zeros_like(centered), where=pnorm[None, :] > cal.NUMERIC_TOL)
            if weighted:
                p = p * p_scale[None, :]
            else:
                p[:, ~eligible] = 0.0
            prepared[model] = _coefficients(bands, p)
        n_genes = counts.shape[1]
        vrna, va, vb = np.zeros(5), np.zeros(5), np.zeros(5)
        vrna_g, va_g, vb_g = np.zeros((5, n_genes)), np.zeros((5, n_genes)), np.zeros((5, n_genes))
        mean_observed = [None] * 5
        seed_vrna = np.zeros((len(seeds), 5))
        seed_c = {model: np.zeros((len(seeds), 5)) for model in prepared}
        for i, seed in enumerate(seeds):
            a, b = cal._halves(counts, task, name, seed)
            if weighted:
                a *= rna_scale[None, :]
                b *= rna_scale[None, :]
            else:
                factor = np.sqrt(0.5 * (np.sum(a * a, axis=0) + np.sum(b * b, axis=0)))
                ok = (factor > cal.NUMERIC_TOL) & eligible
                a[:, ok] /= factor[ok]
                b[:, ok] /= factor[ok]
                a[:, ~ok] = 0.0
                b[:, ~ok] = 0.0
            ac, bc = _coefficients(bands, a), _coefficients(bands, b)
            for k in range(5):
                vrna[k] += np.sum(ac[k] * bc[k])
                current = 0.5 * (ac[k] + bc[k])
                mean_observed[k] = current if mean_observed[k] is None else mean_observed[k] + current
                va[k] += np.sum(ac[k] * ac[k])
                vb[k] += np.sum(bc[k] * bc[k])
                vrna_g[k] += np.sum(ac[k] * bc[k], axis=0)
                va_g[k] += np.sum(ac[k] * ac[k], axis=0)
                vb_g[k] += np.sum(bc[k] * bc[k], axis=0)
                seed_vrna[i, k] = np.sum(ac[k] * bc[k])
                for model, pc in prepared.items():
                    seed_c[model][i, k] = np.sum(pc[k] * current)
            if step:
                step(name, "split")
        n = len(seeds)
        vrna /= n
        va /= n
        vb /= n
        vrna_g /= n
        va_g /= n
        vb_g /= n
        mean_observed = [x / n for x in mean_observed]
        truth = np.log1p(s["counts"])
        models = []
        for model, pc in prepared.items():
            prediction_full = s["predictions"][model]
            vpred = np.asarray([np.sum(x * x) for x in pc])
            covariance = np.asarray([np.sum(pc[k] * mean_observed[k]) for k in range(5)])
            c_g = np.stack([np.sum(pc[k] * mean_observed[k], axis=0) for k in range(5)])
            vm_g = np.stack([np.sum(pc[k] * pc[k], axis=0) for k in range(5)])
            gene_pcc = cal._corr_cols(prediction_full[keep], truth[keep])
            full_pcc = cal._corr_cols(prediction_full, truth)
            models.append(dict(model=model, vpred=vpred, covariance=covariance, c_g=c_g, vm_g=vm_g, seed_c=seed_c[model],
                               gene_pcc=gene_pcc, full_pcc=full_pcc))
        records.append(dict(section=name, task=task, weighted=weighted, reference_sections=others, graph=graph,
                            spots=len(keep), retained=int(keep.sum()), eligible=eligible, vrna=vrna, va=va, vb=vb,
                            vrna_g=vrna_g, va_g=va_g, vb_g=vb_g, seed_vrna=seed_vrna, models=models))
    return records


def _spot_result(records, genes, settings, notes) -> Result:
    scale_rows, gene_rows, summaries = [], [], []
    for r in records:
        for m in r["models"]:
            rows = []
            for k, label in enumerate(RELATIVE):
                rows.append(_record(m["model"], r["section"], label, k + 1, m["covariance"][k], m["vpred"][k], r["vrna"][k],
                                    r["va"][k], r["vb"][k], m["seed_c"][:, k], r["seed_vrna"][:, k]))
            scale_rows.extend(rows)
            rsc = [row["rsc"] for row in rows]
            gene_rows.extend(_gene_rows(m["model"], r["section"], genes, RELATIVE, m["c_g"], m["vm_g"], r["vrna_g"], r["va_g"], r["vb_g"], r["eligible"]))
            summaries.append(dict(model=m["model"], section=r["section"],
                                  fine_rsc=_mean2(rsc[3], rsc[4]), broad_rsc=_mean2(rsc[0], rsc[1]),
                                  broadest_minus_finest=rsc[0] - rsc[4] if all(map(math.isfinite, (rsc[0], rsc[4]))) else math.nan,
                                  fine_split_half_reliability=_mean2(rows[3]["split_half_reliability"], rows[4]["split_half_reliability"]),
                                  whole_section_pearson_r=_nanmean(m["gene_pcc"]),
                                  n_locations=r["retained"], n_genes=int(np.sum(r["eligible"]))))
    scales, summary = pd.DataFrame(scale_rows), pd.DataFrame(summaries)
    if len(records) > 1:
        scales, summary = _all_sections_mean(scales, summary)
    return Result("spots", scales, summary, pd.DataFrame(gene_rows), settings, notes)


def _mean2(x, y) -> float:
    return 0.5 * (x + y) if math.isfinite(x) and math.isfinite(y) else math.nan


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def _all_sections_mean(scales: pd.DataFrame, summary: pd.DataFrame):
    """'All sections': the mean over sections of each section's scores, as the means over samples in the paper."""
    numeric = ["rsc", "pearson_r", "split_half_reliability"]
    pooled = (scales.groupby(["model", "scale", "scale_order"], as_index=False, sort=False)[numeric].mean())
    pooled.insert(1, "section", "All sections")
    pooled["status"] = "MEAN_OVER_SECTIONS"
    keep = [c for c in summary.columns if c not in ("model", "section") and not c.startswith("n_")]
    overall = summary.groupby("model", as_index=False, sort=False)[keep].mean()
    overall.insert(1, "section", "All sections")
    return pd.concat([scales, pooled], ignore_index=True), pd.concat([summary, overall], ignore_index=True)


# ---------------------------------------------------------------- grid

def score_grid_section(counts: sparse.csr_matrix, prediction: np.ndarray, coords: np.ndarray, area: np.ndarray, *,
                       spacing_um: float, models: Sequence[str], seeds=GRID_SEEDS, step=None) -> dict:
    """The steps of :func:`strsc.fourier.score`, unchanged, which also keep the variance of each half,
    the terms of each gene and of each count split. Its RSC equals that of ``fourier.score``."""
    counts = sparse.csr_matrix(counts)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    n, genes = counts.shape
    windows = pd.DataFrame({"center_x_um": coords[:, 0], "center_y_um": coords[:, 1]})
    scaled = (coords - coords.min(axis=0)) / spacing_um
    if not np.allclose(scaled, np.rint(scaled), rtol=0, atol=1e-6):
        raise InputError("The coordinates do not lie on a regular grid with this spacing.")
    iy, ix, ny, nx, masks = fourier.grid_contract(windows, spacing_um)
    names = list(masks)
    zeros = lambda: {band: np.zeros(genes) for band in names}
    vrna_g, va_g, vb_g = zeros(), zeros(), zeros()
    c_g = {model: zeros() for model in models}
    vm_g = {model: zeros() for model in models}
    terms = {(model, band): {"v_rna": 0.0, "c_model": 0.0, "v_model": 0.0} for model in models for band in names}
    halves = {band: {"v_a": 0.0, "v_b": 0.0} for band in names}
    seed_vrna = {band: np.zeros(len(seeds)) for band in names}
    seed_c = {(model, band): np.zeros(len(seeds)) for model in models for band in names}
    for offset in range(0, genes, fourier.GENE_BLOCK):
        stop = min(offset + fourier.GENE_BLOCK, genes)
        block = counts[:, offset:stop].tocsr()
        fp = {model: fourier.fft_fields(fourier.normalize_prediction(prediction[:, offset:stop, j]), iy, ix, ny, nx)
              for j, model in enumerate(models)}
        for model in models:
            for band, mask in masks.items():
                terms[model, band]["v_model"] += float(np.square(np.abs(fp[model][mask, :])).sum())
                vm_g[model][band][offset:stop] += np.square(np.abs(fp[model][mask, :])).sum(axis=0)
        block_vrna = {band: 0.0 for band in names}
        block_c = {(model, band): 0.0 for model in models for band in names}
        for i, seed in enumerate(seeds):
            aa, bb = fourier.split_transform(block, area, int(seed) + offset * 1009)
            fa, fb = (fourier.fft_fields(values, iy, ix, ny, nx) for values in (aa, bb))
            for band, mask in masks.items():
                cross = np.real(np.conj(fa[mask, :]) * fb[mask, :])
                value = float(cross.sum())
                block_vrna[band] += value
                seed_vrna[band][i] += value
                vrna_g[band][offset:stop] += cross.sum(axis=0) / len(seeds)
                ea, eb = np.square(np.abs(fa[mask, :])), np.square(np.abs(fb[mask, :]))
                halves[band]["v_a"] += float(ea.sum()) / len(seeds)
                halves[band]["v_b"] += float(eb.sum()) / len(seeds)
                va_g[band][offset:stop] += ea.sum(axis=0) / len(seeds)
                vb_g[band][offset:stop] += eb.sum(axis=0) / len(seeds)
                judge_sum = fa[mask, :] + fb[mask, :]
                for model in models:
                    product = 0.5 * np.real(np.conj(fp[model][mask, :]) * judge_sum)
                    value = float(product.sum())
                    block_c[model, band] += value
                    seed_c[model, band][i] += value
                    c_g[model][band][offset:stop] += product.sum(axis=0) / len(seeds)
        for band in names:
            for model in models:
                terms[model, band]["v_rna"] += block_vrna[band] / len(seeds)
                terms[model, band]["c_model"] += block_c[model, band] / len(seeds)
        if step:
            step(stop, genes)
    modes = {band: int(mask.sum()) for band, mask in masks.items()}
    return dict(terms=terms, halves=halves, modes=modes, seed_vrna=seed_vrna, seed_c=seed_c, vrna_g=vrna_g, va_g=va_g,
                vb_g=vb_g, c_g=c_g, vm_g=vm_g, grid=(ny, nx), sites=n)


def _grid_result(sections, genes, models, spacing_um, settings, notes) -> Result:
    order = [band for band, *_ in reversed(fourier.WAVELENGTH_BANDS)]  # broadest first
    lows = {band: low for band, low, _ in fourier.WAVELENGTH_BANDS}
    scale_rows, gene_rows, summaries = [], [], []
    for s in sections:
        out = s["out"]
        present = [band for band in order if out["modes"][band] > 0]
        for model in models:
            rows = []
            for k, band in enumerate(present):
                t, h = out["terms"][model, band], out["halves"][band]
                rows.append(_record(model, s["name"], GRID_LABELS[band], k + 1, t["c_model"], t["v_model"], t["v_rna"], h["v_a"], h["v_b"],
                                    out["seed_c"][model, band] / 1.0, out["seed_vrna"][band],
                                    band=band, fourier_modes=out["modes"][band],
                                    partial_window=bool(lows[band] < 2.0 * spacing_um)))
            # per-seed values are sums over gene blocks of that seed's terms
            scale_rows.extend(rows)
            stack = lambda d: np.stack([d[band] for band in present])
            gene_rows.extend(_gene_rows(model, s["name"], genes, [GRID_LABELS[b] for b in present], stack(out["c_g"][model]),
                                        stack(out["vm_g"][model]), stack(out["vrna_g"]), stack(out["va_g"]), stack(out["vb_g"]),
                                        s["eligible"]))
            rsc = [row["rsc"] for row in rows]
            finest = next((row for row in reversed(rows) if math.isfinite(row["rsc"])), None)
            summaries.append(dict(model=model, section=s["name"], broadest_rsc=rsc[0] if rsc else math.nan,
                                  finest_rsc=finest["rsc"] if finest else math.nan, finest_scale=finest["scale"] if finest else "",
                                  broadest_minus_finest=(rsc[0] - finest["rsc"]) if finest and rsc and math.isfinite(rsc[0]) else math.nan,
                                  whole_section_pearson_r=s["pearson"][model], n_locations=s["sites"], n_genes=int(np.sum(s["eligible"]))))
    scales, summary = pd.DataFrame(scale_rows), pd.DataFrame(summaries)
    if len(sections) > 1:
        scales, summary = _all_sections_mean(scales, summary)
    return Result("grid", scales, summary, pd.DataFrame(gene_rows), settings, notes)


# ---------------------------------------------------------------- cells

PAIRS = (("A", "A"), ("A", "B"), ("A", "P"), ("B", "B"), ("B", "P"), ("P", "P"))


def _band_grams_by_gene(diffusion: sparse.csr_matrix, matrices: dict[str, np.ndarray]) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """``cell_core.lazy_band_grams`` for each gene: the same products, summed over cells only."""
    names = list(matrices)
    original = {name: np.asarray(matrices[name], np.float32) for name in names}
    state = {name: value.copy() for name, value in original.items()}
    milestones: dict[int, dict[tuple[str, str], np.ndarray]] = {}
    for step in range(17):
        if step in cell_core.MILESTONES:
            gram = {}
            for i, left in enumerate(names):
                for right in names[i:]:
                    gram[(left, right)] = 0.5 * (np.sum(original[left] * state[right], axis=0, dtype=np.float64)
                                                 + np.sum(original[right] * state[left], axis=0, dtype=np.float64))
            milestones[step] = gram
        if step < 16:
            for name in names:
                state[name] = diffusion @ state[name]
    out = {}
    for band, high, low in (("B1", 16, None), ("B2", 8, 16), ("B3", 4, 8), ("B4", 2, 4), ("B5", 0, 2)):
        out[band] = {pair: value if low is None else value - milestones[low][pair] for pair, value in milestones[high].items()}
    return out


def score_cell_view(counts: sparse.csr_matrix, prediction: np.ndarray, coords: np.ndarray, membership: sparse.csr_matrix,
                    view_coords: np.ndarray, seeds=CELL_SEEDS, specimen: str = "section", step=None) -> dict:
    """One graph view of a section, as ``strsc.cell_score``: the graph is built on the view's cells,
    the counts are split on all cells, and the halves and the prediction are transformed and
    filtered on the view's graph."""
    diffusion, retained, labels, receipt = cell_core.build_lazy_graph(view_coords, CELL_GRAPH["neighbors"], CELL_GRAPH["max_edge_um"],
                                                                      CELL_GRAPH["minimum_component_cells"])
    exposure = np.ones(membership.shape[0])
    pred_view = np.asarray(membership @ prediction, np.float32)
    pred_view /= np.asarray(exposure, np.float32)[:, None]
    pred_view = pred_view[retained]
    genes = counts.shape[1]
    per_seed = {seed: {band: {pair: np.zeros(genes) for pair in PAIRS} for band in cell_core.BANDS} for seed in seeds}
    eligible_any = np.zeros(genes, dtype=bool)
    pcc = np.full(genes, np.nan)
    blocks = [slice(start, min(start + CELL_GENE_BLOCK, genes)) for start in range(0, genes, CELL_GENE_BLOCK)]
    view_counts = (membership @ counts).tocsr()[retained]
    for block in blocks:
        truth = np.log1p(np.asarray(view_counts[:, block].todense(), dtype=np.float64))
        pcc[block] = cal._corr_cols(pred_view[:, block].astype(np.float64), truth)
    del view_counts
    # One count split at a time, so that memory does not grow with the number of splits. Every split
    # and every gene block is computed on its own, so the order changes none of the numbers.
    for k, seed in enumerate(seeds):
        a, b = cell_core.split_counts(counts, specimen, seed)
        a_view, b_view = (membership @ a).tocsr()[retained], (membership @ b).tocsr()[retained]
        del a, b
        for i, block in enumerate(blocks):
            pp, pred_eligible = cell_core.transform_prediction(pred_view[:, block], labels)
            aa, bb, rna_eligible = cell_core.transform_rna_pair(a_view[:, block], b_view[:, block], exposure[retained], labels)
            eligible = pred_eligible & rna_eligible
            aa[:, ~eligible] = 0
            bb[:, ~eligible] = 0
            local_p = pp.copy()
            local_p[:, ~eligible] = 0
            eligible_any[block] |= eligible
            grams = _band_grams_by_gene(diffusion, {"A": aa, "B": bb, "P": local_p})
            for band in cell_core.BANDS:
                for pair in PAIRS:
                    per_seed[seed][band][pair][block] += grams[band][pair]
            if step:
                step(k * len(blocks) + i + 1, len(seeds) * len(blocks))
        del a_view, b_view
    return dict(per_seed=per_seed, eligible=eligible_any, receipt=receipt, pearson=_nanmean(pcc), cells=int(retained.sum()))


def _cell_terms(view, seeds) -> dict:
    """Totals over genes, averaged over seeds; per-gene terms averaged over seeds; and per-seed totals."""
    mean_g = {band: {pair: np.mean([view["per_seed"][s][band][pair] for s in seeds], axis=0) for pair in PAIRS} for band in cell_core.BANDS}
    total = {band: {pair: float(np.sum(mean_g[band][pair])) for pair in PAIRS} for band in cell_core.BANDS}
    seed_total = {band: {pair: np.asarray([float(np.sum(view["per_seed"][s][band][pair])) for s in seeds]) for pair in PAIRS} for band in cell_core.BANDS}
    return dict(mean_g=mean_g, total=total, seed_total=seed_total)


def _cell_result(views, genes, settings, notes) -> Result:
    scale_rows, gene_rows, summaries = [], [], []
    bands = list(cell_core.BANDS)
    for v in views:
        terms = _cell_terms(v["out"], v["seeds"])
        t, st, g = terms["total"], terms["seed_total"], terms["mean_g"]
        rows = []
        for k, band in enumerate(bands):
            c = 0.5 * (t[band][("A", "P")] + t[band][("B", "P")])
            seeds_c = 0.5 * (st[band][("A", "P")] + st[band][("B", "P")])
            rows.append(_record(v["model"], v["section"], RELATIVE[k], k + 1, c, t[band][("P", "P")], t[band][("A", "B")],
                                t[band][("A", "A")], t[band][("B", "B")], seeds_c, st[band][("A", "B")], population=v["population"]))
        scale_rows.extend(rows)
        combined = {}
        for label, members in (("fine", ("B4", "B5")), ("broad", ("B1", "B2"))):
            c = sum(0.5 * (t[b][("A", "P")] + t[b][("B", "P")]) for b in members)
            vm = sum(t[b][("P", "P")] for b in members)
            vr = sum(t[b][("A", "B")] for b in members)
            va_ = sum(t[b][("A", "A")] for b in members)
            vb_ = sum(t[b][("B", "B")] for b in members)
            combined[label] = _record(v["model"], v["section"], label, 0, c, vm, vr, va_, vb_)
        stack = lambda pair: np.stack([g[b][pair] for b in bands])
        gene_rows.extend({**row, "population": v["population"]} for row in _gene_rows(
            v["model"], v["section"], genes, RELATIVE, 0.5 * (stack(("A", "P")) + stack(("B", "P"))), stack(("P", "P")),
            stack(("A", "B")), stack(("A", "A")), stack(("B", "B")), v["out"]["eligible"]))
        rsc = [row["rsc"] for row in rows]
        summaries.append(dict(model=v["model"], section=v["section"], population=v["population"],
                              fine_rsc=combined["fine"]["rsc"], broad_rsc=combined["broad"]["rsc"],
                              broadest_minus_finest=rsc[0] - rsc[4] if all(map(math.isfinite, (rsc[0], rsc[4]))) else math.nan,
                              fine_split_half_reliability=combined["fine"]["split_half_reliability"],
                              whole_section_pearson_r=v["out"]["pearson"], n_locations=v["out"]["cells"],
                              n_genes=int(np.sum(v["out"]["eligible"]))))
    scales, summary = pd.DataFrame(scale_rows), pd.DataFrame(summaries)
    sections = summary["section"].unique()
    if len(sections) > 1:
        # As for the patients of the paper: C, V_model and V_RNA averaged over sections before the ratio.
        pooled_rows, pooled_summary = [], []
        for (model, population), frame in scales.groupby(["model", "population"], sort=False):
            for (scale, order), part in frame.groupby(["scale", "scale_order"], sort=False):
                m = part[["C", "V_model", "V_RNA", "V_A", "V_B"]].mean()
                pooled_rows.append(_record(model, "All sections", scale, order, m.C, m.V_model, m.V_RNA, m.V_A, m.V_B, population=population))
            fine = frame[frame.scale.isin(["Fine", "Finest"])].groupby("section")[["C", "V_model", "V_RNA", "V_A", "V_B"]].sum().mean()
            broad = frame[frame.scale.isin(["Broadest", "Broad"])].groupby("section")[["C", "V_model", "V_RNA", "V_A", "V_B"]].sum().mean()
            f = _record(model, "All sections", "fine", 0, fine.C, fine.V_model, fine.V_RNA, fine.V_A, fine.V_B)
            b = _record(model, "All sections", "broad", 0, broad.C, broad.V_model, broad.V_RNA, broad.V_A, broad.V_B)
            ordered = [r for r in pooled_rows if r["model"] == model and r["population"] == population]
            pooled_summary.append(dict(model=model, section="All sections", population=population, fine_rsc=f["rsc"], broad_rsc=b["rsc"],
                                       broadest_minus_finest=ordered[0]["rsc"] - ordered[4]["rsc"],
                                       fine_split_half_reliability=f["split_half_reliability"],
                                       whole_section_pearson_r=float(summary[(summary.model == model) & (summary.population == population)].whole_section_pearson_r.mean())))
        scales = pd.concat([scales, pd.DataFrame(pooled_rows)], ignore_index=True)
        summary = pd.concat([summary, pd.DataFrame(pooled_summary)], ignore_index=True)
    return Result("cells", scales, summary, pd.DataFrame(gene_rows), settings, notes)


# ---------------------------------------------------------------- the call

def rsc(counts: Counts | str | Path, predictions, layout: str = "spots", *, spacing_um: float | None = None,
        cell_class: str = "cell_class", within: str | Sequence[str] | None = None, all_cells: bool = True,
        genes: Iterable[str] | None = None, splits: int | None = None, max_spots: int | None = None,
        progress: Progress = None) -> Result:
    """Reproducible-signal correlation of one or more predictions, at each spatial scale.

    ``counts``: a :class:`Counts` from :func:`read_counts`, or a path. ``predictions``: a file, a
    :class:`Prediction`, or a dict of model name to file or array (locations × genes of the counts).
    ``layout``: ``"spots"``, ``"grid"`` or ``"cells"`` (see the module docstring). For ``"grid"``,
    ``spacing_um`` is the bin spacing (read from Visium HD files when not given). For ``"cells"``,
    ``within`` names one or more values of the ``cell_class`` column to score on their own graphs,
    and ``all_cells`` whether to score all cells as well. ``genes`` restricts the genes, ``splits``
    uses only the first count splits of the paper (fewer splits give noisier values).
    """
    if layout not in LAYOUTS:
        raise InputError(f"layout must be one of {', '.join(LAYOUTS)}.")
    if not isinstance(counts, Counts):
        counts = read_counts(counts)
    models = _models(predictions, counts)
    shared, columns, complete, notes = _common(counts, models, genes)
    say = progress or (lambda _: None)
    if layout == "spots":
        seeds = SPOT_SEEDS[: splits or len(SPOT_SEEDS)]
        sections = []
        for name in counts.section_names():
            rows = np.flatnonzero((counts.sections == name) & complete)
            if len(rows) < 20:
                raise InputError(f"Section {name} has fewer than 20 spots with every prediction.")
            values = np.asarray(counts.counts[rows][:, columns].todense(), dtype=np.int64)
            sections.append(dict(name=name, counts=values, xy=counts.xy[rows],
                                 predictions={m.name: m.values[np.ix_(rows, columns)] for m in models}))
        weighted = len(sections) >= 2
        if not weighted:
            notes.append("One section: each gene's two halves are divided by a common factor, as for the colorectal bins and breast cancer cells of the paper. "
                         "With two or more sections the genes are weighted by the agreement of their halves in the other sections, as for the HEST samples. The two weightings give different values.")
        total = len(sections) * (1 + len(seeds))
        done = [0]

        def step(name, stage):
            done[0] += 1
            say(dict(stage=stage, section=name, done=done[0], total=total))
        records = score_spot_sections(sections, shared, seeds, step, max_spots)
        settings = _settings("spots", counts, shared, models, seeds, transform="log(1 + 2 N_half), centered by gene",
                             scales="five equal-rank groups of the nonconstant eigenvectors of the neighbor-graph Laplacian (edges within 1.25 × the median nearest-neighbor distance, components of at least 20 spots)",
                             gene_weighting="agreement of the halves in the other sections (HEST)" if weighted else "common factor of each gene's halves",
                             fine_scale="mean RSC at the Fine and Finest scales", broad_scale="mean RSC at the Broadest and Broad scales",
                             task_label=records[0]["task"])
        return _spot_result(records, shared, settings, notes)
    if layout == "grid":
        spacing = float(spacing_um or counts.spacing_um or 0)
        if not math.isfinite(spacing) or spacing <= 0:
            raise InputError("Give the grid spacing in micrometers.")
        if counts.unit != "um" and spacing_um is None:
            raise InputError("The grid needs coordinates in micrometers: give um_per_unit when reading the counts, and the spacing.")
        seeds = GRID_SEEDS[: splits or len(GRID_SEEDS)]
        area_all = counts.column("area")
        if area_all is None:
            notes.append(f"No area column, so every bin counts as a full {spacing:g} × {spacing:g} µm square of tissue.")
        sections = []
        names = counts.section_names()
        parts = max(1, math.ceil(len(columns) / fourier.GENE_BLOCK))
        for index, name in enumerate(names):
            rows = np.flatnonzero((counts.sections == name) & complete)
            block = counts.counts[rows][:, columns].tocsr()
            eligible = np.asarray(block.sum(axis=0)).reshape(-1) > 0
            area = np.asarray(area_all[rows], dtype=float) if area_all is not None else np.full(len(rows), spacing * spacing)
            prediction = np.stack([m.values[np.ix_(rows, columns)] for m in models], axis=2)
            keep_genes = np.flatnonzero(eligible)
            if len(keep_genes) < len(columns):
                notes.append(f"Section {name}: {len(columns) - len(keep_genes):,} gene(s) without a single molecule were left out.")
            say(dict(stage="section", section=name, done=index * parts, total=len(names) * parts))

            def grid_step(stop, _genes, name=name, index=index):
                say(dict(stage="section", section=name, done=index * parts + math.ceil(stop / fourier.GENE_BLOCK), total=len(names) * parts))
            out = score_grid_section(block[:, keep_genes], prediction[:, keep_genes, :], counts.xy[rows], area, spacing_um=spacing,
                                     models=[m.name for m in models], seeds=seeds, step=grid_step)
            truth = np.sqrt(np.asarray(block[:, keep_genes].todense(), dtype=np.float64) / area[:, None])
            pearson = {m.name: _nanmean(cal._corr_cols(prediction[:, keep_genes, j], truth)) for j, m in enumerate(models)}
            sections.append(dict(name=name, out=out, eligible=np.ones(len(keep_genes), bool), sites=len(rows), pearson=pearson,
                                 genes=[shared[g] for g in keep_genes]))
        genes_used = sections[0]["genes"] if len(sections) == 1 else shared
        if len(sections) > 1:
            for s in sections:  # per-gene tables need a common gene axis
                if s["genes"] != shared:
                    s["eligible"] = np.asarray([g in set(s["genes"]) for g in shared])
                    _expand_grid_genes(s, shared)
        settings = _settings("grid", counts, genes_used, models, seeds, spacing_um=spacing,
                             transform="sqrt(N_half / (0.5 a)), centered, each gene's halves divided by a common factor",
                             scales="seven physical scales of a two-dimensional Fourier basis: <128, 128–256, 256–512, 512–1,024, 1,024–2,048, 2,048–4,096 and ≥4,096 µm (full spatial cycles)",
                             gene_weighting="common factor of each gene's halves")
        return _grid_result(sections, genes_used, [m.name for m in models], spacing, settings, notes)
    # cells
    if counts.unit != "um":
        notes.append("The coordinates are taken to be micrometers. The cell graph links neighbors within 40 µm.")
    seeds = CELL_SEEDS[: splits or len(CELL_SEEDS)]
    populations = [within] if isinstance(within, str) else list(within or [])
    labels_all = counts.column(cell_class)
    if populations and labels_all is None:
        raise InputError(f"No {cell_class} column, so the cells cannot be scored within a class.")
    views = []
    names = counts.section_names()
    plan = [(name, pop) for name in names for pop in ((["All cells"] if all_cells or not populations else []) + populations)]
    parts = len(seeds) * max(1, math.ceil(len(columns) / CELL_GENE_BLOCK))
    total = len(plan) * len(models) * parts
    for index, (name, population) in enumerate(plan):
        rows = np.flatnonzero((counts.sections == name) & complete)
        block = counts.counts[rows][:, columns].tocsr()
        prediction = {m.name: m.values[np.ix_(rows, columns)].astype(np.float32) for m in models}
        coords = counts.xy[rows]
        if population == "All cells":
            membership, view_coords = sparse.eye(len(rows), format="csr", dtype=np.float32), coords
        else:
            source = np.flatnonzero(np.asarray(labels_all[rows]).astype(str) == str(population))
            if len(source) < CELL_GRAPH["minimum_component_cells"]:
                raise InputError(f"Section {name}: fewer than {CELL_GRAPH['minimum_component_cells']} cells of class {population}.")
            membership = sparse.coo_matrix((np.ones(len(source), np.float32), (np.arange(len(source)), source)),
                                           shape=(len(source), len(rows))).tocsr()
            view_coords = coords[source]
        for k, (model, values) in enumerate(prediction.items()):
            first = (index * len(models) + k) * parts
            say(dict(stage="view", section=name, population=population, model=model, done=first, total=total))

            def cell_step(done, _parts, name=name, population=population, model=model, first=first):
                say(dict(stage="view", section=name, population=population, model=model, done=first + done, total=total))
            try:
                out = score_cell_view(block, values, coords, membership, view_coords, seeds, name, step=cell_step)
            except cell_core.ContractError as error:
                raise InputError(f"Section {name}, {population}: the cell graph could not be built ({error}).") from None
            views.append(dict(model=model, section=name, population=population, out=out, seeds=seeds))
    settings = _settings("cells", counts, shared, models, seeds, transform="log(1 + 2 N_half), centered within graph components, each gene's halves divided by a common factor",
                         scales="lazy symmetric diffusion S on the graph of the 8 nearest neighbors within 40 µm (Gaussian edge weights, components of at least 20 cells): S^16, S^8 − S^16, S^4 − S^8, S^2 − S^4, I − S^2",
                         gene_weighting="common factor of each gene's halves", fine_scale="C, V_model and V_RNA summed over the Fine and Finest scales",
                         populations=["All cells"] * bool(all_cells or not populations) + populations, cell_class_column=cell_class if populations else None)
    return _cell_result(views, shared, settings, notes)


def _expand_grid_genes(section: dict, shared: list[str]) -> None:
    """Put a section's per-gene arrays on the common gene axis, with zeros for genes it left out."""
    index = [shared.index(g) for g in section["genes"]]
    out = section["out"]
    for key in ("vrna_g", "va_g", "vb_g"):
        for band, values in out[key].items():
            full = np.zeros(len(shared))
            full[index] = values
            out[key][band] = full
    for key in ("c_g", "vm_g"):
        for model in out[key]:
            for band, values in out[key][model].items():
                full = np.zeros(len(shared))
                full[index] = values
                out[key][model][band] = full


# ---------------------------------------------------------------- regions

def _expression(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """log(1 + 10,000 N / total), as in the TLS analysis of the paper; a spot with no molecules stays at 0."""
    return np.log1p(1e4 * counts / np.maximum(totals, 1)[:, None])


def _group_means(values: np.ndarray, codes: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The mean of ``values`` over each group, at every location of the group."""
    return (np.bincount(codes, weights=values, minlength=len(sizes)) / sizes)[codes]


def _focus_units(xy: np.ndarray, focus_rows: np.ndarray) -> np.ndarray:
    """Connected groups of focus spots, such as single TLS: focus spots are joined when they are
    neighbors, closer than 1.25 times the median nearest-neighbor distance of the section's spots."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    near = cKDTree(xy).query(xy, k=2)[0][:, 1]
    radius = cal.EDGE_MULTIPLIER * float(np.median(near[near > 0]))
    pts = xy[focus_rows]
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts), len(pts)))
    return connected_components(graph, directed=False)[1]


REGION_TERMS = ("whole", "between", "within", "focus", "focus_between", "focus_within")
REGION_RESAMPLES = 2000   # resamples of focus structures, and of blocks of 10 x 10 spots, as in the paper
REGION_BLOCK = 10


def _region_blocks(xy: np.ndarray, array_row, array_col) -> np.ndarray:
    """Blocks of 10 x 10 spots, the resampling units of the terms that are not within the focus region.
    With the spot grid of Space Ranger these are 10 array rows by 10 spots of a row (20 array columns
    on Visium's staggered grid); otherwise blocks of 10 spot spacings each way, from the coordinates."""
    if array_row is not None and array_col is not None:
        r = np.asarray(array_row, dtype=np.float64)
        q = np.asarray(array_col, dtype=np.float64)
        if np.isfinite(r).all() and np.isfinite(q).all():
            order = np.lexsort((q, r))
            same_row = np.diff(r[order]) == 0
            steps = np.diff(q[order])[same_row]
            step = float(np.min(steps[steps > 0])) if (steps > 0).any() else 1.0
            keys = np.column_stack([np.floor(r / REGION_BLOCK), np.floor(q / (REGION_BLOCK * step))])
            return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)
    from scipy.spatial import cKDTree

    dist, nearest = cKDTree(xy).query(xy, k=2)
    d0 = float(np.median(dist[:, 1][dist[:, 1] > 0]))
    dy = np.abs(xy[nearest[:, 1], 1] - xy[:, 1])
    r0 = float(np.median(dy[dy > 1e-9 * d0])) if (dy > 1e-9 * d0).any() else d0
    keys = np.column_stack([np.floor((xy[:, 1] - xy[:, 1].min()) / (REGION_BLOCK * r0)),
                            np.floor((xy[:, 0] - xy[:, 0].min()) / (REGION_BLOCK * d0))])
    return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)


def _resample(per_spot: dict, codes: np.ndarray, rng, draws: int) -> dict:
    """Means over spots for resamples of whole units (focus structures or blocks): each draw takes as
    many units as there are, with replacement."""
    n = int(codes.max()) + 1
    sums = {key: np.bincount(codes, weights=value, minlength=n) for key, value in per_spot.items()}
    count = np.bincount(codes, minlength=n).astype(np.float64)
    weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]).astype(np.float64)
    denominator = weights @ count
    return {key: (weights @ value) / denominator for key, value in sums.items()}


def _region_parts(v: np.ndarray, region: tuple, units: tuple | None, focus_rows: np.ndarray | None, sd_all: float, sd_focus: float) -> dict:
    """One gene's terms: the whole section, between and within regions, and within the focus region,
    between its structures and within one structure. Each is divided by the standard deviation of the
    gene's unsplit expression over the spots of the term."""
    rm = _group_means(v, *region)
    res = v - rm
    sa = sd_all if sd_all > 0 else np.inf
    out = dict(whole=(v - v.mean()) / sa, between=(rm - v.mean()) / sa, within=res / sa)
    if units is not None:
        st = sd_focus if sd_focus > 0 else np.inf
        rt = res[focus_rows]
        cm = _group_means(rt, *units)
        out.update(focus=rt / st, focus_between=cm / st, focus_within=(rt - cm) / st)
    return out


def _tested(point: np.ndarray, boot: np.ndarray) -> tuple[bool, float, float]:
    """The paper's rule: a term is evaluated only when the lower limit of the 95% interval of its
    split-half reliability is above zero. ``point`` and the rows of ``boot`` are (C, V_model, V_RNA)."""
    testable = bool(np.percentile(boot[:, 2], 2.5) > 0)
    ok = (boot[:, 1] > TOL) & (boot[:, 2] > 0)
    ratios = boot[ok, 0] / np.sqrt(boot[ok, 1] * boot[ok, 2])
    if testable and ok.mean() > 0.9 and point[1] > TOL and point[2] > 0:
        return True, float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))
    return testable, math.nan, math.nan


def regions(counts: Counts | str | Path, predictions, *, region: str = "region", focus: str | None = None,
            genes: Iterable[str] | None = None, splits: int = REGION_SPLITS, resamples: int = REGION_RESAMPLES,
            progress: Progress = None) -> Result:
    """Agreement between and within annotated regions, as for the TLS sections of the paper
    (Methods, "Decomposition by region" and "TLS-associated genes"). Each prediction and each RNA half
    is split into the mean of its region and the residual from that mean. RSC is computed for the
    whole section, between regions (the region means) and within regions (the residuals); when
    ``focus`` names a region such as TLS, also within it (the residuals of its spots), between its
    structures (connected groups of its spots) and within one structure. There is no spatial
    decomposition.

    As in the paper, measured expression is log(1 + 10,000 N / total), the total over all genes, each
    half with its own total; each count split halves the scored genes' counts gene by gene and the
    remaining molecules of each spot together. Every gene of the RNA halves and of the predictions is
    divided by the standard deviation of the gene's unsplit expression over the spots of the term (all
    spots with a region; the focus spots for the focus terms), so predictions should be on the same
    log scale. C, V_model and V_RNA are summed over genes before the ratio. Intervals are 95%
    percentile intervals from ``resamples`` resamples of focus structures (terms within the focus
    region) and of blocks of 10 x 10 spots (the other terms), and a term is scored in a section only
    when the lower limit of the interval of its split-half reliability is above zero. Over several
    sections, the sums of the sections where a term is scored are averaged, in every resample too."""
    if not isinstance(counts, Counts):
        counts = read_counts(counts)
    labels_all = counts.column(region)
    if labels_all is None:
        raise InputError(f"No {region} column: give a region for every spot.")
    models = _models(predictions, counts)
    shared, columns, complete, notes = _common(counts, models, genes)
    say = progress or (lambda _: None)
    labels_text = np.asarray(["" if (isinstance(v, float) and math.isnan(v)) or v is None else str(v) for v in labels_all], dtype=object)
    labelled = labels_text != ""
    if (~labelled & complete).any():
        notes.append(f"{int((~labelled & complete).sum()):,} spots without a region were left out.")
    ceiling = math.log1p(1e4)   # the largest value log(1 + 10,000 N / total) can take
    for m in models:
        top = np.nanmax(m.values[np.ix_(np.flatnonzero(complete & labelled), columns)]) if (complete & labelled).any() else np.nan
        if np.isfinite(top) and top > ceiling + 1e-9:
            notes.append(f"{m.name}: some values are above {ceiling:.2f}, the largest that log(1 + 10,000 N/total) can take, so the "
                         "predictions are probably on another scale. Between and within regions each gene is divided by the "
                         "standard deviation of the measured expression, as in the paper, so predictions should be on that log scale.")
    labels = {"whole": "Whole section", "between": "Between regions", "within": "Within regions"}
    if focus:
        labels.update(focus=f"Within {focus}", focus_between=f"Between {focus} structures", focus_within=f"Within one {focus} structure")
    kinds = [(k, labels[k]) for k in REGION_TERMS if k in labels]
    order = {k: i + 1 for i, (k, _) in enumerate(kinds)}
    array_row_all, array_col_all = counts.column("array_row"), counts.column("array_col")
    term_rows, gene_rows, pooled_in = [], [], {}
    names = counts.section_names()
    for index, name in enumerate(names):
        rows = np.flatnonzero((counts.sections == name) & complete & labelled)
        if len(rows) < 3:
            raise InputError(f"Section {name}: fewer than three spots with a region and every prediction.")
        say(dict(stage="section", section=name, done=index * splits, total=len(names) * splits))
        totals = np.asarray(counts.counts[rows].sum(axis=1)).reshape(-1).astype(np.int64)          # all genes
        c = np.asarray(counts.counts[rows][:, columns].todense(), dtype=np.int32)
        full = _expression(c, totals)
        region_here = labels_text[rows]
        _, codes = np.unique(region_here, return_inverse=True)
        region_groups = (codes, np.bincount(codes).astype(np.float64))
        focus_rows = (region_here == focus) if focus else None
        units = None
        if focus and int(focus_rows.sum()) >= 2:
            unit_codes = _focus_units(counts.xy[rows], focus_rows)
            units = (unit_codes, np.bincount(unit_codes).astype(np.float64))
            notes.append(f"Section {name}: {int(focus_rows.sum()):,} {focus} spots in {len(units[1]):,} structure{'s' if len(units[1]) != 1 else ''}."
                         + (f" With one structure the intervals of the terms within {focus} say nothing, and there is no part between structures." if len(units[1]) == 1 else ""))
        elif focus:
            notes.append(f"Section {name}: fewer than two spots in {focus}, so the terms within {focus} are missing there.")
        terms_here = [(k, label) for k, label in kinds if units is not None or not k.startswith("focus")]
        sizes = {k: (int(focus_rows.sum()) if k.startswith("focus") else len(rows)) for k, _ in terms_here}
        blocks = _region_blocks(counts.xy[rows], None if array_row_all is None else array_row_all[rows],
                                None if array_col_all is None else array_col_all[rows])
        sd_all = full.std(axis=0)
        sd_focus = full[focus_rows].std(axis=0) if units is not None else np.zeros(len(columns))
        del full
        values = {m.name: m.values[np.ix_(rows, columns)] for m in models}
        n_genes = len(columns)
        acc = {k: {q: np.zeros(n_genes) for q in ("VR", "VA", "VB")} for k, _ in terms_here}
        acc_c = {(k, m): np.zeros(n_genes) for k, _ in terms_here for m in values}
        spot = {k: {q: np.zeros(sizes[k]) for q in ("VR", "VA", "VB")} for k, _ in terms_here}     # summed over genes
        spot_c = {(k, m): np.zeros(sizes[k]) for k, _ in terms_here for m in values}
        seeds_c = {(k, m): np.zeros(splits) for k, _ in terms_here for m in values}
        seeds_vr = {k: np.zeros(splits) for k, _ in terms_here}
        rng = np.random.default_rng(cell_core.stable_seed("strsc.regions", name))
        rest = totals - c.sum(axis=1, dtype=np.int64)
        ha = np.empty_like(c)
        for i in range(splits):
            for j in range(n_genes):             # gene by gene, then the other molecules: the order of the draws
                ha[:, j] = rng.binomial(c[:, j], 0.5)
            ta = rng.binomial(rest, 0.5) + ha.sum(axis=1, dtype=np.int64)
            per_a, per_b = np.maximum(ta, 1)[:, None], np.maximum(totals - ta, 1)[:, None]
            for g in range(n_genes):            # each half one gene at a time: the same values as for all genes at once
                a_g = np.log1p(1e4 * ha[:, g:g + 1] / per_a)[:, 0]
                b_g = np.log1p(1e4 * (c[:, g:g + 1] - ha[:, g:g + 1]) / per_b)[:, 0]
                pa = _region_parts(a_g, region_groups, units, focus_rows, sd_all[g], sd_focus[g])
                pb = _region_parts(b_g, region_groups, units, focus_rows, sd_all[g], sd_focus[g])
                pp = {m: _region_parts(values[m][:, g], region_groups, units, focus_rows, sd_all[g], sd_focus[g]) for m in values}
                for k, _ in terms_here:
                    A, B = pa[k], pb[k]
                    ab = A * B
                    vr = float(np.mean(ab))
                    acc[k]["VR"][g] += vr
                    acc[k]["VA"][g] += float(np.mean(A * A))
                    acc[k]["VB"][g] += float(np.mean(B * B))
                    spot[k]["VR"] += ab
                    spot[k]["VA"] += A * A
                    spot[k]["VB"] += B * B
                    seeds_vr[k][i] += vr
                    for m in values:
                        both = 0.5 * (pp[m][k] * A + pp[m][k] * B)
                        cc = float(np.mean(both))
                        acc_c[k, m][g] += cc
                        spot_c[k, m] += both
                        seeds_c[k, m][i] += cc
            say(dict(stage="section", section=name, done=index * splits + i + 1, total=len(names) * splits))
        vm = {(k, m): np.zeros(n_genes) for k, _ in terms_here for m in values}
        spot_vm = {(k, m): np.zeros(sizes[k]) for k, _ in terms_here for m in values}
        for g in range(n_genes):
            for m in values:
                pp = _region_parts(values[m][:, g], region_groups, units, focus_rows, sd_all[g], sd_focus[g])
                for k, _ in terms_here:
                    vm[k, m][g] = float(np.mean(pp[k] * pp[k]))
                    spot_vm[k, m] += pp[k] * pp[k]
        boot_rng = np.random.default_rng(cell_core.stable_seed("strsc.regions.resamples", name))
        for k, label in terms_here:
            vr_g, va_g, vb_g = (acc[k][q] / splits for q in ("VR", "VA", "VB"))
            per_spot = {q: spot[k][q] / splits for q in ("VR", "VA", "VB")}
            for m in values:
                per_spot[("C", m)] = spot_c[k, m] / splits
                per_spot[("VM", m)] = spot_vm[k, m]
            boot = _resample(per_spot, units[0] if k.startswith("focus") else blocks, boot_rng, resamples)
            use = (sd_focus if k.startswith("focus") else sd_all) > 0
            for m in values:
                c_g, vm_g = acc_c[k, m] / splits, vm[k, m]
                point = np.array([c_g.sum(), vm_g.sum(), vr_g.sum(), va_g.sum(), vb_g.sum()])
                bt = np.column_stack([boot[("C", m)], boot[("VM", m)], boot["VR"], boot["VA"], boot["VB"]])
                testable, lo, hi = _tested(point, bt)
                row = _record(m, name, label, order[k], *point, seeds_c[k, m], seeds_vr[k], term=k, locations=sizes[k])
                row.update(testable=testable, rsc_ci_low=lo, rsc_ci_high=hi, V_RNA_ci_low=float(np.percentile(bt[:, 2], 2.5)),
                           V_RNA_ci_high=float(np.percentile(bt[:, 2], 97.5)))
                if not testable:
                    row.update(rsc=math.nan, status="NOT_TESTABLE")
                term_rows.append(row)
                gene_rows.extend({**r, "term": k} for r in _gene_rows(m, name, shared, [label], c_g[None, :], vm_g[None, :],
                                                                       vr_g[None, :], va_g[None, :], vb_g[None, :], use))
                pooled_in.setdefault((m, k, label), []).append((testable, point, bt))
    scales = pd.DataFrame(term_rows)
    if len(names) > 1:
        pooled = []
        for (model, kind, label), entries in pooled_in.items():
            have = [(p_, b_) for t_, p_, b_ in entries if t_]
            if not have:
                continue
            point = np.mean([p_ for p_, _ in have], axis=0)
            bt = np.mean([b_ for _, b_ in have], axis=0)
            testable, lo, hi = _tested(point, bt)
            row = _record(model, "All sections", label, order[kind], *point, term=kind, sections=len(have))
            row.update(testable=testable, rsc_ci_low=lo, rsc_ci_high=hi, V_RNA_ci_low=float(np.percentile(bt[:, 2], 2.5)),
                       V_RNA_ci_high=float(np.percentile(bt[:, 2], 97.5)))
            if not testable:
                row.update(rsc=math.nan, status="NOT_TESTABLE")
            pooled.append(row)
        scales = pd.concat([scales, pd.DataFrame(pooled)], ignore_index=True)
    # every term gets a column, also where it was scored in no section
    summary_rows = {}
    for _, r in scales.iterrows():
        summary_rows.setdefault((r["model"], r["section"]), {"model": r["model"], "section": r["section"]})[r["term"]] = r["rsc"]
    summary = pd.DataFrame(list(summary_rows.values())).reindex(columns=["model", "section"] + [k for k, _ in kinds])
    settings = _settings("regions", counts, shared, models, tuple(range(splits)), region_column=region, focus=focus,
                         transform="log(1 + 10,000 N / total over all genes), each half with its own total; each split halves the scored genes' counts gene by gene and the other molecules of each spot together",
                         terms=[label for _, label in kinds],
                         structures=(f"connected groups of {focus} spots: neighbors closer than 1.25 times the median nearest-neighbor distance" if focus else None),
                         gene_weighting="every gene of the RNA halves and of the predictions divided by the standard deviation of its unsplit expression over the spots of the term",
                         intervals=f"95% percentile intervals from {resamples:,} resamples of {focus or 'focus'} structures (terms within {focus or 'the focus region'}) and of blocks of 10 x 10 spots (other terms); a term is scored only when the lower limit for its V_RNA is above zero",
                         pooling="mean over the sections where a term is scored, in every resample too")
    return Result("regions", scales, summary, pd.DataFrame(gene_rows), settings, notes)


# ---------------------------------------------------------------- the paper's test with the expected expression

def _expected(values: np.ndarray, transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """E[transform(X)] for X ~ Poisson(N / 2), for every integer N in ``values``: the expected
    transformed half-count when the measured counts N are taken as Poisson rates."""
    from scipy import stats

    unique, inverse = np.unique(values, return_inverse=True)
    out = np.zeros(len(unique))
    for i, n in enumerate(unique):
        mu = n / 2.0
        if mu == 0:
            out[i] = float(transform(np.zeros(1))[0])
            continue
        k = np.arange(0, int(math.ceil(mu + 12 * math.sqrt(mu) + 30)) + 1)
        out[i] = float(np.sum(stats.poisson.pmf(k, mu) * transform(k.astype(np.float64))))
    return out[inverse].reshape(values.shape)


def ideal_check(counts: Counts | str | Path, layout: str = "spots", *, draws: int = 3, spacing_um: float | None = None,
                cell_class: str = "cell_class", within: str | Sequence[str] | None = None, genes: Iterable[str] | None = None,
                splits: int | None = None, seed: int = 20261106, progress: Progress = None) -> pd.DataFrame:
    """The paper's test of RSC (Methods, "Expected expression as a prediction"): the measured counts are
    taken as Poisson rates, count matrices are drawn from them, and on each the expected transformed
    half-count is scored as the prediction. Its RSC should be near 1 at every scale; in the paper it was
    0.956 to 1.000. Returns the mean, lowest and highest RSC over the draws at each scale."""
    if not isinstance(counts, Counts):
        counts = read_counts(counts)
    columns = np.arange(len(counts.genes)) if genes is None else np.asarray([counts.genes.index(g) for g in genes])
    gene_names = [counts.genes[i] for i in columns]
    dense = np.asarray(counts.counts[:, columns].todense(), dtype=np.int64)
    if layout == "grid":
        spacing = float(spacing_um or counts.spacing_um or 0)
        area = counts.column("area")
        area = np.asarray(area, dtype=float) if area is not None else np.full(len(counts), spacing * spacing)
        root = _expected(dense, np.sqrt)
        expected = root / np.sqrt(0.5 * area)[:, None]
    else:
        expected = _expected(dense, lambda k: np.log1p(2.0 * k))
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(draws):
        if progress:
            progress(dict(stage="draw", done=d, total=draws))
        drawn = rng.poisson(dense).astype(np.int64)
        simulated = Counts(sparse.csr_matrix(drawn), list(gene_names), counts.ids, counts.sections, counts.xy, counts.unit,
                           counts.columns, counts.spacing_um, "drawn counts")
        result = rsc(simulated, {"Expected expression": expected}, layout, spacing_um=spacing_um, cell_class=cell_class,
                     within=within, splits=splits)
        frame = result.scales[["section", "scale", "scale_order", "rsc"] + (["population"] if "population" in result.scales else [])].copy()
        frame["draw"] = d + 1
        frames.append(frame)
    joined = pd.concat(frames, ignore_index=True)
    keys = ["section"] + (["population"] if "population" in joined else []) + ["scale", "scale_order"]
    out = joined.groupby(keys, as_index=False, sort=False)["rsc"].agg(rsc_mean="mean", rsc_min="min", rsc_max="max")
    out["draws"] = draws
    return out


# ---------------------------------------------------------------- bundle

def _bundle(result: Result, title: str, synthetic: bool) -> dict:
    layout = result.layout
    workflow = {"spots": "hest", "grid": "fourier", "cells": "cell", "regions": "regions"}[layout]
    bundle = evaluation_export.Bundle(workflow, title, synthetic)
    details_common = {k: v for k, v in result.settings.items() if k in ("toolkit", "count_splits", "gene_weighting", "transform")}
    if layout == "spots":
        bundle.metric("pearson_graph_support", "Whole-section Pearson r", "Mean gene-wise Pearson r with log(1 + count) on the connected spots.", "correlation")
        bundle.metric("rsc_broad", "Broad-scale RSC", "Mean RSC at the Broadest and Broad scales, as for the spot-level models of the paper.")
        bundle.metric("rsc_fine", "Fine-scale RSC", "Mean RSC at the Fine and Finest scales, as for the spot-level models of the paper.")
        for row in result.summary.to_dict("records"):
            if row["section"] == "All sections":
                continue
            for field_, metric in (("whole_section_pearson_r", "pearson_graph_support"), ("broad_rsc", "rsc_broad"), ("fine_rsc", "rsc_fine")):
                bundle.row(row["model"], "Sections", row["section"], "specimen", metric, row[field_], "connected spots", None, details_common)
        for k, label in enumerate(RELATIVE, start=1):
            bundle.metric(f"rsc_B{k}", f"RSC, {label}", "Relative scale of the neighbor graph, from B1 (broadest) to B5 (finest).")
        bundle.metric("rna_split_agreement", "Split-half reliability", "Correlation between the two RNA halves at this scale.", "correlation", "split_half")
        for row in result.scales.to_dict("records"):
            if row["section"] == "All sections":
                continue
            details = {**details_common, **{k: row[k] for k in ("C", "V_model", "V_RNA", "V_A", "V_B", "pearson_r", "rsc_split_min", "rsc_split_max")}}
            bundle.row(row["model"], "Sections", row["section"], "specimen", f"rsc_B{row['scale_order']}", row["rsc"], "connected spots", row["status"], details)
            key = ("Split-half reliability", f"Sections | B{row['scale_order']}", row["section"], "rna_split_agreement")
            if key not in bundle.identities:
                bundle.row("Split-half reliability", f"Sections | B{row['scale_order']}", row["section"], "specimen", "rna_split_agreement",
                           row["split_half_reliability"], "connected spots", None, details_common, role="split_half")
    elif layout == "grid":
        bundle.metric("physical_rsc", "RSC by physical scale", "RSC in this window of physical wavelength (a full spatial cycle).")
        bundle.metric("rna_split_agreement", "Split-half reliability", "Correlation between the two RNA halves in this window.", "correlation", "split_half")
        spacing = result.settings.get("spacing_um")
        for row in result.scales.to_dict("records"):
            if row["section"] == "All sections":
                continue
            task = f"{row['band']} | grid {spacing:g} um"
            details = {**details_common, **{k: row[k] for k in ("C", "V_model", "V_RNA", "V_A", "V_B", "pearson_r", "fourier_modes", "partial_window")}}
            bundle.row(row["model"], task, row["section"], "specimen", "physical_rsc", row["rsc"],
                       "occupied grid bins" + ("; partial window" if row.get("partial_window") else ""), row["status"], details)
            key = ("Split-half reliability", task, row["section"], "rna_split_agreement")
            if key not in bundle.identities:
                bundle.row("Split-half reliability", task, row["section"], "specimen", "rna_split_agreement", row["split_half_reliability"],
                           "occupied grid bins", None, details_common, role="split_half")
    elif layout == "cells":
        bundle.metric("cell_rsc", "Cell-level RSC", "RSC on the cell graph of this view and diffusion scale.")
        bundle.metric("rna_split_agreement", "Split-half reliability", "Correlation between the two RNA halves on this cell graph view.", "correlation", "split_half")
        for row in result.scales.to_dict("records"):
            if row["section"] == "All sections":
                continue
            task = f"{row['population']} | B{row['scale_order']}"
            bundle.row(row["model"], task, row["section"], "specimen", "cell_rsc", row["rsc"], "connected cells", row["status"],
                       {**details_common, **{k: row[k] for k in ("C", "V_model", "V_RNA", "V_A", "V_B", "pearson_r")}})
            key = ("Split-half reliability", task, row["section"], "rna_split_agreement")
            if key not in bundle.identities:
                bundle.row("Split-half reliability", task, row["section"], "specimen", "rna_split_agreement", row["split_half_reliability"],
                           "connected cells", None, details_common, role="split_half")
    else:
        bundle.metric("region_rsc", "RSC by region term", "RSC of the whole section, the region means (between), the residuals (within), and within the focus region, between its structures and within one structure. Terms that were not testable have no value.")
        bundle.metric("rna_split_agreement", "Split-half reliability", "Correlation between the two RNA halves in this term.", "correlation", "split_half")
        for row in result.scales.to_dict("records"):
            if row["section"] == "All sections":
                continue
            bundle.row(row["model"], row["scale"], row["section"], "specimen", "region_rsc", row["rsc"], "annotated spots", row["status"],
                       {**details_common, **{k: row[k] for k in ("C", "V_model", "V_RNA", "V_A", "V_B", "pearson_r", "rsc_ci_low", "rsc_ci_high", "testable") if k in row}})
            key = ("Split-half reliability", row["scale"], row["section"], "rna_split_agreement")
            if key not in bundle.identities:
                bundle.row("Split-half reliability", row["scale"], row["section"], "specimen", "rna_split_agreement", row["split_half_reliability"],
                           "annotated spots", None, details_common, role="split_half")
    bundle.data["notes"] = [f"Reproducible-signal correlation (RSC) computed with {result.settings.get('toolkit')}, layout {layout}."] + list(result.notes)
    bundle.data["settings"] = evaluation_export.finite_json(result.settings)
    return bundle.data
