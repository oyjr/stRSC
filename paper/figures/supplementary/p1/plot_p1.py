"""Rebuild P1 figures from the adjacent CSVs; no scoring, fitting or resampling.

    python plot_p1.py

One-time CSV extraction (optional):
    python plot_p1.py --source-root PATH/TO/metric_concordance

Dependencies: Python 3.10+, numpy, matplotlib. Outputs remain beside this file.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import fsum
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data"
Q = ["FULL_INDEPENDENT", "PATHOLOGY_INDEPENDENT", "MATCHED_ORIGINAL", "MATCHED_SAME_PATHOLOGY"]
LABELS = ["Full section", "Within pathology", "RNA matched", "RNA matched\n+ within pathology"]
COLORS = ["#526F92", "#138A86", "#D68B34", "#785F8D"]
METHODS = ["DeepPT", "DeepSpaCE", "EGNv1", "EGNv2", "GeneCodeR", "HisToGene", "Hist2ST", "ST-Net", "TCGN", "THItoGene", "iStar"]
PATIENTS = list("ABCDEFGH")
INK = "#24272A"
GRAY = "#BBC0C6"


def read(path, delimiter=","):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def mean(values):
    values = list(values)
    return fsum(values) / len(values)


def write(name, rows):
    with (DATA / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_sources(root):
    """Export descriptive arithmetic only from completed, independently checked tables."""
    DATA.mkdir(exist_ok=True)
    patient = read(root / "four_q/FOUR_Q_PATIENT_RESULTS.tsv", "\t")
    shared = read(root / "four_q/SHARED_STRATA_SENSITIVITY.tsv", "\t")
    ranks = read(root / "RAW_C_G_MODEL_RANKS.tsv", "\t")
    lookup = {(r["arena"], r["patient"], r["model"]): float(r["C"]) for r in patient}
    # Coverage is an acceptance condition for this export, not a new scientific test.
    assert len(lookup) == 4 * 8 * 11
    assert set(lookup) == {(q, p, m) for q in Q for p in PATIENTS for m in METHODS}
    write("panel_a_patient_four_q.csv", [
        dict(arena=q, patient=p, C=mean(lookup[q, p, m] for m in METHODS), models=11, split_seeds=10)
        for q in Q for p in PATIENTS
    ])
    write("panel_b_patient_method_change.csv", [
        dict(patient=p, model=m, full_C=lookup[Q[0], p, m], within_pathology_C=lookup[Q[1], p, m],
             delta_C_pp=100 * (lookup[Q[1], p, m] - lookup[Q[0], p, m]))
        for m in METHODS for p in PATIENTS
    ])
    grouped = defaultdict(list)
    for r in shared:
        grouped[(r["arena"], r["patient"], r["model"])].append(float(r["C"]))
    assert set(grouped) == set(lookup) and all(len(v) == 10 for v in grouped.values())
    rows = []
    for q in Q:
        for p in PATIENTS:
            rows.append(dict(arena=q, patient=p, support="Native Q support",
                             C=mean(lookup[q, p, m] for m in METHODS), models=11, split_seeds=10))
            rows.append(dict(arena=q, patient=p, support="Shared strata",
                             C=mean(mean(grouped[q, p, m]) for m in METHODS), models=11, split_seeds=10))
    write("panel_c_shared_strata.csv", rows)
    write("appendix_rank_attribution.csv", ranks)
    write("all_patient_method_scores.csv", [
        dict(arena=q, patient=p, model=m, C=lookup[q, p, m])
        for q in Q for p in PATIENTS for m in METHODS
    ])
    write("support_and_source_judge.csv", [
        {k: r[k] for k in ["arena", "patient", "source_judge_concordance", "selected", "judged", "genes", "strata", "positions", "mean_selected_distance", "mean_abs_source_delta", "mean_source_depth_ratio"]}
        for r in patient if r["model"] == METHODS[0]
    ])
    write("rank_change_attribution.csv", read(root / "RAW_C_G_RANK_ATTRIBUTION.tsv", "\t"))


def style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "font.weight": "normal",
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#7B8086",
        "axes.linewidth": .65, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax, letter, title):
    pos = ax.get_position()
    ax.figure.text(pos.x0 - .052, pos.y1 + .03, letter, fontsize=12, weight="bold", va="bottom")
    ax.set_title(title, loc="left", pad=17, weight="normal")


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main_figure():
    a = read(DATA / "panel_a_patient_four_q.csv")
    b = read(DATA / "panel_b_patient_method_change.csv")
    c = read(DATA / "panel_c_shared_strata.csv")
    lookup = {(r["arena"], r["patient"]): float(r["C"]) for r in a}
    patient = np.array([[lookup[q, p] for q in Q] for p in PATIENTS])
    means = patient.mean(axis=0)
    fig = plt.figure(figsize=(180 / 25.4, 6.45))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.13], height_ratios=[1.15, 1],
                           left=.092, right=.963, bottom=.12, top=.91, hspace=.73, wspace=.49)
    ax = fig.add_subplot(grid[0, 0])
    panel(ax, "a", "Concordance across four pair sets")
    for row in patient:
        ax.plot(range(4), row, color=GRAY, lw=.85, marker="o", ms=3,
                markerfacecolor="white", markeredgewidth=.7, zorder=2)
    ax.plot(range(4), means, color=INK, lw=1.7, zorder=4)
    for i, (value, color) in enumerate(zip(means, COLORS)):
        ax.scatter(i, value, s=39, color=color, edgecolor="white", linewidth=.7, zorder=5)
    ax.axhline(.5, color="#9B9DA1", lw=.8, ls=(0, (3, 3)), zorder=1)
    ax.set(xticks=range(4), xticklabels=["Full", "Within\npathology", "RNA\nmatched", "Matched\n+ within"],
           ylabel="Directional concordance, C", ylim=(.487, .755), xlim=(-.18, 3.18))
    ax.set_yticks([.5, .55, .6, .65, .7, .75])
    ax.legend(handles=[Line2D([0], [0], color=GRAY, marker="o", mfc="white", ms=3, lw=.8, label="Each patient"),
                       Line2D([0], [0], color=INK, marker="o", ms=4, lw=1.7, label="Patient-equal mean")],
              loc="upper right", frameon=False, handlelength=1.5, borderaxespad=.1)
    ax.text(.04, .83, "Within − full: −3.95 pp\nLower in 8/8 patients", transform=ax.transAxes,
            fontsize=8, color=COLORS[1], va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=.94))

    bx = fig.add_subplot(grid[0, 1])
    panel(bx, "b", "Pathology-only change")
    lookup_b = {(r["model"], r["patient"]): float(r["delta_C_pp"]) for r in b}
    change = np.array([[lookup_b[m, p] for p in PATIENTS] for m in METHODS])
    cmap = LinearSegmentedColormap.from_list("change", ["#285878", "#FAFAF8", "#D68B34"])
    im = bx.imshow(change, cmap=cmap, vmin=-20, vmax=20, aspect="auto")
    bx.set(xticks=range(8), xticklabels=PATIENTS, yticks=range(11), yticklabels=METHODS,
           xlabel="Patient")
    bx.tick_params(length=0)
    for spine in bx.spines.values():
        spine.set_visible(False)
    for i in range(11):
        for j in range(8):
            v = change[i, j]
            label = f"{v:+.1f}" if abs(v) >= .05 else "0.0"
            bx.text(j, i, label.replace("-", "−"), ha="center", va="center", fontsize=7,
                    color="white" if v < -12 else INK)
    bx.set_xticks(np.arange(-.5, 8, 1), minor=True)
    bx.set_yticks(np.arange(-.5, 11, 1), minor=True)
    bx.grid(which="minor", color="white", linewidth=.7)
    bx.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=bx, orientation="horizontal", fraction=.052, pad=.23, aspect=28)
    cb.set_ticks([-20, -10, 0, 10, 20], labels=["−20", "−10", "0", "+10", "+20"])
    cb.set_label("Within pathology − full, ΔC (pp)", labelpad=2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7)

    cx = fig.add_subplot(grid[1, :])
    panel(cx, "c", "Four-set ordering persists on shared strata")
    cg = defaultdict(list)
    for r in c:
        cg[(r["arena"], r["support"])].append(float(r["C"]))
    native = np.array([mean(cg[q, "Native Q support"]) for q in Q])
    shared = np.array([mean(cg[q, "Shared strata"]) for q in Q])
    for i, (v, s, color) in enumerate(zip(native, shared, COLORS)):
        cx.plot([i - .13, i + .13], [v, s], color=color, lw=1.4, zorder=2)
        cx.scatter(i - .13, v, marker="o", s=35, color=color, zorder=4)
        cx.scatter(i + .13, s, marker="s", s=34, facecolor="white", edgecolor=color, linewidth=1.4, zorder=4)
        cx.annotate(f"{v:.4f}", (i - .13, v), xytext=(-3, 6), textcoords="offset points", ha="right", fontsize=8)
        cx.annotate(f"{s:.4f}", (i + .13, s), xytext=(3, 6), textcoords="offset points", ha="left", fontsize=8)
    cx.axhline(.5, color="#9B9DA1", lw=.8, ls=(0, (3, 3)), zorder=1)
    cx.set(xticks=range(4), xticklabels=LABELS, ylabel="Patient-equal mean C", xlim=(-.42, 3.43), ylim=(.496, .597))
    cx.set_yticks([.5, .52, .54, .56, .58])
    cx.legend(handles=[Line2D([0], [0], color=INK, marker="o", ls="none", ms=5, label="Native Q support"),
                       Line2D([0], [0], color=INK, marker="s", ls="none", mfc="white", ms=5, label="Shared strata")],
              loc="upper right", frameon=False, ncol=2, handletextpad=.4, columnspacing=1.2)
    fig.text(.092, .031, "8 patients · 11 methods · 10 fixed RNA splits; patient-level averages shown. 0.5 is the half-concordance reference.\nShared strata = common gene × source direction × distance quartile; specific pairs and signal difficulty can differ.",
             fontsize=7.5, color="#565D65", va="bottom", linespacing=1.45)
    save(fig, "p1_four_q")
    return dict(four_q_mean_C={q: float(v) for q, v in zip(Q, means)},
                shared_strata_mean_C={q: float(v) for q, v in zip(Q, shared)},
                pathology_minus_full_pp=float(100 * (means[1] - means[0])),
                patients_with_pathology_decline=int(np.sum(patient[:, 1] < patient[:, 0])),
                patient_method_pairs_with_pathology_decline=int(np.sum(change < 0)))


def rank_figure():
    rows = read(DATA / "appendix_rank_attribution.csv")
    arenas = ["BROAD_STRUCTURE_ALLOWED", "COMPOSITION_MATCHED_ORIGINAL", "JOINT_STRUCTURE_MATCHED"]
    titles = ["Original broad", "Original RNA matched", "Original matched + within"]
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 4.45))
    fig.subplots_adjust(left=.19, right=.96, bottom=.30, top=.84, wspace=1.53)
    for i, (ax, arena, title) in enumerate(zip(axes, arenas, titles)):
        subset = [r for r in rows if r["arena"] == arena]
        changed = sum(float(r["C_rank"]) != float(r["G_rank"]) for r in subset)
        for r in subset:
            c, g = float(r["C_rank"]), float(r["G_rank"])
            color = COLORS[2] if c != g else "#9BA5B0"
            ax.plot([0, 1], [c, g], color=color, lw=1.25 if c != g else .8, marker="o", ms=3.2)
            ax.text(-.08, c, r["model"], ha="right", va="center", fontsize=8, color=INK)
            ax.text(1.09, g, str(int(g)), ha="left", va="center", fontsize=7.5, color="#565D65")
        ax.set(xlim=(-.03, 1.05), ylim=(11.6, .4), xticks=[0, 1], xticklabels=["Raw C", "Existing G"], yticks=[])
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9, loc="center", pad=35)
        ax.text(-1.13, 1.17, "abc"[i], transform=ax.transAxes, fontsize=12, weight="bold")
        ax.text(.5, -.08, f"{changed}/11 methods change rank", transform=ax.transAxes, ha="center", fontsize=8)
    fig.text(.065, .04, "Ranks use patient-equal model means in the three pair sets. Higher scores rank first.\nOrange: changed rank. Tie handling and baseline subtraction leave ranks unchanged;\npatient-specific RNA denominators account for the changes in the matched tasks.",
             fontsize=8, color="#565D65", linespacing=1.5)
    save(fig, "p1_original_rank_attribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.source_root:
        export_sources(args.source_root)
    style()
    numbers = main_figure()
    rank_figure()
    metadata = {
        "status": "COMPLETE", "analysis": "P1 completed four-Q concordance and existing raw-C/G rank attribution",
        "independent_patients": 8, "methods": 11, "fixed_split_seeds": 10,
        "source_run": "runs/STRSC-EVIDENCE-CLOSURE-20260918/metric_concordance",
        "source_tables": ["four_q/FOUR_Q_PATIENT_RESULTS.tsv", "four_q/SHARED_STRATA_SENSITIVITY.tsv", "RAW_C_G_MODEL_RANKS.tsv", "RAW_C_G_RANK_ATTRIBUTION.tsv"],
        "aggregation": "Fixed split seeds averaged within patient and method; methods and then patients weighted equally. No pair-count weighting across patients.",
        "C": "Hierarchically weighted agreement of predicted pair direction with judge-half RNA. Prediction ties receive 0.5; judge-RNA ties are excluded.",
        "reference_0_5": "Half-concordance reference; expected for an independent balanced random pair direction. It is not the observed majority-direction baseline or an RNA-normalized gain.",
        "panels": {"a": "All 8 patients, each averaging 11 methods, on all four pair sets.", "b": "All 88 patient-method pathology-only minus full-section contrasts, percentage points; one decimal shown, full precision in CSV.", "c": "Patient-equal mean C using native versus shared gene/source-direction/distance-quartile strata; all 64 underlying patient means in CSV.", "rank_appendix": "All 11 models in each of the three pair sets; existing C and G ranks only."},
        "interpretation": "Lower concordance on pathology-only pairs occurs without RNA context/depth matching; four-set mean ordering persists with shared-stratum weighting.",
        "boundary": "Question sets also differ in distances, depth, source signals and support. Shared strata do not equate exact pairs or signal difficulty. Differences do not isolate a single causal factor.",
        "statistics": "Descriptive only. No new tests, P values, bootstrap, scoring, training or endpoints.",
        "numbers": numbers,
        "render": "Arial; 180 mm-wide vector PDF/SVG with editable SVG text; 300 dpi PNG. Run python plot_p1.py from any directory.",
        "visual_check": "One visual inspection of both PNGs, followed by corrections to panel-letter alignment, a redundant reference label and appendix footer spacing. No review loop.",
    }
    (HERE / "METADATA.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"figures": ["p1_four_q", "p1_original_rank_attribution"], **numbers}, indent=2))


if __name__ == "__main__":
    main()
