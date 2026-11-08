"""Paired-patient hotspot enrichment under native and smoothed predictions."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA = HERE
SOURCES = {"DeepSpot2Cell": "remote_D2C_coarse", "sCellST": "remote_sCellST_coarse"}
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8A6D3B"]


def main():
    patient_tables, means = [], []
    for model, directory in SOURCES.items():
        patient = pd.read_csv(DATA / directory / "PATIENT_CONTRASTS.tsv", sep="\t")
        cohort = pd.read_csv(DATA / directory / "COHORT_CONTRASTS.tsv", sep="\t")
        if set(patient.model) != {model} or len(patient) != 10:
            raise ValueError("Unexpected final model/patient/denominator inputs")
        patient["source_table"] = str(Path(directory) / "PATIENT_CONTRASTS.tsv")
        patient_tables.append(patient)
        means.append(cohort)
    patients = pd.concat(patient_tables, ignore_index=True)
    cohort = pd.concat(means, ignore_index=True)
    # This is a display table, not a new endpoint or a reweighting of patients.
    patients[["release", "model", "condition", "patient", "denominator", "n_sections",
              "native_contrast", "coarse_contrast", "native_minus_coarse", "jaccard",
              "overlap_fraction", "source_table"]].to_csv(
        HERE / "FINAL_PAIRED_PATIENT_CONTROLS.tsv", sep="\t", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 4.5), sharey=True)
    plotted = patients[patients.denominator.eq("common")]
    low = plotted[["native_contrast", "coarse_contrast"]].min().min()
    high = plotted[["native_contrast", "coarse_contrast"]].max().max()
    span = high - low
    limits = (min(0, low) - 0.09 * span, max(0, high) + 0.17 * span)
    patient_ids = sorted(patients.patient.unique())
    for row_index, denominator in enumerate(["common"]):
        for col_index, model in enumerate(SOURCES):
            ax = axes[col_index]
            frame = patients[patients.model.eq(model) & patients.denominator.eq(denominator)]
            for patient_id, colour in zip(patient_ids, COLOURS):
                record = frame[frame.patient.eq(patient_id)].iloc[0]
                ax.plot([0, 1], [record.native_contrast, record.coarse_contrast],
                        color=colour, marker="o", lw=1.05, ms=4, alpha=0.9)
            mean = cohort[cohort.model.eq(model) & cohort.denominator.eq(denominator)].iloc[0]
            ax.plot([0, 1], [mean.native_contrast, mean.coarse_contrast],
                    color="#222222", marker="D", ms=6, lw=2.15, zorder=10)
            ax.axhline(0, color="#777777", lw=0.65, linestyle=(0, (3, 3)), zorder=0)
            ax.set_xticks([0, 1], ["Unsmoothed", "Smoothed"])
            ax.set_xlim(-0.22, 1.22)
            ax.set_ylim(limits)
            ax.tick_params(width=0.7, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
            ax.set_title(model, fontsize=10.5, loc="left", pad=10)
            text = f"Mean: {mean.native_contrast:+.3f} → {mean.coarse_contrast:+.3f}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
            if col_index == 0:
                ax.set_ylabel("Hotspot enrichment")
            ax.text(-0.13, 1.07, chr(97 + row_index * 2 + col_index),
                    transform=ax.transAxes, fontsize=11, weight="bold")
    handles = [Line2D([0], [0], color=c, marker="o", lw=1, ms=4, label=p.replace("patient_", "P"))
               for p, c in zip(patient_ids, COLOURS)]
    handles.append(Line2D([0], [0], color="#222222", marker="D", lw=2, ms=5, label="Mean"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065),
               ncol=3, frameon=False, fontsize=9, handlelength=1.8, columnspacing=2.2)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.855, bottom=0.29, wspace=0.22)
    fig.savefig(HERE / "FINAL_PAIRED_PATIENT_CONTROLS.png", dpi=220, facecolor="white")
    fig.savefig(HERE / "FINAL_PAIRED_PATIENT_CONTROLS.pdf", facecolor="white")
    plt.close(fig)
    print("Final paired-patient table and figure saved")


if __name__ == "__main__":
    main()
