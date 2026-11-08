"""New same-support classical metrics and the unchanged FASN case.

All supplied metric values are read from completed results. Only equal-model
patient display means and average-rank spatial display colors are calculated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
SUPPORT = "ALL_METRICS_ALL_MODELS_DEFINED"


def source_data():
    raw = pd.read_csv(HERE / "SourceData_classical_patient_models.tsv", sep="\t")
    raw = raw.loc[raw.aggregation_support.eq(SUPPORT)].copy()
    assert raw.groupby(["patient", "scope"]).size().eq(11).all()
    assert set(raw.patient) == set("ABCDEFGH")
    paired = raw.groupby(["patient", "scope"], sort=True)[["spearman", "C"]].mean().reset_index()
    rows = []
    for panel, metric in [("g", "spearman"), ("h", "C")]:
        for row in paired.itertuples():
            rows.append(dict(panel=panel, statistic=metric, patient=row.patient,
                             scope=row.scope, value=getattr(row, metric),
                             unit="equal mean of 11 unchanged models", n_models=11))
        for scope in ["WHOLE", "WITHIN_REGION"]:
            value = float(paired.loc[paired.scope.eq(scope), metric].mean())
            rows.append(dict(panel=panel, statistic=metric, patient="EQUAL_PATIENT_MEAN",
                             scope=scope, value=value, unit="equal mean of 8 patients", n_models=11))
    pd.DataFrame(rows).to_csv(HERE / "SourceData_g_h.tsv", sep="\t", index=False)
    spatial = pd.read_csv(HERE / "SourceData_i_spatial.tsv", sep="\t")
    for raw_col, display_col in [("RNA_full_count", "RNA_rank_fraction"),
                                 ("prediction_native", "prediction_rank_fraction")]:
        spatial[display_col] = rankdata(spatial[raw_col], method="average") / (len(spatial) + 1)
    spatial.to_csv(HERE / "SourceData_i_spatial.tsv", sep="\t", index=False)
    cases = pd.read_csv(HERE / "SourceData_i_cases.tsv", sep="\t")
    case = cases.loc[cases.role.eq("candidate")].iloc[0]
    assert (case.gene, case.model, case.patient, case.section) == ("FASN", "Hist2ST", "B", "B1")
    return paired, spatial, case


def add_panels(fig, style):
    paired, spatial, case = source_data()
    width, height = fig.get_size_inches()
    ink, grid = style.COL_TEXT, style.COL_GRID

    def text(x, y, label, **kwargs):
        return fig.text(x / width, y / height, label, color=ink,
                        fontsize=kwargs.pop("fontsize", 7.0), **kwargs)

    def panel_header(x, letter, title, title_offset=.17):
        text(x, 1.79, letter, fontsize=10, weight="bold", va="center")
        text(x + title_offset, 1.79, title, fontsize=8, va="center")

    axes = []
    summaries = {}
    for panel, left, metric, title, limits, ticks, reference in [
        ("g", .77, "spearman", "Spearman ρ", (-.01, .26), [0, .10, .20], 0),
        ("h", 2.51, "C", "Concordance C", (.49, .71), [.50, .60, .70], .5),
    ]:
        panel_header(left, panel, title)
        ax = fig.add_axes([left / width, .59 / height, 1.29 / width, .94 / height])
        axes.append(ax)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(style.COL_SPINE)
            ax.spines[side].set_linewidth(.6)
        ax.set_xlim(-.25, 1.25)
        ax.set_ylim(limits)
        ax.set_yticks(ticks)
        ax.set_xticks([0, 1], ["Whole", "Within\nregions"])
        ax.tick_params(axis="both", labelsize=7, length=0, pad=3, colors=ink)
        ax.grid(axis="y", color=grid, linewidth=.42)
        ax.axhline(reference, color=style.COL_ZERO, lw=.6, ls=(0, (2, 2)), zorder=1)
        for patient in style.PATIENTS:
            selected = paired.loc[paired.patient.eq(patient)].set_index("scope")
            values = selected.loc[["WHOLE", "WITHIN_REGION"], metric].to_numpy()
            item = style.PATIENT_STYLE[patient]
            ax.plot([0, 1], values, color=item["color"], alpha=.35, lw=.55, zorder=2)
            ax.scatter([0, 1], values, s=12, marker=item["marker"],
                       facecolor=item["color"] if item["filled"] else "white",
                       edgecolor=item["color"], linewidth=.45, zorder=3)
        means = paired.groupby("scope")[metric].mean().loc[["WHOLE", "WITHIN_REGION"]].to_numpy()
        summaries[metric] = means
        ax.plot([0, 1], means, ls="none", marker="_", markersize=10,
                markeredgewidth=1.65, color=style.COL_SPINE, zorder=5)
        # Equal patient means are descriptive; no replacement intervals are shown.
        text(left + .645, .20, f"Means {means[0]:.3f} / {means[1]:.3f}", ha="center", va="center")

    case_left = 4.18
    panel_header(case_left, "i", "FASN · Hist2ST · B1")
    plotted = None
    for left, col, label in [(4.23, "RNA_rank_fraction", "Measured RNA"),
                             (5.68, "prediction_rank_fraction", "Prediction")]:
        ax = fig.add_axes([left / width, .72 / height, 1.24 / width, .75 / height])
        axes.append(ax)
        plotted = ax.scatter(spatial.pixel_x, spatial.pixel_y, c=spatial[col],
                             cmap="viridis", vmin=0, vmax=1, s=4, linewidth=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_axis_off()
        text(left + .62, 1.56, label, ha="center", va="center")
    color_ax = fig.add_axes([4.87 / width, .585 / height, 1.00 / width, .045 / height])
    colorbar = fig.colorbar(plotted, cax=color_ax, orientation="horizontal", ticks=[0, 1])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7, length=0, pad=2, colors=ink)
    text(6.00, .598, "Rank fraction", va="center")
    text(case_left + 1.40, .34,
         f"Whole ρ {case.WHOLE_spearman:.3f}; within ρ {case.WITHIN_REGION_spearman:.3f}",
         ha="center", va="center")
    text(case_left + 1.40, .17,
         f"Whole C {case.WHOLE_C:.3f}; within C {case.WITHIN_REGION_C:.3f}",
         ha="center", va="center")
    return axes, summaries, case
