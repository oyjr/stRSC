"""Programme: frozen patient points and means, with supplied mean CIs only.

Drawing code is the plot_results portion of the archived summarize_and_plot.py.
No aggregation, bootstrap, test, scoring, or archived main function is executed.
Run: python figure_iter1.py
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
import pandas as pd


# === DATA SECTOR (edit here) ===
# Source provenance: original PAIRED_PATIENT.tsv and PAIRED_COHORT.tsv from
# output/stRSC_ProgrammeAlignment_2026-09-09, copied unchanged beside this file.
# These are verbatim frozen plotting columns, not recomputed patient statistics.
PAIRED_PATIENT_TSV = """comparison_id	patient	delta_panel_fine_q	delta_programme_q	delta_ordering_accuracy	delta_common_hotspot_contrast
DeepSpot2Cell_neighbouring_vs_standard	patient_1	0.003681165892694599	-0.06599725735898304	-0.019900923386885205	-0.10562433619152015
DeepSpot2Cell_neighbouring_vs_standard	patient_2	0.0002564189956180997	-0.058770523630090776	0.00733219292657894	-0.00468720115576865
DeepSpot2Cell_neighbouring_vs_standard	patient_3	0.0015748041880377	0.06118518701030824	-0.007017187475405251	0.031665392715501464
DeepSpot2Cell_neighbouring_vs_standard	patient_4	0.004790418578018998	0.03616634566866696	-0.005782757759769774	0.2406066224529818
DeepSpot2Cell_neighbouring_vs_standard	patient_5	0.003152218770544904	-0.3451550767939128	-0.011671358756519812	-0.5183103818358674
DeepSpot2Cell_neighbouring_vs_non_neighbouring	patient_1	0.002116918166298498	-0.059460074037501635	-0.017191783849582976	-0.08796835140574888
DeepSpot2Cell_neighbouring_vs_non_neighbouring	patient_2	0.0006972758244139995	-0.03526223243148663	0.0006147738652177992	0.0027189278959124374
DeepSpot2Cell_neighbouring_vs_non_neighbouring	patient_3	0.0021955501062667	0.03923245736663837	-0.004302950557193652	0.04516803962994542
DeepSpot2Cell_neighbouring_vs_non_neighbouring	patient_4	0.007142740515118796	0.09499878371822804	0.004065844847018774	0.287256926521355
DeepSpot2Cell_neighbouring_vs_non_neighbouring	patient_5	0.000375973152628703	-0.31671339919711905	-0.01361259038878515	-0.5636534360026288
sCellST_neighbouring_vs_standard	patient_1	0.0033222950422756004	0.026521085575758663	0.007079063072225056	0.02968068943900168
sCellST_neighbouring_vs_standard	patient_2	-0.0029101435287642004	0.06254917282283312	0.010119253859114496	0.05379783512117842
sCellST_neighbouring_vs_standard	patient_3	-0.0077548043080754	0.023300916496702344	0.0012453909070947455	0.0002368956729634497
sCellST_neighbouring_vs_standard	patient_4	-0.0094661810049219	-0.05060161313220157	-0.007277093388950107	-0.012340511087910733
sCellST_neighbouring_vs_standard	patient_5	-0.0005431481063764999	0.006392357032818541	-0.0022014438020719895	-0.021391164055667933
sCellST_neighbouring_vs_non_neighbouring	patient_1	0.0016195698675417003	-0.00753264065970094	-0.000396121078859224	-0.0002498052889010616
sCellST_neighbouring_vs_non_neighbouring	patient_2	-0.0080842364478475	0.020501192093778098	-0.001784081087521927	0.007174107794291673
sCellST_neighbouring_vs_non_neighbouring	patient_3	-0.005938802669276	0.0046867431458625375	0.0024662855314347976	0.005502036107920147
sCellST_neighbouring_vs_non_neighbouring	patient_4	-0.010421626472236604	-0.08047307008917665	-0.0064852071828260804	-0.052686824816415845
sCellST_neighbouring_vs_non_neighbouring	patient_5	-0.0037116206908357	0.06339234106229757	0.002655510977007003	0.14774718274358667
sCellST_phikon_v2_vs_resnet50	patient_1	0.02064736174492	-0.10795782090227647	0.0015423102417169154	-0.24384730012496858
sCellST_phikon_v2_vs_resnet50	patient_2	0.002712804049300001	-0.06479772303278322	0.041331701109425134	-0.12134697845752629
sCellST_phikon_v2_vs_resnet50	patient_3	-0.01033868681704	0.17718042113064225	0.0027689608694825285	0.06970525397251465
sCellST_phikon_v2_vs_resnet50	patient_4	0.009871795266399995	-0.1314256530447117	-0.018569415538982836	-0.08385021104068491
sCellST_phikon_v2_vs_resnet50	patient_5	0.0466600815256	-0.6828650408493535	-0.024458555217211575	-0.4847940502801664
"""

PAIRED_COHORT_TSV = """comparison_id	delta_panel_fine_q	delta_programme_q	delta_ordering_accuracy	delta_common_hotspot_contrast
DeepSpot2Cell_neighbouring_vs_standard	0.00269100528498286	-0.07451426502080227	-0.0074080068904002205	-0.07126998080293459
DeepSpot2Cell_neighbouring_vs_non_neighbouring	0.0025056915529453393	-0.05544089291624819	-0.006085341216665041	-0.06329557867223298
sCellST_neighbouring_vs_standard	-0.0034703963811724805	0.01363238375918222	0.0017930341294824404	0.009996749017912976
sCellST_neighbouring_vs_non_neighbouring	-0.005307343282530821	0.00011491311061212484	-0.0007087225681530862	0.021497339308096317
sCellST_phikon_v2_vs_resnet50	0.013910671153835999	-0.16197316333969652	0.0005230002928860334	-0.1728266571861663
"""

# Verbatim source columns: mean_difference, ci95_low, ci95_high, p_holm_20.
# Pointwise approximate two-sided 95% patient-block percentile bootstrap CIs;
# 20,000 draws were computed upstream. They are not simultaneous intervals or
# exact-test inversions. No test or bootstrap is computed by this drawing script.
INTERVAL_SOURCE = "inputs/statistics/PROGRAMME_INFERENCE.tsv"
INTERVAL_TSV = """comparison_id	endpoint	mean_difference	ci95_low	ci95_high	p_holm_20
DeepSpot2Cell_neighbouring_vs_standard	delta_panel_fine_q	0.0026910052849828002	0.0012050454135172401	0.0040190775423943406	1
DeepSpot2Cell_neighbouring_vs_standard	delta_programme_q	-0.074514265020802273	-0.21161388166863243	0.030744929868121697	1
DeepSpot2Cell_neighbouring_vs_standard	delta_ordering_accuracy	-0.0074080068904001997	-0.014747429022293201	0.00066160650956234026	1
DeepSpot2Cell_neighbouring_vs_standard	delta_common_hotspot_contrast	-0.07126998080293459	-0.3128611095638279	0.12230166600233135	1
DeepSpot2Cell_neighbouring_vs_non_neighbouring	delta_panel_fine_q	0.0025056915529452798	0.00085268322407671998	0.00486420949520734	1
DeepSpot2Cell_neighbouring_vs_non_neighbouring	delta_programme_q	-0.055440892916248187	-0.19407356285244404	0.050385722636956246	1
DeepSpot2Cell_neighbouring_vs_non_neighbouring	delta_ordering_accuracy	-0.0060853412166650394	-0.013182339806785921	0.0010116573734558398	1
DeepSpot2Cell_neighbouring_vs_non_neighbouring	delta_common_hotspot_contrast	-0.063295578672232969	-0.33710449044321233	0.15530427121084572	1
sCellST_neighbouring_vs_standard	delta_panel_fine_q	-0.0034703963811724601	-0.0074704228309517605	0.00052963006860683995	1
sCellST_neighbouring_vs_standard	delta_programme_q	0.013632383759182198	-0.020396567464828725	0.043468158399604016	1
sCellST_neighbouring_vs_standard	delta_ordering_accuracy	0.0017930341294824202	-0.0035423366949898596	0.0071284049539547005	1
sCellST_neighbouring_vs_standard	delta_common_hotspot_contrast	0.0099967490179129568	-0.013445290922838759	0.035746736742925216	1
sCellST_neighbouring_vs_non_neighbouring	delta_panel_fine_q	-0.0053073432825308199	-0.0086121473110785992	-0.00140377914718538	1
sCellST_neighbouring_vs_non_neighbouring	delta_programme_q	0.00011491311061209986	-0.045690131766690539	0.037657651681185803	1
sCellST_neighbouring_vs_non_neighbouring	delta_ordering_accuracy	-0.00070872256815307988	-0.0037546834209130396	0.0019694943876048402	1
sCellST_neighbouring_vs_non_neighbouring	delta_common_hotspot_contrast	0.021497339308096303	-0.029076866109407146	0.090033170147230085	1
sCellST_phikon_v2_vs_resnet50	delta_panel_fine_q	0.01391067115383598	-0.0015311789313799998	0.031052449657192005	1
sCellST_phikon_v2_vs_resnet50	delta_programme_q	-0.16197316333969652	-0.44427013329662407	0.058431557888987698	1
sCellST_phikon_v2_vs_resnet50	delta_ordering_accuracy	0.00052300029288604032	-0.016902726254134342	0.021638929731754998	1
sCellST_phikon_v2_vs_resnet50	delta_common_hotspot_contrast	-0.1728266571861663	-0.33941522155111037	-0.023716349849621919	1
"""

COMPARISONS = [
    ("DeepSpot2Cell_neighbouring_vs_standard", "training", "DeepSpot2Cell", "neighbouring", "standard"),
    ("DeepSpot2Cell_neighbouring_vs_non_neighbouring", "training", "DeepSpot2Cell", "neighbouring", "non_neighbouring"),
    ("sCellST_neighbouring_vs_standard", "training", "sCellST", "neighbouring", "standard"),
    ("sCellST_neighbouring_vs_non_neighbouring", "training", "sCellST", "neighbouring", "non_neighbouring"),
    ("sCellST_phikon_v2_vs_resnet50", "encoder", "sCellST", "phikon_v2", "resnet50"),
]
ENDPOINTS = ["delta_panel_fine_q", "delta_programme_q", "delta_ordering_accuracy", "delta_common_hotspot_contrast"]
DISPLAY_FACTORS = {endpoint: 1 for endpoint in ENDPOINTS}
UNITS = {"delta_panel_fine_q": "q units", "delta_programme_q": "q units",
         "delta_ordering_accuracy": "accuracy fraction",
         "delta_common_hotspot_contrast": "normalised log-expression contrast units"}
# === END DATA SECTOR ===


def plot_results(paired_patients, paired_cohort, output):
    """Original drawing function with only supplied CI marks and floor checks added."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    patient_names = sorted(paired_patients.patient.unique())
    # L1-PIL exact marker-centre samples, also identical to frozen plot_results.
    colours = ["#33658A", "#B5654A", "#55856D", "#81719B", "#A78B45"]
    # L1 frozen mean colour; new CI shares the mean's black visual role.
    mean_colour = "#171717"
    offsets = np.linspace(-0.18, 0.18, len(patient_names))
    fig, axes = plt.subplots(1, 4, figsize=(260 / 25.4, 6.3), sharey=True)
    titles = ["Original panel-wide\nFine + Finest recovery", "Aligned programme\nContinuous recovery",
              "Aligned programme\nLocal ordering", "Aligned programme\nHotspot enrichment"]
    xlabels = ["Change in panel q", "Change in programme q",
               "Change in direction agreement", "Change in RNA contrast"]
    labels = ["DeepSpot2Cell\nNeighbouring − standard", "DeepSpot2Cell\nNeighbouring − non-neighbouring",
              "sCellST\nNeighbouring − standard", "sCellST\nNeighbouring − non-neighbouring",
              "sCellST\nPhikon-v2 − ResNet50"]
    interval_rows = list(csv.DictReader(StringIO(INTERVAL_TSV), delimiter="\t"))
    ci_lookup = {(row["comparison_id"], row["endpoint"]): row for row in interval_rows}
    plotted_intervals, plotted_points, plotted_means = [], [], []
    mean_source_differences, original_limits = [], []
    plotted_rails = []
    # L1 reviewer correction: a shared rail below the fixed patient strip.
    # The y axis is inverted; positive offset is downward. No x data change.
    ci_y_offset = 0.33

    for panel, (ax, endpoint, title, xlabel) in enumerate(zip(axes, ENDPOINTS, titles, xlabels), 1):
        # L1 grey zero reference (not a general vertical grid).
        ax.axvline(0, color="0.5", linewidth=0.8, zorder=0)
        values = paired_patients[endpoint].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        width = max(float(np.max(np.abs(finite))) * 1.23 if finite.size else 0.1, 1e-4)
        original_limits.append((-width, width))
        # Layout-only extent check: never trim existing limits or omit negatives.
        ci_extent = max(abs(float(row[key])) for row in interval_rows
                        if row["endpoint"] == endpoint for key in ("ci95_low", "ci95_high"))
        width = max(width, ci_extent * 1.05)
        ax.set_xlim(-width, width)
        ax.set_axisbelow(True)
        for y, (comparison_id, *_rest) in enumerate(COMPARISONS):
            part = paired_patients[paired_patients.comparison_id == comparison_id].set_index("patient")
            # L1 frozen pale row guides (#ebebeb before raster anti-aliasing).
            ax.axhline(y, color="0.92", linewidth=0.6, zorder=0)
            ci = ci_lookup[(comparison_id, endpoint)]
            estimate, lo, hi = [float(ci[key]) for key in ("mean_difference", "ci95_low", "ci95_high")]
            assert lo <= estimate <= hi
            assert float(ci["p_holm_20"]) == 1.0
            factor = DISPLAY_FACTORS[endpoint]
            assert factor == 1
            # CIs use their true x endpoints on a consistently displaced rail.
            # This clears every unchanged patient/mean glyph without stretching
            # a narrow interval or moving its mean diamond.
            ci_y = y + ci_y_offset
            rail = ax.hlines(ci_y, lo, hi, color=mean_colour, linewidth=1.0, zorder=2)
            caps, = ax.plot([lo, hi], [ci_y, ci_y], linestyle="None", marker="|", markersize=8,
                            markeredgewidth=1.0, color=mean_colour, zorder=2)
            plotted_rails.append((ax, comparison_id, endpoint, ci_y, lo, hi, rail, caps))
            for patient, colour, offset in zip(patient_names, colours, offsets):
                value = float(part.loc[patient, endpoint])
                assert np.isfinite(value)
                artist = ax.scatter(value, y + offset, color=colour, s=30, linewidths=0, zorder=3)
                plotted_points.append((ax, comparison_id, endpoint, patient, value, y + offset, artist))
            mean_row = paired_cohort[paired_cohort.comparison_id == comparison_id].iloc[0]
            mean = float(mean_row[endpoint])
            assert np.isfinite(mean)
            # The original mean-diamond coordinate is retained, including its
            # source decimal precision; CI estimate agrees within 6e-17.
            artist = ax.scatter(mean, y, marker="D", color=mean_colour, s=46,
                                edgecolors="white", linewidths=0.6, zorder=4)
            plotted_means.append((ax, comparison_id, endpoint, mean, y, artist))
            mean_source_differences.append(abs(mean - estimate))
            plotted_intervals.append({
                "source_file": INTERVAL_SOURCE,
                "source_key": f"comparison_id={comparison_id};endpoint={endpoint}",
                "comparison_id": comparison_id, "endpoint": endpoint,
                "estimate": ci["mean_difference"], "lower": ci["ci95_low"], "upper": ci["ci95_high"],
                "displayed_unit_factor": factor, "panel": panel, "panel_title": title.replace("\n", " / "),
                "units": UNITS[endpoint], "axis_min": -width, "axis_max": width,
                "original_mean_diamond": repr(mean), "p_holm_20": ci["p_holm_20"],
                "mean_row_y": y, "ci_rail_y": ci_y, "ci_rail_offset_rows": ci_y_offset,
            })
        ax.set_title(title, fontsize=9.5, fontweight="normal", pad=12)
        ax.set_xlabel(xlabel, labelpad=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(COMPARISONS)), labels)
    axes[0].tick_params(axis="y", pad=12)
    axes[0].set_ylim(len(COMPARISONS) - 0.55, -0.55)
    handles = [Line2D([], [], color=colour, marker="o", linestyle="None", markersize=5, label=str(patient))
               for patient, colour in zip(patient_names, colours)]
    handles.append(Line2D([], [], color=mean_colour, marker="D", linestyle="None", linewidth=1,
                          markersize=5, label="Mean (diamond); 95% CI rail below"))
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, 0.05), ncol=3,
                        frameon=False, columnspacing=1.8, handletextpad=0.5, fontsize=8)
    fig.suptitle("Panel accuracy and programme outcomes", x=0.62, y=0.98, fontsize=12)
    left = 0.245
    footer = "Points: patient means, with sections weighted equally. Diamonds: patients weighted equally.\n"
    footer += "Positive changes favour the first condition. Undefined results remain undefined."
    # L1 preserved grey footer; interval definition is in the updated legend.
    fig.text(left, 0.012, footer, fontsize=7.3, color="0.3")
    fig.subplots_adjust(left=left, right=0.985, top=0.84, bottom=0.25, wspace=0.3)
    for text in fig.findobj(match=Text):
        text.set_zorder(8)
    legend.set_zorder(9)

    check_lines = floor_checks(fig, axes, plotted_intervals, plotted_points, plotted_means,
                              original_limits, mean_source_differences, plotted_rails)
    # Export at the manuscript landscape width without shrinking the type.
    fig.savefig(output / "figure_iter1.pdf")
    fig.savefig(output / "img_iter1.png", dpi=220)
    fig.savefig(output / "figure.svg")
    with (output / "intervals_plotted.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted_intervals[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(plotted_intervals)
    (output / "floor_selfcheck_iter1.txt").write_text("\n".join(check_lines) + "\n", encoding="utf-8")
    plt.close(fig)
    print("\n".join(check_lines))


def floor_checks(fig, axes, intervals, points, means, original_limits, mean_differences, rails):
    """Bounded geometry/source checks only; not a statistical analysis."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_records = []
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            view_lo, view_hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if not view_lo <= tick.get_loc() <= view_hi:
                    continue
                for text in (tick.label1, tick.label2):
                    if text.get_visible() and text.get_text():
                        text_records.append(("tick", text))
        text_records += [("title", ax.title), ("xlabel", ax.xaxis.label)]
        text_records += [("annotation", text) for text in ax.texts if text.get_text()]
    text_records += [("figure", text) for text in fig.texts if text.get_text()]
    for legend in fig.legends:
        text_records += [("legend", text) for text in legend.get_texts()]
    boxes = [(kind, text, text.get_window_extent(renderer)) for kind, text in text_records]
    overlaps = []
    for index, (kind_a, text_a, box_a) in enumerate(boxes):
        for kind_b, text_b, box_b in boxes[index + 1:]:
            if box_a.overlaps(box_b):
                overlaps.append(f"{kind_a} {text_a.get_text()!r} / {kind_b} {text_b.get_text()!r}")
    assert not overlaps, "Text overlaps: " + "; ".join(overlaps)
    clipped = [text.get_text() for _, text, box in boxes
               if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1)]
    assert not clipped, f"Clipped text: {clipped}"
    assert all(text.get_zorder() > 4 for _, text, _ in boxes)
    assert min(text.get_fontsize() for _, text, _ in boxes) >= 7
    assert len(axes) == 4 and len(intervals) == 20 and len(points) == 100 and len(means) == 20
    assert len({(row["comparison_id"], row["endpoint"]) for row in intervals}) == 20
    assert max(mean_differences) < 1e-14
    for ax, _, _, _, value, y, artist in points:
        assert ax.get_xlim()[0] < value < ax.get_xlim()[1]
        assert np.array_equal(np.asarray(artist.get_offsets()), [[value, y]])
    for ax, _, _, value, y, artist in means:
        assert np.array_equal(np.asarray(artist.get_offsets()), [[value, y]])
    for row in intervals:
        assert row["axis_min"] < float(row["lower"]) <= float(row["estimate"]) <= float(row["upper"]) < row["axis_max"]
    # Exact artist-based rail and glyph bounds after the renderer resolves all
    # transforms. Check the entire capped rail against every patient and mean.
    glyph_boxes = []
    for row in points + means:
        ax, artist = row[0], row[-1]
        path_box = artist.get_paths()[0].get_extents(Affine2D(artist.get_transforms()[0]))
        centre = artist.get_offset_transform().transform(artist.get_offsets()[0])
        edge_pad = max(artist.get_linewidths(), default=0) * fig.dpi / 144
        glyph_box = Bbox.from_extents(path_box.x0 + centre[0], path_box.y0 + centre[1],
                                     path_box.x1 + centre[0], path_box.y1 + centre[1]).padded(edge_pad)
        glyph_boxes.append((ax, glyph_box))
    rail_glyph_gaps = []
    for ax, _, _, ci_y, lo, hi, rail, caps in rails:
        assert np.array_equal(rail.get_segments()[0], [[lo, ci_y], [hi, ci_y]])
        assert np.array_equal(caps.get_xdata(), [lo, hi])
        assert np.array_equal(caps.get_ydata(), [ci_y, ci_y])
        rail_box = caps.get_window_extent(renderer).padded(0.5 * fig.dpi / 72)
        assert ax.bbox.contains(rail_box.x0, rail_box.y0) and ax.bbox.contains(rail_box.x1, rail_box.y1)
        for glyph_ax, glyph_box in glyph_boxes:
            if glyph_ax is not ax:
                continue
            assert not rail_box.overlaps(glyph_box), "CI rail overlaps an unchanged patient/mean glyph."
            if rail_box.x0 <= glyph_box.x1 and glyph_box.x0 <= rail_box.x1:
                gap = max(glyph_box.y0 - rail_box.y1, rail_box.y0 - glyph_box.y1)
                rail_glyph_gaps.append(gap * 72 / fig.dpi)
    assert len(rails) == 20
    negative_count = sum(value < 0 for _, _, _, _, value, _, _ in points)
    # Calculate the exact original tight-export geometry before serialization.
    tight = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
    gutters = [axes[index + 1].bbox.x0 - axes[index].bbox.x1 for index in range(3)]
    unchanged_limits = all(tuple(ax.get_xlim()) == old for ax, old in zip(axes, original_limits))
    return [
        "text_bbox_pairwise_disjoint: PASS",
        "annotation_tick_overlap: PASS (no data annotations added)",
        "cross_panel_text_bleed: PASS",
        "clipped_labels: PASS",
        "text_obscured_by_marks: PASS (text zorder 8; CI 2, patients 3, mean diamonds 4)",
        "panel_topology: PASS (4 columns x 5 comparisons)",
        f"frozen_patient_points: PASS ({len(points)}; all {negative_count} negative points retained)",
        "original_mean_diamond_equality: PASS (20/20 exact original source coordinates)",
        f"mean_source_estimate_agreement: PASS (max abs difference {max(mean_differences):.17g})",
        "supplied_ci_rows: PASS (20 unique original comparison_id/endpoint keys)",
        "all_points_and_intervals_inside_limits: PASS",
        "ci_rail_artist_endpoints: PASS (20 exact source lower/upper x coordinates)",
        "ci_rails_unobscured: PASS (20/20 capped rail boxes disjoint from all unchanged glyph boxes)",
        f"minimum_rail_glyph_vertical_gap_pt: {min(rail_glyph_gaps):.6f}",
        "ci_rail_offset_rows: 0.33 (consistent downward offset for all 20 intervals)",
        f"original_per_column_limits_unchanged: {'PASS' if unchanged_limits else 'EXPANDED_FOR_CI'}",
        "local_ordering_units: PASS (raw fraction, displayed_unit_factor=1)",
        "holm_adjustment_display: PASS (all 20 source adjusted P=1; no stars or significance claims)",
        f"minimum_visible_font_pt: PASS ({min(text.get_fontsize() for _, text, _ in boxes):.1f})",
        "pdf_fonttype: PASS (42; vector text and marks)",
        f"agg_tight_extent_mm: {tight.width * 25.4:.6f} x {tight.height * 25.4:.6f}",
        f"agg_tight_extent_aspect: {tight.width / tight.height:.8f}",
        f"column_gutters_at_{fig.dpi:g}_dpi_px: {','.join(f'{gap:.4f}' for gap in gutters)}",
        "analysis_execution: NONE (drawing and deterministic layout/serialization checks only)",
        "render_scope: editable figure from bundled tables",
    ]


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent
    patient_data = pd.read_csv(StringIO(PAIRED_PATIENT_TSV), sep="\t", float_precision="round_trip")
    mean_data = pd.read_csv(StringIO(PAIRED_COHORT_TSV), sep="\t", float_precision="round_trip")
    plot_results(patient_data, mean_data, destination)
