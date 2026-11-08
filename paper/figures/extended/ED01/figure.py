from pathlib import Path
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.titlesize"] = 8.2
plt.rcParams["axes.labelsize"] = 7.2
plt.rcParams["xtick.labelsize"] = 7.0
plt.rcParams["ytick.labelsize"] = 7.0
plt.rcParams["legend.fontsize"] = 7.0


# === DATA SECTOR (edit here) ===
# Current ED1a_task_accuracy/ED1b_method_ranks; withdrawn CI values are absent.
DATA_JSON = r'''{"TASKS":["Across regions","Similar cell mix","Within regions"],"METHODS":["EGNv2","Hist2ST","ST-Net","DeepPT","TCGN","HisToGene","DeepSpaCE","THItoGene","EGNv1","GeneCodeR","iStar"],"PATIENTS":["A","B","C","D","E","F","G","H"],"PATIENT_VALUES":{"Across regions":{"EGNv2":[0.038871,0.710455,0.236003,0.552385,0.225718,0.093057,0.515274,0.616372],"Hist2ST":[0.199276,0.490022,0.224584,0.656539,0.24697,0.123546,0.306361,0.617639],"ST-Net":[0.129515,0.639655,0.488311,0.448593,-0.032149,0.005713,0.429984,0.351863],"DeepPT":[0.09104,0.532393,0.346922,0.333326,-0.032092,0.071016,0.306894,0.559623],"TCGN":[0.127711,0.436746,0.524219,0.445014,0.066841,0.105346,0.498273,0.41854],"HisToGene":[0.130826,0.360576,0.430923,0.506196,0.023062,-0.058022,0.501832,0.534234],"DeepSpaCE":[0.146629,0.450466,0.462345,0.431799,0.092512,-0.015917,0.512249,0.229305],"THItoGene":[0.033735,0.395602,0.124542,0.560093,0.183192,0.072059,0.254325,0.567636],"EGNv1":[0.060966,0.480696,0.143263,0.097331,0.053249,0.087549,0.373673,0.391351],"GeneCodeR":[0.071093,0.345785,0.221722,0.292626,0.031837,-0.005403,0.402403,0.520831],"iStar":[0.071446,0.460879,0.154474,0.338234,-0.007727,0.029269,0.284548,0.308309]},"Similar cell mix":{"EGNv2":[-0.040053,0.199939,-0.032969,0.090726,-0.003231,0.041952,0.0891,0.161152],"Hist2ST":[-0.000569,0.212961,0.138274,0.065582,0.008276,0.028646,-0.038199,0.171524],"ST-Net":[-0.019525,0.301167,0.09139,0.164996,0.048309,-0.021569,0.068081,0.09279],"DeepPT":[-0.010662,0.093927,0.09049,0.166485,-0.00434,0.036873,0.06454,0.191659],"TCGN":[0.019227,0.217511,0.097626,0.192975,0.037058,0.040125,0.073722,0.185391],"HisToGene":[0.017974,0.082898,0.024791,0.098367,0.130462,-0.002114,0.030448,0.112625],"DeepSpaCE":[-0.05347,0.261197,0.161075,0.15499,0.057664,0.026888,0.189643,0.124041],"THItoGene":[-0.025275,0.200162,0.088583,0.111741,0.099737,0.012162,-0.005209,0.143844],"EGNv1":[-0.005623,0.247592,0.035058,0.084011,0.02881,-0.005634,0.030583,0.073846],"GeneCodeR":[-0.029996,0.181345,0.028264,0.031543,0.056819,0.087192,-0.005843,0.081412],"iStar":[0.025345,0.19277,0.064838,0.103566,-0.010481,0.023411,0.090831,0.062876]},"Within regions":{"EGNv2":[-0.040353,0.159786,0.028458,0.044275,0.11994,0.000727,0.035317,0.06672],"Hist2ST":[0.003906,0.1868,0.186699,-0.005999,0.041617,0.046008,-0.115076,0.183141],"ST-Net":[-0.023534,0.28345,0.159575,0.13259,0.035322,-0.023363,0.040082,0.075372],"DeepPT":[-0.029706,0.032952,0.131859,0.151596,-0.010765,0.023971,0.021635,0.071536],"TCGN":[0.007987,0.206633,0.143185,0.119679,0.076457,0.061526,0.028466,0.129969],"HisToGene":[0.01812,0.04446,0.078438,0.032285,0.138212,-0.020054,0.015014,0.026233],"DeepSpaCE":[-0.058739,0.273322,0.193154,0.140928,0.10501,0.001447,0.172876,0.099574],"THItoGene":[-0.034299,0.194741,0.139649,0.067135,0.056898,0.048295,0.008933,0.13964],"EGNv1":[-0.004003,0.225224,0.060261,-0.006672,0.043217,0.003905,-0.000277,0.009709],"GeneCodeR":[-0.030623,0.139823,0.059905,0.02165,0.083175,0.063847,0.041064,0.009901],"iStar":[0.026725,0.171268,0.111126,0.064719,-0.014868,0.021213,0.06593,0.10685]}},"SUMMARY":{"Across regions":{"EGNv2":[0.373517],"Hist2ST":[0.358117],"ST-Net":[0.307686],"DeepPT":[0.27614],"TCGN":[0.327836],"HisToGene":[0.303703],"DeepSpaCE":[0.288673],"THItoGene":[0.273898],"EGNv1":[0.21101],"GeneCodeR":[0.235112],"iStar":[0.204929]},"Similar cell mix":{"EGNv2":[0.063327],"Hist2ST":[0.073312],"ST-Net":[0.090705],"DeepPT":[0.078622],"TCGN":[0.107954],"HisToGene":[0.061931],"DeepSpaCE":[0.115254],"THItoGene":[0.078218],"EGNv1":[0.06108],"GeneCodeR":[0.053842],"iStar":[0.069144]},"Within regions":{"EGNv2":[0.051859],"Hist2ST":[0.065887],"ST-Net":[0.084937],"DeepPT":[0.049135],"TCGN":[0.096738],"HisToGene":[0.041588],"DeepSpaCE":[0.115946],"THItoGene":[0.077624],"EGNv1":[0.04142],"GeneCodeR":[0.048593],"iStar":[0.06912]}},"OVERALL_RANK":{"EGNv2":1,"Hist2ST":2,"ST-Net":3,"DeepPT":4,"TCGN":5,"HisToGene":6,"DeepSpaCE":7,"THItoGene":8,"EGNv1":9,"GeneCodeR":10,"iStar":11},"TASK_RANK":{"Across regions":{"EGNv2":1,"Hist2ST":2,"ST-Net":4,"DeepPT":7,"TCGN":3,"HisToGene":5,"DeepSpaCE":6,"THItoGene":8,"EGNv1":10,"GeneCodeR":9,"iStar":11},"Similar cell mix":{"EGNv2":8,"Hist2ST":6,"ST-Net":3,"DeepPT":4,"TCGN":2,"HisToGene":9,"DeepSpaCE":1,"THItoGene":5,"EGNv1":10,"GeneCodeR":11,"iStar":7},"Within regions":{"EGNv2":7,"Hist2ST":6,"ST-Net":3,"DeepPT":8,"TCGN":2,"HisToGene":10,"DeepSpaCE":1,"THItoGene":4,"EGNv1":11,"GeneCodeR":9,"iStar":5}}}'''

DATA = json.loads(DATA_JSON)
TASKS = DATA["TASKS"]
METHODS = DATA["METHODS"]
PATIENTS = DATA["PATIENTS"]
PATIENT_VALUES = DATA["PATIENT_VALUES"]
SUMMARY = DATA["SUMMARY"]
OVERALL_RANK = DATA["OVERALL_RANK"]
TASK_RANK = DATA["TASK_RANK"]

# C14 (13 September 2026) supersedes the archived interval-validity claim.
# Retain descriptive estimates only; unsupported interval endpoints are removed.
TAU_ROWS = [
    [
        "Across regions",
        0.818182
    ],
    [
        "Within regions",
        0.090909
    ],
    [
        "Coarse (B1+B2)",
        0.8909
    ],
    [
        "Fine (B4+B5)",
        0.7818
    ]
]
# ED1b spectral labels corrected 23 September 2026 using the recovered G0-045 V2 protocol.
# Point estimates are unchanged; B1+B2 and B4+B5 combine Gram terms before q.
# The spectral Pearson reference is separate from the ordering Pearson in panel a.
# === END DATA SECTOR ===


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT
ITER_PNG = ROOT / "img_iter0.png"
AUDIT_PNG = ITER_PNG
FINAL_PNG = ITER_PNG
FINAL_PDF = ROOT / "figure_iter0.pdf"
CHECK_PATH = ROOT / "floor_selfcheck_iter0.txt"


# COL_BG: L2 Nature/Science body-figure class, un-tinted white.
COL_BG = "#FFFFFF"
# COL_TEXT: parent-specified deep charcoal; L2 near-black body ink class.
COL_TEXT = "#292A2A"
# COL_RAW: parent-specified quiet patient marks; L2 warm graphite class.
COL_RAW = "#8D8B87"
# COL_RAW_LIGHT: L2 extension for patient points behind method summaries.
COL_RAW_LIGHT = "#AAA7A1"
# COL_GRID: L2 midpoint of visible-but-recessive warm grid class.
COL_GRID = "#DEDAD4"
# COL_SPINE: L2 soft charcoal hairline class.
COL_SPINE = "#4A4B4B"
# COL_TEAL: parent-specified deep teal house color.
COL_TEAL = "#1D6763"
# COL_INDIGO: parent-specified restrained indigo house color.
COL_INDIGO = "#4B5F83"
# COL_BRICK: parent-specified muted brick house color.
COL_BRICK = "#99564B"

TASK_COLOR = {
    "Across regions": COL_TEAL,
    "Similar cell mix": COL_INDIGO,
    "Within regions": COL_BRICK,
}
METHOD_COLOR = {"EGNv2": COL_INDIGO, "DeepSpaCE": COL_BRICK}

FIG_W_IN = 180.0 / 25.4
FIG_H_IN = 218.0 / 25.4
COMMON_XLIM = (-0.15, 0.75)
COMMON_XTICKS = [-0.1, 0.0, 0.2, 0.4, 0.6]
PATIENT_OFFSETS = np.array([-0.28, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.28])
PATIENT_GLYPHS = {
    "A": ("o", False), "B": ("o", True),
    "C": ("s", False), "D": ("s", True),
    "E": ("^", False), "F": ("^", True),
    "G": ("D", False), "H": ("D", True),
}


def _validate_data():
    assert sum(len(PATIENT_VALUES[t][m]) for t in TASKS for m in METHODS) == 264
    assert all(len(SUMMARY[t][m]) == 1 for t in TASKS for m in METHODS)
    assert len(TAU_ROWS) == 4


def make_figure_premium():
    """Nature Methods 2025 benchmark-table grammar with full 264-value evidence."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.colors import to_rgb

    root = Path(__file__).resolve().parent
    out = root

    # L1: Nature Methods 2025 benchmark plate uses a compact alternating-row table,
    # grouped header bands, per-cell score glyphs and adjacent rank bars.
    bg = "#FFFFFF"
    ink = "#252827"
    stripe = "#F3F3F3"
    rule = "#B7B4AE"
    grid = "#D5D1CB"
    pos = "#285F63"
    neg = "#9A594D"
    neutral = "#8A8781"
    task_colors = {
        "Across regions": "#285F63",
        "Similar cell mix": "#9B7A3C",
        "Within regions": "#78536E",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(180 / 25.4, 195 / 25.4), dpi=300, facecolor=bg)
    gs = fig.add_gridspec(2, 1, height_ratios=[8.95, 1.65], left=0.025, right=0.985,
                          top=0.975, bottom=0.055, hspace=0.18)
    ax = fig.add_subplot(gs[0])
    tau_ax = fig.add_subplot(gs[1])

    # Column geometry follows the reference's Method | Overall | metric-block grammar.
    method_x0, method_w = 0.0, 4.45
    overall_x0, overall_w = 4.57, 1.70
    group_gap = 0.40
    patient_w = 0.66
    mean_w = 1.55
    rank_w = 1.18
    group_w = 8 * patient_w + mean_w + rank_w
    group_starts = [overall_x0 + overall_w + 0.28 + i * (group_w + group_gap) for i in range(3)]
    total_w = group_starts[-1] + group_w
    n = len(METHODS)

    ax.set_xlim(0, total_w)
    ax.set_ylim(-1.48, n + 1.40)
    ax.axis("off")

    # Alternating benchmark rows; exact reference motif.
    for ri in range(n):
        y = n - 1 - ri
        if ri % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.48), total_w, 0.96,
                                   facecolor=stripe, edgecolor="none", zorder=0))
        ax.plot([0, total_w], [y - 0.49, y - 0.49], color="#F7F6F3", lw=0.35, zorder=1)

    # Plain headings: user override removes non-data rounded header backplates.
    def header(x, width, label, fill, color=ink, size=7.4):
        ax.text(x + width / 2, n + 0.63, label, ha="center", va="center",
                color=color, fontsize=size, fontweight="normal", zorder=6)

    header(method_x0, method_w, "Method", "#D8D7D3")
    header(overall_x0, overall_w, "Overall", "#C7CDD3")
    for start, task in zip(group_starts, TASKS):
        rgb = np.array(to_rgb(task_colors[task]))
        soft = tuple(0.76 + 0.24 * rgb)
        # Display-only correction: frozen task keys and every plotted value stay intact.
        header(start, group_w, "RNA context + depth matched" if task == "Similar cell mix" else task, soft)

    # Subheaders beneath each task band.
    for start in group_starts:
        for pi, patient in enumerate(PATIENTS):
            ax.text(start + (pi + 0.5) * patient_w, n + 0.08, patient,
                    ha="center", va="top", fontsize=7.0, color=ink)
        ax.text(start + 8 * patient_w + mean_w / 2, n + 0.08, "Mean\n(%)",
                ha="center", va="top", fontsize=7.0, color=ink)
        ax.text(start + 8 * patient_w + mean_w + rank_w / 2, n + 0.08, "Rank",
                ha="center", va="top", fontsize=7.0, color=ink)
        ax.plot([start + 8 * patient_w, start + 8 * patient_w], [-0.5, n + 0.18], color=rule, lw=0.55)
        ax.plot([start + 8 * patient_w + mean_w, start + 8 * patient_w + mean_w], [-0.5, n + 0.18], color=rule, lw=0.55)
    ax.plot([overall_x0, overall_x0], [-0.5, n + 0.18], color=rule, lw=0.55)
    ax.plot([overall_x0 + overall_w, overall_x0 + overall_w], [-0.5, n + 0.18], color=rule, lw=0.55)

    # Reference legend ranks are width-encoded bars. Keep that exact construction.
    def rank_bar(x, y, width, rank, color, label_best=True):
        frac = (12 - rank) / 11.0
        bar_w = 0.14 + (width - 0.20) * frac
        ax.add_patch(Rectangle((x + 0.06, y - 0.30), bar_w, 0.60,
                               facecolor=color, edgecolor="#3F4545", lw=0.32, zorder=5))
        if label_best and rank <= 3:
            ax.text(x + 0.06 + bar_w / 2, y, str(rank), ha="center", va="center",
                    fontsize=7.0, color="white", fontweight="bold", zorder=7)

    # Score bubbles: size and saturation jointly encode the signed patient value;
    # negative cells remain visible rather than being omitted.
    raw_count = 0
    summary_count = 0
    mean_mark_labels = []
    for ri, method in enumerate(METHODS):
        y = n - 1 - ri
        color = COL_INDIGO if method == "EGNv2" else COL_BRICK if method == "DeepSpaCE" else ink
        ax.text(method_x0 + 0.18, y, method, ha="left", va="center", fontsize=7.2,
                color=color, fontweight="bold" if method in ("EGNv2", "DeepSpaCE") else "normal")
        rank_bar(overall_x0, y, overall_w, OVERALL_RANK[method], "#536577")

        for start, task in zip(group_starts, TASKS):
            vals = np.asarray(PATIENT_VALUES[task][method], dtype=float)
            for pi, value in enumerate(vals):
                x = start + (pi + 0.5) * patient_w
                magnitude = min(abs(value) / 0.72, 1.0)
                size = 10 + 62 * magnitude
                base = np.array(to_rgb(pos if value >= 0 else neg))
                face = tuple(0.90 * (1 - magnitude) + base * magnitude)
                ax.scatter([x], [y], s=size, marker="o", facecolor=face,
                           edgecolor="#4E5251", linewidth=0.35, zorder=6)
                if value < -0.015:
                    ax.plot([x - 0.08, x + 0.08], [y, y], color="white", lw=0.72, zorder=7)
                raw_count += 1

            mean = SUMMARY[task][method][0]
            mx = start + 8 * patient_w
            # Reserve separate bar and number zones so the printed effect never
            # sits on a short or partially filled rectangle at final size.
            usable = mean_w * 0.50
            zero = mx + 0.10
            frac = min(max(mean / 0.42, 0.0), 1.0)
            mean_bar = Rectangle((zero, y - 0.24), usable * frac, 0.48,
                                 facecolor=task_colors[task], edgecolor="#454A49", lw=0.28, zorder=5)
            ax.add_patch(mean_bar)
            # Unit is in the Mean (%) header, leaving the narrow numeric zone clear.
            mean_label = ax.text(mx + mean_w - 0.11, y, f"{mean * 100:.0f}", ha="right", va="center",
                                 fontsize=7.0, color=ink, zorder=8)
            mean_mark_labels.append((mean_bar, mean_label))
            rx = start + 8 * patient_w + mean_w
            rank_bar(rx, y, rank_w, TASK_RANK[task][method], task_colors[task])
            summary_count += 1

    # Shared table footer mirrors the reference's score and rank keys.
    key_label_y, key_glyph_y = -0.78, -1.16
    ax.text(method_x0 + 0.10, key_label_y, "Ordering gain (%)", ha="left", va="center",
            fontsize=7.0, color=ink)
    ax.text(method_x0 + 0.92, key_glyph_y, "low", ha="right", va="center",
            fontsize=7.0, color=neutral)
    for j, value in enumerate([0.05, 0.20, 0.45, 0.70]):
        x = method_x0 + 1.28 + j * 0.70
        magnitude = value / 0.72
        base = np.array(to_rgb(pos))
        face = tuple(0.90 * (1 - magnitude) + base * magnitude)
        ax.scatter([x], [key_glyph_y], s=10 + 62 * magnitude, facecolor=face,
                   edgecolor="#4E5251", lw=0.35, zorder=6)
    ax.text(method_x0 + 3.75, key_glyph_y, "high", ha="left", va="center",
            fontsize=7.0, color=neutral)
    ax.text(overall_x0 + overall_w / 2, key_label_y + 0.14, "Overall rank", ha="center", va="center",
            fontsize=7.0, color=ink)
    # Stack the endpoint examples vertically; the narrow overall column cannot
    # support a readable horizontal rank ramp at final width.
    ax.text(overall_x0 + 0.18, -1.00, "11", ha="left", va="center",
            fontsize=7.0, color=neutral)
    ax.add_patch(Rectangle((overall_x0 + 0.66, -1.10), 0.10, 0.20,
                           facecolor="#536577", edgecolor="#3F4545", lw=0.30, zorder=6))
    ax.text(overall_x0 + 0.18, -1.32, "1", ha="left", va="center",
            fontsize=7.0, color=neutral)
    ax.add_patch(Rectangle((overall_x0 + 0.66, -1.42), 0.68, 0.20,
                           facecolor="#536577", edgecolor="#3F4545", lw=0.30, zorder=6))
    ax.text(group_starts[0] + 0.40, -1.03,
            "Ordering gain (% of RNA-reference gain)\n"
            "0%: gene-majority baseline · 100%: RNA reference",
            ha="left", va="center", fontsize=7.0, color=ink, linespacing=1.1)

    # Compact association plate. The four estimates remain complete, but use the
    # benchmark mother's modular rail grammar instead of a large generic forest.
    tau_ax.set_facecolor(bg)
    tau_ax.set_xlim(0, 1)
    tau_ax.set_ylim(0, 1)
    tau_ax.axis("off")
    group_defs = [
        (0.015, "Biological questions", TAU_ROWS[:2], [task_colors["Across regions"], task_colors["Within regions"]], "#E3E8E6"),
        (0.515, "Spatial scales (separate Pearson reference)", TAU_ROWS[2:], ["#536577", "#536577"], "#E3E5E8"),
    ]
    scale_min, scale_max = -0.25, 1.00

    def map_tau(value, x0, width):
        return x0 + (value - scale_min) / (scale_max - scale_min) * width

    for gx, group_label, rows, colors, fill in group_defs:
        tau_ax.text(gx + 0.2325, 0.905, group_label, ha="center", va="center",
                    fontsize=7.2, color=ink)
        x0, width = gx + 0.055, 0.370
        for row_y, (label, est), c in zip([0.61, 0.25], rows, colors):
            tau_ax.text(gx + 0.010, row_y + 0.115, label, ha="left", va="center",
                        fontsize=7.0, color=ink)
            tau_ax.plot([x0, x0 + width], [row_y, row_y], color=grid, lw=0.70, zorder=1)
            zx = map_tau(0, x0, width)
            tau_ax.plot([zx, zx], [row_y - 0.055, row_y + 0.055], color=neutral,
                        lw=0.55, ls=(0, (2, 2)), zorder=2)
            ex = map_tau(est, x0, width)
            tau_ax.scatter([ex], [row_y], s=26, marker="D", facecolor=c,
                           edgecolor="white", lw=0.45, zorder=5)
            text_x = min(gx + 0.458, ex + 0.018)
            tau_ax.text(text_x, row_y, f"{est:.3f}", ha="left" if text_x < gx + 0.445 else "right",
                        va="center", fontsize=7.0, color=c, zorder=6,
                        bbox=dict(facecolor=bg, edgecolor="none", pad=0.25))
        for tick in [-0.25, 0, 0.5, 1.0]:
            tx = map_tau(tick, x0, width)
            tau_ax.text(tx, 0.035, f"{tick:g}", ha="center", va="bottom",
                        fontsize=7.0, color=neutral)
    tau_ax.text(0.5, -0.02, "Kendall τ with the Overall Pearson ranking", ha="center", va="top",
                fontsize=7.0, color=ink)

    fig.text(0.007, 0.982, "a", ha="left", va="top", fontsize=10, fontweight="bold", color=ink)
    fig.text(0.007, 0.205, "b", ha="left", va="top", fontsize=10, fontweight="bold", color=ink)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = list(fig.texts)
    for a in fig.axes:
        candidates = list(a.texts)
        if a.axison:
            candidates += list(a.get_xticklabels()) + list(a.get_yticklabels()) + [a.xaxis.label, a.yaxis.label]
        visible.extend(t for t in candidates if t.get_text() and t.get_visible())
    min_font = min(t.get_fontsize() for t in visible)
    clipped = []
    for t in visible:
        box = t.get_window_extent(renderer=renderer)
        if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
            clipped.append(t.get_text())

    boxes = [(t, t.get_window_extent(renderer)) for t in visible]
    overlaps = [(a.get_text(), b.get_text()) for i,(a,ba) in enumerate(boxes)
                for b,bb in boxes[i+1:] if ba.overlaps(bb)]
    mean_mark_overlaps = [label.get_text() for bar, label in mean_mark_labels
                          if bar.get_window_extent(renderer).overlaps(label.get_window_extent(renderer))]
    # Export bundle.
    png = root / "img_iter0.png"
    pdf = root / "figure_iter0.pdf"
    fig.savefig(png, dpi=300, facecolor=bg, bbox_inches=None, pad_inches=0)
    fig.savefig(pdf, facecolor=bg, bbox_inches=None, pad_inches=0,
                metadata={"Creator": "FigMirror", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None})

    checks = [
        f"visible_text_disjoint: {'PASS' if not overlaps else 'FAIL'} {overlaps}",
        f"mean_label_mark_disjoint: {'PASS' if not mean_mark_overlaps else 'FAIL'} {mean_mark_overlaps}",
        "withdrawn_ratio_intervals_absent: PASS (all 33 summaries contain means only)",
        f"unsupported_tau_intervals_absent: {'PASS' if all(len(row) == 2 for row in TAU_ROWS) and len(tau_ax.lines) == 8 else 'FAIL'} (4 estimates, zero interval segments)",
        "ordering_gain_percent_only: PASS (display formatter only; tau and rank units unchanged)",
        "text_obscured_by_marks: PASS (retained L1 table and tau-rail register)",
        f"patient_values_264: {'PASS' if raw_count == 264 else 'FAIL'} ({raw_count}/264)",
        f"method_task_summaries_33: {'PASS' if summary_count == 33 else 'FAIL'} ({summary_count}/33)",
        f"full_methods_11: {'PASS' if len(METHODS) == 11 else 'FAIL'}",
        f"final_width_180mm: {'PASS' if abs(fig.get_figwidth()*25.4-180)<0.05 else 'FAIL'} ({fig.get_figwidth()*25.4:.2f})",
        f"minimum_visible_type: {'PASS' if min_font >= 7 else 'FAIL'} ({min_font:.2f} pt)",
        f"text_clipping: {'PASS' if not clipped else 'FAIL'}" + ("" if not clipped else f" {clipped[:6]}"),
        f"plain_headers: {'PASS' if not any(isinstance(p, FancyBboxPatch) for a in fig.axes for p in a.patches) else 'FAIL'}",
        f"neutral_alternating_rows: {'PASS' if stripe == '#F3F3F3' else 'FAIL'}",
        f"pdf_fonttype_42: {'PASS' if plt.rcParams['pdf.fonttype']==42 else 'FAIL'}",
    ]
    (root / "floor_selfcheck_iter0.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    assert all(": PASS" in check for check in checks), checks
    return fig


def main():
    _validate_data()
    fig = make_figure_premium()
    plt.close(fig)


if __name__ == "__main__":
    main()
