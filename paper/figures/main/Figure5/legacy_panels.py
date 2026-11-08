from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation, Text
from matplotlib.ticker import FormatStrFormatter
import numpy as np


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.titlesize"] = 8.6
plt.rcParams["axes.titleweight"] = "regular"
plt.rcParams["axes.labelsize"] = 7.2
plt.rcParams["xtick.labelsize"] = 7.0
plt.rcParams["ytick.labelsize"] = 7.0
plt.rcParams["legend.fontsize"] = 7.0


# === DATA SECTOR (edit here) ===
# Complete selected current F4a/F4b/F4d source records, 7 September 2026.
TLS_PATIENTS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H"
]

TLS_METHODS = [
    "DeepPT",
    "DeepSpaCE",
    "EGNv1",
    "EGNv2",
    "GeneCodeR",
    "HisToGene",
    "Hist2ST",
    "ST-Net",
    "TCGN",
    "THItoGene",
    "iStar"
]

TLS_STATE_MEASURED = [
    0.29853151823004065,
    0.24815441007160505,
    0.37485714988228647,
    0.4708270206341013,
    0.12522892566779906,
    0.2440019103117454,
    0.3305802315679014,
    0.40575775697904176
]

TLS_HOTSPOT_MEASURED = [
    0.6139228295819936,
    0.6373542458808619,
    0.7506944444444444,
    0.7890557184750733,
    0.5950934129429892,
    0.6343926846100759,
    0.6420820668693009,
    0.7002078332650312
]

TLS_STATE_MODELS = {
    "DeepPT": [
        0.06478008052693518,
        0.0022603556952146974,
        0.0763290340555202,
        0.15768257304802144,
        0.004885652288360893,
        0.05240285761891265,
        0.13480218377640657,
        0.21982993290525554
    ],
    "DeepSpaCE": [
        0.06678751818211706,
        0.08151883505238071,
        0.1919743630445758,
        0.19295878203608102,
        0.0351117909663766,
        -0.0015840348397209883,
        0.0332245764856948,
        0.1434237893765097
    ],
    "EGNv1": [
        0.012537303531018163,
        0.027049025883526118,
        0.1740208219944038,
        0.13275326834147425,
        0.020695818850217713,
        -0.0145658382743032,
        0.027097340461324983,
        0.13411980877168264
    ],
    "EGNv2": [
        0.0005588431270561859,
        0.03379659166831447,
        0.27370972655099474,
        0.048286929626193895,
        0.08064006591579001,
        0.02904405927076443,
        0.15061648220682347,
        0.2734386011637957
    ],
    "GeneCodeR": [
        0.07636904159019203,
        -0.016561144598661162,
        0.020749337494673458,
        -0.0783231180878212,
        -0.008626222204470348,
        -0.03299201969165666,
        0.048630250444864985,
        0.004045419524166852
    ],
    "HisToGene": [
        0.05541175568906388,
        -0.07528828709907313,
        0.009730699366903376,
        -0.0822078851070776,
        0.027076814779756903,
        0.021406175397693315,
        0.08609072974171939,
        0.0031404759292095142
    ],
    "Hist2ST": [
        0.09575706835200096,
        -0.050454412408730755,
        -0.10360743634638148,
        -0.06798223853229456,
        -0.03602408208933228,
        0.004318874510046479,
        0.09448016812334367,
        0.047398648059863185
    ],
    "ST-Net": [
        -0.05257077469505318,
        0.0873272397495202,
        0.16561493666085308,
        0.07616194454925526,
        0.025693371805814665,
        -0.030290076959566932,
        0.08036227657454928,
        0.19373614406246117
    ],
    "TCGN": [
        -0.02921300770134063,
        0.03319043239291419,
        0.0427630801856389,
        0.07741247146595884,
        -0.024052890938651945,
        0.052415507436705075,
        0.12813300801730118,
        0.25079499896257695
    ],
    "THItoGene": [
        0.15155178577123332,
        -0.1893376346419732,
        0.010686282516550811,
        0.02035577892208646,
        0.024910390566616775,
        -0.003508097996153274,
        -0.01031561321296531,
        -0.0070727819149252
    ],
    "iStar": [
        -0.03623999764090062,
        0.06046382239200608,
        -0.015181477167561266,
        0.07289747042041395,
        0.017580044286384936,
        0.05864274987738085,
        0.12608225118152655,
        0.08643512020430488
    ]
}

TLS_HOTSPOT_MODELS = {
    "DeepPT": [
        0.5453766651355075,
        0.5190811153358682,
        0.58184650997151,
        0.6438709677419354,
        0.4910310734463277,
        0.5386237635150678,
        0.5969528875379939,
        0.6066140741174404
    ],
    "DeepSpaCE": [
        0.5136793752870923,
        0.5784759188846641,
        0.6713497150997151,
        0.6561466275659824,
        0.49622817154596816,
        0.5095968484011962,
        0.5737196048632219,
        0.6018185410690241
    ],
    "EGNv1": [
        0.5318350941662839,
        0.5547623574144487,
        0.6197204415954416,
        0.5907302052785923,
        0.5185501733436055,
        0.4939722797331493,
        0.5105610435663628,
        0.592189420994087
    ],
    "EGNv2": [
        0.5133899862195682,
        0.5304277566539924,
        0.6529113247863247,
        0.5011730205278593,
        0.5506018875192604,
        0.5460668276972624,
        0.5772150455927052,
        0.6394575844505591
    ],
    "GeneCodeR": [
        0.5196302250803859,
        0.5080291508238276,
        0.5000623219373219,
        0.4501935483870968,
        0.5101157229070366,
        0.5062140556705774,
        0.5207218844984802,
        0.521919091388092
    ],
    "HisToGene": [
        0.5189710610932476,
        0.4963846641318124,
        0.46403133903133903,
        0.43805571847507335,
        0.5180221173600411,
        0.5239423740510697,
        0.5610258358662614,
        0.503781979977753
    ],
    "Hist2ST": [
        0.49223702342673403,
        0.4674397972116603,
        0.40121082621082615,
        0.46794134897360706,
        0.446003466872111,
        0.4754474350126524,
        0.5516970618034448,
        0.5304687957379544
    ],
    "ST-Net": [
        0.5085806155259531,
        0.6118631178707223,
        0.6216791310541311,
        0.5754046920821114,
        0.4868242809450437,
        0.5070652173913044,
        0.5561309523809523,
        0.645310579005913
    ],
    "TCGN": [
        0.524152503445108,
        0.5148510773130546,
        0.5709045584045584,
        0.5949208211143695,
        0.48563093862352336,
        0.5309489302967564,
        0.5607738095238095,
        0.632674462853463
    ],
    "THItoGene": [
        0.505493798805696,
        0.33164131812420783,
        0.4412126068376068,
        0.4708768328445748,
        0.5406619157678481,
        0.4743041177823787,
        0.4678115501519756,
        0.5227775013172531
    ],
    "iStar": [
        0.505512172714745,
        0.5520880861850443,
        0.4715366809116809,
        0.5615366568914956,
        0.48996452876219826,
        0.515978835978836,
        0.5633915906788247,
        0.5711894795386687
    ]
}

CELL_POINTS = [
    [
        "patient_1",
        "DeepSpot2Cell",
        1.2407255736691232,
        0.13663346838306975
    ],
    [
        "patient_1",
        "sCellST",
        1.261716584536723,
        0.1135520659384534
    ],
    [
        "patient_2",
        "DeepSpot2Cell",
        0.8590386280758903,
        -0.4463301677106774
    ],
    [
        "patient_2",
        "sCellST",
        0.8690361300267433,
        0.14740263792753738
    ],
    [
        "patient_3",
        "DeepSpot2Cell",
        0.4058179161937105,
        -0.043560874300430596
    ],
    [
        "patient_3",
        "sCellST",
        0.4134713406182339,
        0.0302416497129342
    ],
    [
        "patient_4",
        "DeepSpot2Cell",
        0.9803098361752178,
        -0.1366215218594238
    ],
    [
        "patient_4",
        "sCellST",
        0.9957888602721299,
        0.2142507052129797
    ],
    [
        "patient_5",
        "DeepSpot2Cell",
        1.3660470166591503,
        0.3242776887068668
    ],
    [
        "patient_5",
        "sCellST",
        1.4113078628056352,
        0.1068345706625457
    ]
]
# === END DATA SECTOR ===


OUTDIR = Path(__file__).resolve().parent
OUTPUT_PNG = OUTDIR / "img_iter0.png"
OUTPUT_PDF = OUTDIR / "figure_iter0.pdf"
OUTPUT_CHECK = OUTDIR / "floor_selfcheck_iter0.txt"
OUTPUT_DPI = 220

# Author-authorized v97 editorial reflow: one common-boundary task group.

# COL_WHITE: L1 large background field; clean reference is white.
COL_WHITE = "#ffffff"
# COL_AXES_BG: L1-perceived warm-neutral field, kept very close to white.
COL_AXES_BG = "#ffffff"
# COL_ROW: explicit user override; remove decorative alternating tint.
COL_ROW = "#FFFFFF"
# Project-authoritative v68 print-register palette. These deeper pigments
# preserve the baseline's scientific color semantics.
COL_MEASURED = "#1F5A58"  # measured RNA / principal reference: deep petrol
COL_EGNV2 = "#365F78"  # spot-scale result / principal blue
COL_DEEPSPACE = "#9B5E4C"  # biological-task accent: terracotta
COL_SCELLST = "#68546E"  # sCellST: aubergine
COL_DEEPSPOT = "#9B5E4C"  # DeepSpot2Cell: terracotta
COL_NEUTRAL = "#85827D"  # ordinary/baseline result: warm graphite
COL_PATIENT = "#AAA39B"  # secondary patient marks: warm grey
COL_NEGATIVE = "#9C4D43"  # negative result: brick
COL_TEXT = "#252B2D"  # body text: warm near-black
COL_SPINE = "#454B4D"  # axes/spines: thin dark grey
COL_GRID = "#DCD7D1"  # visible, recessive warm grid
COL_ZERO = "#DCD7D1"  # reference guides share the grid register

TLS_PATIENT_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h"]
CELL_PATIENT_MARKERS = ["o", "s", "^", "D", "P"]
PATIENT_EQUAL_MEAN_MARKER = "D"
CELL_COLORS = {"sCellST": COL_SCELLST, "DeepSpot2Cell": COL_DEEPSPOT}


def mean_ci95(values):
    """Patient-equal mean and two-sided 95% t interval for n=8."""
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    half = 2.364624 * float(a.std(ddof=1)) / np.sqrt(len(a))
    return mean, mean - half, mean + half


def style_axis(ax, all_four=False, grid=True):
    ax.set_facecolor(COL_AXES_BG)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis="both", color=COL_GRID, linewidth=0.45, alpha=0.90, zorder=0.4)
    for side in ["left", "bottom", "right", "top"]:
        visible = all_four or side in ["left", "bottom"]
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color(COL_SPINE)
        ax.spines[side].set_linewidth(0.65)
        ax.spines[side].set_zorder(10)
    ax.tick_params(
        axis="both", which="major", direction="out", length=2.2, width=0.55,
        color=COL_SPINE, labelcolor=COL_TEXT, pad=2.5,
    )
    for title in [ax.title, ax._left_title, ax._right_title]:
        title.set_color(COL_TEXT)
    ax.xaxis.label.set_color(COL_TEXT)
    ax.yaxis.label.set_color(COL_TEXT)


def raise_text_zorders(ax):
    for text in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        text.set_zorder(25)
    for title in [ax.title, ax._left_title, ax._right_title]:
        title.set_zorder(25)
    ax.xaxis.label.set_zorder(25)
    ax.yaxis.label.set_zorder(25)
    for text in ax.texts:
        text.set_zorder(25)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_zorder(30)
        for text in leg.get_texts() + [leg.get_title()]:
            text.set_zorder(31)


def draw_tls_field(ax, measured, model_values, highlight, xlim, xticks, xlabel, reference=None):
    labels = ["Measured RNA"] + TLS_METHODS
    rows = [measured] + [model_values[m] for m in TLS_METHODS]
    y_positions = np.arange(len(labels))[::-1]
    patient_offsets = np.linspace(-0.16, 0.16, len(TLS_PATIENTS))

    for row_i, (label, values, y) in enumerate(zip(labels, rows, y_positions)):
        if row_i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=COL_ROW, alpha=0.50, zorder=0.1)
        if label == "Measured RNA":
            colour = COL_MEASURED
        elif label == "EGNv2":
            colour = COL_EGNV2
        elif label == "DeepSpaCE":
            colour = COL_DEEPSPACE
        else:
            colour = COL_NEUTRAL
        point_colour = colour if label in {"Measured RNA", "EGNv2", "DeepSpaCE"} else COL_PATIENT

        mean, lo, hi = mean_ci95(values)
        ax.add_patch(Rectangle(
            (lo, y - 0.205), hi - lo, 0.41,
            facecolor=to_rgba(colour, 0.20), edgecolor=colour,
            linewidth=0.55, zorder=2,
        ))
        ax.plot([lo, hi], [y, y], color=colour, linewidth=0.72, zorder=3)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=colour, linewidth=0.62, zorder=3)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=colour, linewidth=0.62, zorder=3)

        for patient_i, value in enumerate(values):
            ax.scatter(
                [value], [y + patient_offsets[patient_i]], s=10.5,
                marker=TLS_PATIENT_MARKERS[patient_i], facecolor=point_colour,
                edgecolor=COL_WHITE, linewidth=0.35, alpha=1.0, zorder=5,
            )
        ax.plot(
            [mean], [y], marker=PATIENT_EQUAL_MEAN_MARKER, markersize=3.4,
            markerfacecolor=COL_AXES_BG, markeredgecolor=COL_TEXT,
            markeredgewidth=0.72, linestyle="none", zorder=6,
        )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel, labelpad=3.4)
    style_axis(ax, all_four=True, grid=True)
    if reference is not None:
        ax.axvline(
            reference, color=COL_ZERO, linewidth=0.65,
            linestyle=(0, (2.3, 2.3)), zorder=1,
        )

    for text, label in zip(ax.get_yticklabels(), labels):
        if label == "Measured RNA":
            text.set_color(COL_MEASURED)
            text.set_fontweight("regular")
        elif label == "EGNv2":
            text.set_color(COL_EGNV2)
            text.set_fontweight("regular")
        elif label == "DeepSpaCE":
            text.set_color(COL_DEEPSPACE)
            text.set_fontweight("regular")
        else:
            text.set_color(COL_TEXT)
            text.set_fontweight("regular")
    raise_text_zorders(ax)


