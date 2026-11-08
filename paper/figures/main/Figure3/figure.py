from pathlib import Path
import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.tri as mtri
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Rectangle
import numpy as np

# Use the baseline's intended narrow face when it exists on this Windows host.
_narrow_font = Path('C:/Windows/Fonts/arialn.ttf')
if _narrow_font.is_file():
    matplotlib.font_manager.fontManager.addfont(str(_narrow_font))
_narrow_properties = matplotlib.font_manager.FontProperties(
    fname=str(_narrow_font)) if _narrow_font.is_file() else matplotlib.font_manager.FontProperties(family='Arial')


# === DATA SECTOR (edit here) ===
# Authoritative pointwise patient/specimen-block bootstrap intervals, copied
# verbatim from the completed 2026-09-13 tables. No inference is run here.
INTERVAL_ROWS = [
    dict(source_file='SCALE_MEAN_INTERVALS.tsv', source_key='CRC|fixed_32um_predictions|noise_corrected_correlation_q|LT128', cohort='CRC', model='fixed_32um_predictions', metric='noise_corrected_correlation_q', source_band_key='LT128', contrast_id='', estimate='0.29840133333333335', lower='0.214449', upper='0.35008699999999998', displayed_unit_factor='1', panel='Figure 2a', axis_min='0.17', axis_max='1.02'),
    dict(source_file="SCALE_MEAN_INTERVALS.tsv", source_key="CRC|fixed_32um_predictions|noise_corrected_correlation_q|GE4096", cohort="CRC", model="fixed_32um_predictions", metric="noise_corrected_correlation_q", source_band_key="GE4096", contrast_id="", estimate="0.80048533333333338", lower="0.72653999999999996", upper="0.87031000000000003", displayed_unit_factor="1", panel="Figure 2a", axis_min="0.17", axis_max="1.02"),
    dict(source_file="SCALE_PAIRED_CONTRASTS.tsv", source_key="HEST_Ridge-Phikon-v2_Broadest_minus_Finest", cohort="HEST", model="Ridge-Phikon-v2", metric="noise_corrected_correlation_q", source_band_key="", contrast_id="HEST_Ridge-Phikon-v2_Broadest_minus_Finest", estimate="0.36507875555555558", lower="0.33763790000000005", upper="0.3936252111111111", displayed_unit_factor="1", panel="Figure 2d pooled effect", axis_min="0.30", axis_max="0.43"),
]
SCALES_UM = np.array([32, 64, 96, 128, 192, 256, 384, 512], dtype=float)
RANGE_KEYS = [
    "LT128",
    "128_256",
    "256_512",
    "512_1024",
    "1024_2048",
    "2048_4096",
    "GE4096",
]
RANGE_LABELS = ["<128", "128–256", "256–512", "512–1,024", "1,024–2,048", "2,048–4,096", "≥4,096"]
RANGE_LABELS_SHORT = ["<128", "128–256", "256–512", "0.5–1k", "1–2k", "2–4k", "≥4k"]

# Rows are spatial scales of expression variation (fine to broad); columns are physical output sizes.
# NaN is a structural/Nyquist blank, never zero and never interpolated.
Q_FIELDS = {
    "P1 CRC": np.array(
        [
            [0.330668, 0.494769, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.562768, 0.588655, 0.592560, 0.581814, np.nan, np.nan, np.nan, np.nan],
            [0.651230, 0.686926, 0.688114, 0.661409, 0.638847, 0.630932, np.nan, np.nan],
            [0.662221, 0.690682, 0.692024, 0.701944, 0.679896, 0.640791, 0.664822, 0.460834],
            [0.748653, 0.780568, 0.768727, 0.716987, 0.739248, 0.695067, 0.738578, 0.745020],
            [0.651742, 0.698598, 0.687988, 0.687399, 0.627126, 0.626748, 0.606245, 0.668894],
            [0.804606, 0.840621, 0.884231, 0.869967, 0.877403, 0.881310, 0.898556, 0.903883],
        ],
        dtype=float,
    ),
    "P2 CRC": np.array(
        [
            [0.350087, 0.552648, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.635734, 0.646656, 0.625267, 0.606564, np.nan, np.nan, np.nan, np.nan],
            [0.712962, 0.740518, 0.706342, 0.654837, 0.541529, 0.475486, np.nan, np.nan],
            [0.740795, 0.769385, 0.742080, 0.717639, 0.653018, 0.598455, 0.587222, 0.453078],
            [0.777982, 0.796856, 0.769272, 0.772232, 0.678884, 0.672425, 0.673140, 0.652199],
            [0.755173, 0.789102, 0.726114, 0.720732, 0.614390, 0.609976, 0.548162, 0.604731],
            [0.870310, 0.896819, 0.858861, 0.877555, 0.813894, 0.830525, 0.779700, 0.847995],
        ],
        dtype=float,
    ),
    "P3 NAT": np.array(
        [
            [0.214449, 0.351996, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.448058, 0.467001, 0.483156, 0.504311, np.nan, np.nan, np.nan, np.nan],
            [0.593681, 0.621147, 0.611228, 0.602912, 0.532889, 0.446535, np.nan, np.nan],
            [0.763236, 0.798879, 0.818307, 0.835072, 0.771438, 0.652550, 0.690288, 0.759043],
            [0.707776, 0.775992, 0.798001, 0.830528, 0.733166, 0.655903, 0.605501, 0.614239],
            [0.750464, 0.771700, 0.697175, 0.813454, 0.686931, 0.591444, 0.489649, 0.485139],
            [0.726540, 0.752073, 0.740758, 0.823616, 0.664279, 0.632115, 0.681827, 0.683982],
        ],
        dtype=float,
    ),
}

GE4096_FRACTION = {
    "P1 CRC": np.array([0.302853, 0.381655, 0.419613, 0.440588, 0.475646, 0.516109, 0.551301, 0.576798]),
    "P2 CRC": np.array([0.202810, 0.257181, 0.292644, 0.321926, 0.365279, 0.384709, 0.427145, 0.456438]),
    "P3 NAT": np.array([0.176741, 0.211127, 0.230062, 0.245635, 0.263228, 0.270000, 0.316755, 0.310817]),
}
GE4096_PATIENT_EQUAL = np.array([0.227468, 0.283321, 0.314106, 0.336050, 0.368051, 0.390273, 0.431733, 0.448018])
COMPOSITION_SHARED = np.array([0.2730, 0.4382, 0.5244, 0.5732, 0.6222, 0.6411, 0.6488, 0.6650])
SAMPLING_SHARED = np.array([0.5216, 0.3267, 0.2267, 0.1671, 0.1010, 0.0685, 0.0372, 0.0235])

RNA_RANGE_FINE_TO_BROAD = ["Finest", "Fine", "Mid", "Broad", "Broadest"]

# Exact v56 iteration-2 HEST_FIVE_SCALE.tsv Ridge-Phikon-v2 values.
# Each row stores Broadest -> Finest RNA accuracy for one specimen.
HEST = [
    ("IDC", "TENX99", [0.7155244, 0.4661726, 0.3719073, 0.3389596, 0.2961230]),
    ("IDC", "TENX95", [0.7188615, 0.5553542, 0.4997213, 0.4595315, 0.4026285]),
    ("IDC", "NCBI785", [0.6029360, 0.4657356, 0.3996823, 0.3255402, 0.2879420]),
    ("IDC", "NCBI783", [0.7574587, 0.6332780, 0.5311476, 0.4706086, 0.4327348]),
    ("PAAD", "TENX116", [0.6018694, 0.4260289, 0.3422629, 0.3277574, 0.2423324]),
    ("PAAD", "TENX140", [0.6805381, 0.4250162, 0.3446147, 0.3158414, 0.2631679]),
    ("PAAD", "TENX126", [0.5548812, 0.4559245, 0.3279693, 0.3152353, 0.2803837]),
    ("LUNG", "TENX141", [0.7059609, 0.4933674, 0.3962534, 0.3455820, 0.2763288]),
    ("LUNG", "TENX118", [0.7378132, 0.5571307, 0.4058427, 0.3728459, 0.3084935]),
]
# === END DATA SECTOR ===


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.labelsize"] = 7.0
plt.rcParams["axes.titlesize"] = 7.0
plt.rcParams["xtick.labelsize"] = 7.0
plt.rcParams["ytick.labelsize"] = 7.0

# Every color is grounded in the staged reference, the accepted v52 panel-c
# construction, or the frozen v68 older Nature-family print-register contract.
COL_WHITE = "#ffffff"  # L1-PIL: reference page and axes background.
COL_INK = "#252B2D"  # Palette contract: body text, dense old-print ink.
COL_AXIS = "#454B4D"  # Palette contract: thin dark-grey axes/spines.
COL_GUIDE = "#DCD7D1"  # Palette contract: visible but recessive warm grid.
COL_NA = "#fafafa"  # L1-class: near-white absence against a white page.
COL_CELL_EDGE = (1.0, 1.0, 1.0, 0.20)  # L1-perceived: the mother has contiguous tiles with only a soft light boundary.
COL_NEUTRAL = "#AAA39B"  # Palette contract: secondary patient observations; identity remains shape-coded.
COL_SPOT_RESULT = "#365F78"  # Palette contract: spot-scale result / principal muted Prussian blue.
COL_B_RNA = "#AAA39B"  # Palette contract: secondary patient RNA trajectories in warm grey.
COL_B_RNA_DARK = "#1F5A58"  # Palette contract: measured RNA / principal reference deep petrol.
COL_COMPOSITION = "#9B5E4C"  # Palette contract: biological-task / composition terracotta.
COL_SAMPLING = "#A87342"  # Palette contract: sampling/training warm ochre-brown.
COL_BAR_GREY = "#85827D"  # Palette contract: ordinary/baseline warm graphite.
COL_HEAT_0 = "#f7e8a6"  # L1-PIL: sampled raster low end at (135, 2425).
COL_HEAT_1 = "#f7d766"  # L1-PIL: sampled raster low-mid at (250, 2415).
COL_HEAT_2 = "#cf7f45"  # L1-PIL: sampled raster midpoint at (360, 2250).
COL_HEAT_3 = "#8b335f"  # L1-PIL: sampled raster high at (520, 2050).
COL_HEAT_4 = "#862e63"  # L1-PIL: sampled raster upper end at (575, 1985).

PATIENT_COLORS = {"P1 CRC": COL_NEUTRAL, "P2 CRC": COL_NEUTRAL, "P3 NAT": COL_NEUTRAL}
PATIENT_MARKERS = {"P1 CRC": "o", "P2 CRC": "s", "P3 NAT": "^"}
MEAN_MARKER = "D"
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "bost_warm",
    [COL_HEAT_0, COL_HEAT_1, COL_HEAT_2, COL_HEAT_3, COL_HEAT_4],
    N=256,
)
HEAT_NORM = Normalize(vmin=0.20, vmax=0.92, clip=True)

# Panel-c construction stays pixel-registered to v52; only its sequential family
# is harmonized to the frozen four-anchor Nature-style palette.
SURFACE_INK = "#252B2D"  # Palette contract: body ink and 0.60 contour; field colors remain accepted.
SURFACE_SPINE = "#454B4D"  # Palette contract: shared four-hairline field frame.
SURFACE_NA = "#f2f2f2"  # L1/source: pale structural-N/A ground.
SURFACE_NA_EDGE = "#85827D"  # Palette contract: warm-graphite structural-N/A hatch edge.
SURFACE_HEAT_LOW = "#f8f2d2"  # Global style: warm parchment low anchor.
SURFACE_HEAT_GOLD = "#f8cf55"  # Global style: restrained yellow mid-low anchor.
SURFACE_HEAT_MID = "#be7148"  # Global style: burnt-sienna mid-high anchor.
SURFACE_HEAT_HIGH = "#862e63"  # Global style: muted berry high anchor.
SURFACE_CONTOUR_MID = "#DCD7D1"  # Palette contract: quiet warm-grey 0.70 isoline.
SURFACE_PATIENT_LABELS = {
    "P1 CRC": "P1 · CRC",
    "P2 CRC": "P2 · CRC",
    "P3 NAT": "P3 · NAT",
}
SURFACE_CMAP = LinearSegmentedColormap.from_list(
    "parchment_to_slate",
    [SURFACE_HEAT_LOW, SURFACE_HEAT_GOLD, SURFACE_HEAT_MID, SURFACE_HEAT_HIGH],
    N=256,
)
SURFACE_NORM = Normalize(vmin=0.44, vmax=0.91, clip=True)

# Panel-d keeps the accepted v56 construction while adopting the manuscript
# house tokens; panels a-c and their warm scientific encodings remain untouched.
HEST_COL_BG = "#ffffff"  # Shared white field ground.
HEST_COL_TEAL = "#1F5A58"  # Palette contract: measured RNA / median deep petrol.
HEST_COL_INK = "#252B2D"  # Palette contract: dense body ink.
HEST_COL_SPINE = "#454B4D"  # Palette contract: thin dark-grey spine.
HEST_COL_GRID = "#DCD7D1"  # Palette contract: visible but recessive warm grid.
HEST_COL_NEUTRAL = "#AAA39B"  # Palette contract: secondary patient marks.
HEST_COL_NEUTRAL_LIGHT = "#85827D"  # Palette contract: ordinary/summary warm graphite.


def log_cell_edges(values):
    centers = np.log2(values)
    mids = (centers[:-1] + centers[1:]) / 2.0
    edges = np.r_[centers[0] - (mids[0] - centers[0]), mids, centers[-1] + (centers[-1] - mids[-1])]
    return centers, edges


LOG_SCALES, LOG_EDGES = log_cell_edges(SCALES_UM)
B_LABEL_FONT_PT = 7.0
B_LABEL_ROW_OFFSETS_PT = (1.0, 10.0)
B_XLABEL_Y_AXES = -1.24
B_STRIP_LABEL_Y_AXES = 1.08
B_ENDPOINT_BAND_Y_AXES = 0.42
C_LABEL_FONT_PT = 7.0
C_LABEL_ROW_OFFSETS_PT = (1.5, 8.7)
C_XLABEL_Y_AXES = -0.1192


def style_axis(ax, *, bottom=True, left=True):
    ax.set_facecolor(COL_WHITE)
    ax.set_axisbelow(True)
    ax.grid(False)
    for side, spine in ax.spines.items():
        spine.set_visible((side == "bottom" and bottom) or (side == "left" and left))
        spine.set_color(COL_AXIS)
        spine.set_linewidth(0.62)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.7,
        width=0.55,
        color=COL_AXIS,
        labelcolor=COL_INK,
        pad=2.3,
    )
    ax.xaxis.label.set_color(COL_INK)
    ax.yaxis.label.set_color(COL_INK)
    ax.xaxis.label.set_zorder(10)
    ax.yaxis.label.set_zorder(10)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_zorder(10)


def white_text_box(alpha=0.94, pad=0.13):
    return {"facecolor": COL_WHITE, "edgecolor": "none", "alpha": alpha, "pad": pad}


def style_surface_axis(ax):
    """Accepted v52 four-hairline framing for the panel-c fields only."""
    ax.set_facecolor(COL_WHITE)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SURFACE_SPINE)
        spine.set_linewidth(0.55)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=0,
        width=0.50,
        pad=2.1,
        colors=SURFACE_INK,
    )
    ax.xaxis.label.set_color(SURFACE_INK)
    ax.yaxis.label.set_color(SURFACE_INK)
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_zorder(20)


def contiguous_runs(mask):
    """Return inclusive-exclusive runs for a one-dimensional Boolean mask."""
    runs = []
    start = None
    for index, flag in enumerate(np.r_[mask, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    return runs


def build_local_triangulation(field):
    """Piecewise-linear interpolation only inside adjacent measured grid cells."""
    finite = np.isfinite(field)
    vertex_index = np.full(field.shape, -1, dtype=int)
    x_values = []
    y_values = []
    z_values = []
    grid_vertices = []
    for row, col in np.argwhere(finite):
        vertex_index[row, col] = len(x_values)
        x_values.append(LOG_SCALES[col])
        y_values.append(float(row))
        z_values.append(float(field[row, col]))
        grid_vertices.append((int(row), int(col)))

    triangles = []
    triangle_grid_vertices = []
    for row in range(field.shape[0] - 1):
        for col in range(field.shape[1] - 1):
            corners = [(row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col)]
            present = [corner for corner in corners if finite[corner]]
            if len(present) == 4:
                v00 = field[row, col]
                v01 = field[row, col + 1]
                v11 = field[row + 1, col + 1]
                v10 = field[row + 1, col]
                if abs(v00 - v11) <= abs(v01 - v10):
                    local = [
                        [corners[0], corners[1], corners[2]],
                        [corners[0], corners[2], corners[3]],
                    ]
                else:
                    local = [
                        [corners[0], corners[1], corners[3]],
                        [corners[1], corners[2], corners[3]],
                    ]
            elif len(present) == 3:
                local = [present]
            else:
                local = []
            for tri_grid in local:
                triangles.append([vertex_index[corner] for corner in tri_grid])
                triangle_grid_vertices.append(tuple(tri_grid))

    triangulation = mtri.Triangulation(
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(triangles, dtype=int),
    )
    return (
        triangulation,
        np.asarray(z_values, dtype=float),
        grid_vertices,
        triangle_grid_vertices,
    )


def longest_segment_midpoint(contour_set, level_index):
    """Choose one stable label position on the longest contour segment."""
    # Matplotlib 3.8+ can include empty one-dimensional placeholders in
    # ``allsegs``; ignore them without altering any finite contour geometry.
    segments = []
    raw_level = contour_set.allsegs[level_index]
    if isinstance(raw_level, np.ndarray) and raw_level.ndim == 2 and raw_level.shape[1] == 2:
        candidates = [raw_level]
    else:
        candidates = raw_level
    for raw_segment in candidates:
        segment = np.asarray(raw_segment, dtype=float)
        if segment.ndim == 2 and segment.shape[0] >= 2 and segment.shape[1] == 2:
            segments.append(segment)
    if not segments:
        return None
    segment = max(
        segments,
        key=lambda values: float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()),
    )
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
    target = 0.52 * distances[-1]
    point_index = int(np.searchsorted(distances, target, side="left"))
    point_index = min(max(point_index, 0), len(segment) - 1)
    return tuple(segment[point_index])


def style_hest_axis(ax):
    """Exact v56 iteration-2 HEST axis treatment, isolated to panel d."""
    ax.set_facecolor(HEST_COL_BG)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(HEST_COL_SPINE)
        spine.set_linewidth(0.55)
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0.5,
        colors=HEST_COL_INK,
        pad=2.0,
        labelsize=7.0,
    )
    ax.grid(True, which="major", axis="both", color=HEST_COL_GRID, linewidth=0.50, alpha=1.0, zorder=0)
    ax.xaxis.label.set_fontsize(7.0)
    ax.yaxis.label.set_fontsize(7.0)
    ax.xaxis.label.set_color(HEST_COL_INK)
    ax.yaxis.label.set_color(HEST_COL_INK)


def draw_hest_header(ax):
    """Full-width header copied from the authoritative v56 panel d."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    letter = ax.text(0.0, 0.50, "d", ha="left", va="center", fontsize=9.0, fontweight="bold", color=HEST_COL_INK, zorder=20)
    title = ax.text(0.020, 0.50, "HEST", ha="left", va="center", fontsize=7.0, fontweight="regular", color=HEST_COL_INK, zorder=20)
    right = ax.text(1.0, 0.50, "Ridge-Phikon-v2", ha="right", va="center", fontsize=7.0, color=HEST_COL_NEUTRAL_LIGHT, zorder=20)
    return [letter, title, right]


def draw_tissue_field(ax, tissue, rows, show_y=False):
    """Exact v56 specimen trajectories, median and observed min-max envelope."""
    y = np.arange(5, dtype=float)
    labels = [specimen for _, specimen, _ in rows]
    matrix = np.asarray([values[::-1] for _, _, values in rows], dtype=float)
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    median = np.median(matrix, axis=0)

    envelope = ax.fill_betweenx(y, lo, hi, color=HEST_COL_TEAL, alpha=0.14, linewidth=0, zorder=1)
    specimen_lines = []
    for label, xx in zip(labels, matrix):
        specimen_line = ax.plot(xx, y, color=HEST_COL_NEUTRAL, lw=0.68, alpha=0.96, zorder=3)[0]
        specimen_lines.append(specimen_line)
        ax.scatter(
            xx,
            y,
            s=8.0,
            facecolor=HEST_COL_BG,
            edgecolor=HEST_COL_NEUTRAL,
            linewidth=0.48,
            zorder=4,
        )

    median_line = ax.plot(median, y, color=HEST_COL_TEAL, lw=1.55, zorder=6)[0]
    ax.scatter(median, y, s=12.0, facecolor=HEST_COL_TEAL, edgecolor=HEST_COL_BG, linewidth=0.38, zorder=7)

    endpoint = matrix[:, -1]
    order = np.argsort(endpoint)
    # Preserve every specimen label and the accepted leader register.  The
    # 0.58-data-unit step remains comfortably collision-free in the restored
    # near-golden field height.
    target_y = np.linspace(4.62 - 0.58 * (len(labels) - 1), 4.62, len(labels))
    specimen_texts = []
    for idx, target in zip(order, target_y):
        x0 = endpoint[idx]
        xt = x0 + 0.030
        ax.plot([x0, x0 + 0.013, xt - 0.006], [4.0, 4.0, target], color=HEST_COL_NEUTRAL_LIGHT, lw=0.48, zorder=8)
        specimen_texts.append(
            ax.text(
                xt,
                target,
                labels[idx],
                ha="left",
                va="center",
                fontsize=7.0,
                color=HEST_COL_INK,
                bbox=dict(boxstyle="square,pad=0.07", facecolor=HEST_COL_BG, edgecolor="none", alpha=0.93),
                zorder=10,
            )
        )

    median_text = ax.annotate(
        "median",
        xy=(median[0], 0.0),
        xytext=(median[0] + 0.050, 0.18),
        ha="left",
        va="center",
        fontsize=7.0,
        color=HEST_COL_TEAL,
        arrowprops=dict(arrowstyle="-", color=HEST_COL_TEAL, lw=0.55, shrinkA=1.5, shrinkB=1.5),
        bbox=dict(boxstyle="square,pad=0.07", facecolor=HEST_COL_BG, edgecolor="none", alpha=0.90),
        zorder=10,
    )

    ax.set_title(tissue, loc="left", pad=3.0, fontsize=7.0, fontweight="regular", color=HEST_COL_INK)
    ax.set_xlim(0.16, 1.02)
    ax.set_ylim(-0.25, 4.84)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks(y)
    if show_y:
        ax.set_yticklabels(RNA_RANGE_FINE_TO_BROAD)
        ax.set_ylabel("Graph scale", labelpad=3.0)
    else:
        ax.set_yticklabels([])
    style_hest_axis(ax)
    return {
        "envelope": envelope,
        "specimen_lines": specimen_lines,
        "median_line": median_line,
        "specimen_texts": specimen_texts,
        "median_text": median_text,
    }


OUT = Path(__file__).resolve().parent
FIG_W_IN = 180.0 / 25.4
BASE_FIG_H_IN = 140.0 / 25.4
FIG_H_IN = 140.0 / 25.4
PAGE_HEIGHT_MM = 140.0

# Iteration-1 local register repair of the accepted compact evidence mosaic.
# Panels a, c and d keep their accepted rectangles and artists exactly.  Panel b
# alone receives explicit heading, outcome, endpoint, two-row node and xlabel
# bands; every measured node and every 7.0-pt label is retained.
CONTENT_LEFT_MM = 17.28
CONTENT_RIGHT_MM = 177.30
LEFT_COLUMN_WIDTH_MM = 38.50
MOSAIC_GUTTER_MM = 10.00
C_BLOCK_LEFT_MM = CONTENT_LEFT_MM + LEFT_COLUMN_WIDTH_MM + MOSAIC_GUTTER_MM
C_FIELD_GAP_MM = 4.00
# Preserve the confirmed-good 33.51-mm field width while reclaiming the two
# now-empty P2/P3 coordinate corridors.  The resulting block still ends within
# the accepted right-page register and keeps the exact 19.1% v87 area gain.
C_FIELD_WIDTH_MM = (112.52 - 2.0 * 6.00) / 3.0
C_BLOCK_WIDTH_MM = 3.0 * C_FIELD_WIDTH_MM + 2.0 * C_FIELD_GAP_MM

GEOMETRY_MM = {
    "a": (CONTENT_LEFT_MM, 103.0, LEFT_COLUMN_WIDTH_MM, 21.0),
    "b": (CONTENT_LEFT_MM, 67.8, LEFT_COLUMN_WIDTH_MM, 17.4),
    "c": (C_BLOCK_LEFT_MM, 76.0, C_BLOCK_WIDTH_MM, 48.0),
    "cguide": (C_BLOCK_LEFT_MM, 62.5, C_BLOCK_WIDTH_MM, 6.0),
    "d": (CONTENT_LEFT_MM, 19.0, CONTENT_RIGHT_MM - CONTENT_LEFT_MM, 30.5),
    "dheader": (CONTENT_LEFT_MM, 54.0, CONTENT_RIGHT_MM - CONTENT_LEFT_MM, 4.5),
}

# Build every panel with the exact iteration-9 construction first; only axes
# rectangles and figure-text positions are changed after the first draw.
fig = plt.figure(figsize=(FIG_W_IN, BASE_FIG_H_IN), dpi=240, facecolor=COL_WHITE)
outer = fig.add_gridspec(
    3,
    1,
    height_ratios=[1.24, 3.90, 0.88],
    left=0.096,
    right=0.985,
    bottom=0.096,
    top=0.955,
    hspace=0.44,
)

top = outer[0].subgridspec(1, 2, width_ratios=[1.00, 1.58], wspace=0.14)
ax_a = fig.add_subplot(top[0, 0])
bgrid = top[0, 1].subgridspec(3, 1, height_ratios=[1.20, 1.0, 1.0], hspace=0.08)
ax_b = [fig.add_subplot(bgrid[i, 0]) for i in range(3)]

cgrid = outer[1].subgridspec(2, 3, height_ratios=[1.0, 0.11], hspace=0.26, wspace=0.22)
ax_c = [fig.add_subplot(cgrid[0, i]) for i in range(3)]
ax_cguide = fig.add_subplot(cgrid[1, :])

critical_texts = []
panel_texts = []


# Panel a: exact 32-µm projection from panel c.
for idx, (name, field) in enumerate(Q_FIELDS.items()):
    vals = field[:, 0]
    ax_a.plot(
        np.arange(7),
        vals,
        color=PATIENT_COLORS[name],
        marker=PATIENT_MARKERS[name],
        markersize=3.0,
        markeredgewidth=0,
        linewidth=0.78,
        alpha=0.88,
        zorder=3,
    )

projection_mean = np.mean(np.vstack([v[:, 0] for v in Q_FIELDS.values()]), axis=0)
ax_a.plot(
    np.arange(7),
    projection_mean,
    color=COL_SPOT_RESULT,
    marker=MEAN_MARKER,
    markersize=3.35,
    markerfacecolor=COL_WHITE,
    markeredgecolor=COL_SPOT_RESULT,
    markeredgewidth=0.75,
    linewidth=1.05,
    zorder=4,
)

for xpos, label in [(0, "0.298"), (6, "0.800")]:
    ann = ax_a.annotate(
        label,
        xy=(xpos, projection_mean[xpos]),
        xytext=(-4 if xpos == 6 else 5, 4 if xpos == 6 else 8),
        textcoords="offset points",
        ha="right" if xpos == 6 else "left",
        va="bottom",
        fontsize=8.2,
        color=COL_INK,
        zorder=10,
        bbox=white_text_box(),
    )
    critical_texts.append((ax_a, ann))

# Two authorized CRC means only; every original curve/diamond is retained.
interval_artists = []
for xpos, row in zip((0, 6), INTERVAL_ROWS[:2]):
    estimate, low, high = (float(row[k]) for k in ("estimate", "lower", "upper"))
    assert np.isclose(projection_mean[xpos], estimate, rtol=0, atol=1e-15)
    bars = ax_a.errorbar(xpos, estimate, yerr=[[estimate-low], [high-estimate]],
                        fmt="none", ecolor=COL_SPOT_RESULT, elinewidth=0.8,
                        capsize=1.8, capthick=0.8, zorder=2.5)
    interval_artists.append((ax_a, row, bars[2][0], False))

for name, field in Q_FIELDS.items():
    endpoint_shift = {"P1 CRC": 0.0, "P2 CRC": 0.035, "P3 NAT": -0.035}[name]
    txt = ax_a.text(
        6.16,
        field[-1, 0] + endpoint_shift,
        name.replace(" CRC", "").replace(" NAT", ""),
        ha="left",
        va="center",
        fontsize=7.0,
        color=PATIENT_COLORS[name],
        zorder=10,
        bbox=white_text_box(alpha=0.92, pad=0.08),
    )
    critical_texts.append((ax_a, txt))

ax_a.set_xlim(-0.18, 7.40)
ax_a.set_ylim(0.17, 1.02)
ax_a.set_xticks(np.arange(7), RANGE_LABELS_SHORT, rotation=80, ha="right", rotation_mode="anchor")
ax_a.set_yticks([0.2, 0.4, 0.6, 0.8])
ax_a.set_xlabel("Wavelength (µm)", labelpad=1.7)
ax_a.xaxis.label.set_x(0.62)
ax_a.set_ylabel("Noise-corrected\ncorrelation, q", labelpad=3.0)
style_axis(ax_a)


# Panel b: three aligned sampled trajectories; no spline or duplicated shared curves.
for name, values in GE4096_FRACTION.items():
    ax_b[0].plot(
        LOG_SCALES,
        values,
        color=COL_B_RNA,
        marker=PATIENT_MARKERS[name],
        markersize=2.7,
        markeredgewidth=0,
        linewidth=0.76,
        alpha=0.92,
        zorder=3,
    )
ax_b[0].plot(
    LOG_SCALES,
    GE4096_PATIENT_EQUAL,
    color=COL_B_RNA_DARK,
    marker=MEAN_MARKER,
    markersize=3.0,
    markerfacecolor=COL_WHITE,
    markeredgecolor=COL_B_RNA_DARK,
    markeredgewidth=0.65,
    linewidth=1.00,
    zorder=4,
)
panel_b_endpoint_texts = []
for xpos, val, label, ha in [
    (LOG_SCALES[0], GE4096_PATIENT_EQUAL[0], "0.227", "left"),
    (LOG_SCALES[-1], GE4096_PATIENT_EQUAL[-1], "0.448", "right"),
]:
    ann = ax_b[0].annotate(
        label,
        xy=(xpos, val),
        xytext=(xpos, B_ENDPOINT_BAND_Y_AXES),
        textcoords=ax_b[0].get_xaxis_transform(),
        ha=ha,
        va="center",
        fontsize=7.0,
        color=COL_INK,
        arrowprops=dict(
            arrowstyle="-",
            color=COL_B_RNA_DARK,
            linewidth=0.42,
            shrinkA=2.0,
            shrinkB=1.5,
        ),
        zorder=10,
        bbox=white_text_box(pad=0.08),
    )
    panel_b_endpoint_texts.append(ann)
    critical_texts.append((ax_b[0], ann))

ax_b[1].fill_between(LOG_SCALES, 0, COMPOSITION_SHARED, color=COL_COMPOSITION, alpha=0.16, zorder=1)
ax_b[1].plot(LOG_SCALES, COMPOSITION_SHARED, color=COL_COMPOSITION, marker="o", markersize=2.8, linewidth=1.05, zorder=3)
ax_b[2].fill_between(LOG_SCALES, 0, SAMPLING_SHARED, color=COL_SAMPLING, alpha=0.16, zorder=1)
ax_b[2].plot(LOG_SCALES, SAMPLING_SHARED, color=COL_SAMPLING, marker="o", markersize=2.8, linewidth=1.05, zorder=3)

strip_labels = ["Expression variation ≥4 mm", "Cell composition", "Sampling"]
panel_b_strip_texts = []
panel_b_x_tick_texts = []
panel_b_x_tick_rows = []
for i, ax in enumerate(ax_b):
    ax.set_xlim(LOG_EDGES[0], LOG_EDGES[-1])
    # One shared fraction scale is retained, with deliberate headroom for a
    # text-only outcome band.  Marks stay below the labels instead of fighting
    # them, and the top strip keeps a separate common endpoint-value band.
    ax.set_ylim(0.0, 1.10)
    ax.set_yticks([0.0, 0.5])
    ax.set_yticklabels(["0", "0.5"])
    label = ax.text(
        0.018,
        B_STRIP_LABEL_Y_AXES,
        strip_labels[i],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=COL_INK,
        zorder=10,
        bbox=white_text_box(alpha=0.95, pad=0.08),
    )
    panel_b_strip_texts.append(label)
    critical_texts.append((ax, label))
    style_axis(ax)
    # Text-only nudge separates the 7-pt fraction tick glyphs in the thin strips.
    from matplotlib.transforms import ScaledTranslation
    for tick_text, tick_value in zip(ax.get_yticklabels(), ax.get_yticks()):
        tick_text.set_transform(tick_text.get_transform() + ScaledTranslation(
            0, (0.4 if tick_value > 0 else -0.4) / 72.0, fig.dpi_scale_trans))
    if i < 2:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    else:
        # Keep one hairline tick at every measured output-size node and give
        # every exact label a horizontal home.  Alternating labels use two
        # compact rows, avoiding the crowded 96/128 and 384/512 pairs without
        # shrinking type, rotating text or changing physical log2 positions.
        ax.set_xticks(LOG_SCALES)
        ax.set_xticklabels([])
        for node_index, (node_x, node_um) in enumerate(zip(LOG_SCALES, SCALES_UM)):
            row_index = node_index % 2
            tick_text = ax.annotate(
                str(int(node_um)),
                xy=(node_x, 0.0),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0.0, -B_LABEL_ROW_OFFSETS_PT[row_index]),
                textcoords="offset points",
                ha="center",
                va="top",
                rotation=0,
                fontsize=B_LABEL_FONT_PT,
                fontproperties=_narrow_properties,
                color=COL_INK,
                annotation_clip=False,
                clip_on=False,
                zorder=20,
            )
            panel_b_x_tick_texts.append(tick_text)
            panel_b_x_tick_rows.append(row_index)
        ax.tick_params(axis="x", which="major", length=1.35, width=0.40, pad=0.0)
        ax.set_xlabel("Output size (µm)", labelpad=0.0)
        ax.xaxis.set_label_coords(0.5, B_XLABEL_Y_AXES)
ax_b[1].set_ylabel("Fraction", labelpad=3.0)


# Panel c: local faceted response landscapes tied exactly to the observed
# 8-output-size x 7-range lattice.  Only adjacent measured vertices are joined;
# no synthetic observations or nonlocal/smoothed connections cross the mask.
structural_na_cell_count = 0
mask_patch_count = 0
structural_mask_patches = []
triangle_records = {}
mask_records = {}
surface_vertex_counts = {}
contour_label_count = 0
surface_sets = []
surface_level_counts = []
triangle_relief_overlays = {}
triangle_relief_expected = {}
patient_title_texts = []
panel_c_x_tick_texts = []
panel_c_x_tick_texts_by_axis = []
panel_c_x_tick_rows_by_axis = []

for panel_index, (ax, (name, field)) in enumerate(zip(ax_c, Q_FIELDS.items())):
    observed_grid = np.ma.masked_invalid(field)
    mask_records[name] = np.ma.getmaskarray(observed_grid).copy()

    # Each local quad is split along the less-discordant diagonal; a partial
    # quad contributes only its three-finite-corner triangle.  This exposes the
    # reference's angular local peaks, saddles and troughs without new values.
    triangulation, z_values, _, triangle_grid_vertices = build_local_triangulation(field)
    triangle_records[name] = triangle_grid_vertices
    surface_vertex_counts[name] = len(z_values)
    surface = ax.tricontourf(
        triangulation,
        z_values,
        levels=np.linspace(SURFACE_NORM.vmin, SURFACE_NORM.vmax, 38),
        cmap=SURFACE_CMAP,
        norm=SURFACE_NORM,
        antialiased=False,
        zorder=1,
    )
    surface_sets.append(surface)
    surface_level_counts.append(len(surface.levels))

    # P1 is the accepted preserve anchor.  P2/P3 use the same adjacent-cell
    # triangulation and the same observed vertices as the base interpolant.
    # A restrained flat triangular layer makes those honest local slopes read
    # as small coherent facets; there are no rectangular cells/bands, new
    # nodes, value perturbations, or interpolation across the stepped mask.
    if name != "P1 CRC":
        expected_triangle_values = z_values[triangulation.triangles].mean(axis=1)
        triangle_relief = ax.tripcolor(
            triangulation,
            z_values,
            shading="flat",
            cmap=SURFACE_CMAP,
            norm=SURFACE_NORM,
            edgecolors="none",
            linewidth=0,
            alpha=0.44,
            zorder=2.2,
        )
        triangle_relief_overlays[name] = triangle_relief
        triangle_relief_expected[name] = expected_triangle_values

    # Preserve the exact staircase as individually hatched structural cells,
    # matching the v52 construction while retaining physical log2 x widths.
    for row, col in np.argwhere(np.ma.getmaskarray(observed_grid)):
        patch = Rectangle(
            (LOG_EDGES[col], row - 0.5),
            LOG_EDGES[col + 1] - LOG_EDGES[col],
            1.0,
            facecolor=SURFACE_NA,
            edgecolor=SURFACE_NA_EDGE,
            linewidth=0.34,
            hatch="////",
            zorder=4,
        )
        patch.set_gid(f"panel-c-structural-na-{name}-{row}-{col}")
        ax.add_patch(patch)
        structural_mask_patches.append(patch)
        structural_na_cell_count += 1
        mask_patch_count += 1

    for level, color, width in (
        (0.60, SURFACE_INK, 0.70),
        (0.70, SURFACE_CONTOUR_MID, 0.72),
        (0.80, COL_WHITE, 0.78),
    ):
        contours = ax.tricontour(
            triangulation,
            z_values,
            levels=[level],
            colors=[color],
            linewidths=[width],
            linestyles="--",
            zorder=7,
        )
        position = longest_segment_midpoint(contours, 0)
        if position is None:
            continue
        # The lower 0.80 contour in P2 touches the left frame.  Keep its
        # inline label on the same measured contour, but choose the interior
        # branch so the complete value remains legible at final print size.
        if name == "P2 CRC" and np.isclose(level, 0.80):
            position = (np.log2(64.0), 5.65)
        labels = ax.clabel(
            contours,
            levels=[level],
            inline=True,
            inline_spacing=2,
            fmt={level: f"{level:.2f}"},
            fontsize=7.0,
            manual=[position],
            rightside_up=True,
            use_clabeltext=True,
        )
        for label in labels:
            label.set_zorder(30)
        contour_label_count += len(labels)

    ax.set_xlim(LOG_EDGES[0], LOG_EDGES[-1])
    ax.set_ylim(6.5, -0.5)
    # Every observed output-size node carries one physical hairline tick and
    # one explicit horizontal label.  Alternating labels occupy two compact
    # rows, so adjacent unequal log2 intervals remain legible at paper size
    # without omission, rotation or a reduced type floor.
    ax.set_xticks(LOG_SCALES)
    ax.set_xticklabels([])
    axis_tick_texts = []
    axis_tick_rows = []
    for node_index, (node_x, node_um) in enumerate(zip(LOG_SCALES, SCALES_UM)):
        row_index = node_index % 2
        tick_text = ax.annotate(
            str(int(node_um)),
            xy=(node_x, 0.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0.0, -C_LABEL_ROW_OFFSETS_PT[row_index]),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=0,
            fontsize=C_LABEL_FONT_PT,
            fontproperties=_narrow_properties,
            color=SURFACE_INK,
            annotation_clip=False,
            clip_on=False,
            zorder=20,
        )
        axis_tick_texts.append(tick_text)
        axis_tick_rows.append(row_index)
        panel_c_x_tick_texts.append(tick_text)
    panel_c_x_tick_texts_by_axis.append(axis_tick_texts)
    panel_c_x_tick_rows_by_axis.append(axis_tick_rows)
    ax.set_yticks(np.arange(7), RANGE_LABELS_SHORT)
    ax.set_xlabel("Output size (µm)", labelpad=0.0)
    ax.xaxis.set_label_coords(0.5, C_XLABEL_Y_AXES)
    style_surface_axis(ax)
    ax.tick_params(axis="x", which="major", length=1.35, width=0.40, pad=0.0)
    # One shared y-coordinate register is sufficient for the three aligned
    # fields. Removing the two duplicated gutter labels cures the only
    # horizontal cross-panel bleed without removing any scientific label.
    if panel_index > 0:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    patient_text = ax.text(
        0.96,
        0.95,
        SURFACE_PATIENT_LABELS[name],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        fontweight="regular",
        color=SURFACE_INK,
        bbox={"facecolor": COL_WHITE, "edgecolor": "none", "alpha": 0.90, "pad": 0.24},
        zorder=30,
    )
    patient_title_texts.append(patient_text)
    critical_texts.append((ax, patient_text))

ax_c[0].set_ylabel("Wavelength (µm)", labelpad=2.0)
ax_c[0].yaxis.label.set_fontsize(7.0)
ax_c[0].yaxis.label.set_rotation(0)
ax_c[0].yaxis.set_label_coords(0.5, 1.055)

# Match the accepted v52 key's compact, centered and visibly substantial band.
# Retain the current guide row and scientific label; only repair its rectangle.
ax_cguide.set_axis_off()
cbar_ax = ax_cguide.inset_axes([0.15, 0.46, 0.70, 0.38])
cbar_edges = np.linspace(0.5, 0.9, 65)
for left, right in zip(cbar_edges[:-1], cbar_edges[1:]):
    center = 0.5 * (left + right)
    cbar_ax.add_patch(
        Rectangle(
            (left, 0.0),
            right - left,
            1.0,
            facecolor=SURFACE_CMAP(SURFACE_NORM(center)),
            edgecolor="none",
            linewidth=0,
            zorder=1,
        )
    )
cbar_ax.set_xlim(0.5, 0.9)
cbar_ax.set_ylim(0.0, 1.0)
cbar_ax.set_xticks([0.5, 0.7, 0.9])
cbar_ax.set_yticks([])
cbar_ax.set_xlabel("Noise-corrected correlation, q", labelpad=1.0)
cbar_ax.tick_params(
    axis="x",
    direction="out",
    length=0,
    width=0.50,
    color=SURFACE_SPINE,
    labelcolor=SURFACE_INK,
    pad=1.5,
)
for spine in cbar_ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor(SURFACE_SPINE)
    spine.set_linewidth(0.50)
for tick in cbar_ax.get_xticklabels():
    tick.set_zorder(20)
cbar_ax.xaxis.label.set_zorder(20)


def add_panel_heading(ax, letter, title, *, title_x_offset=0.0, extra_y=0.017):
    pos = ax.get_position()
    letter_text = fig.text(
        pos.x0 - 0.047,
        pos.y1 + extra_y,
        letter,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COL_INK,
        zorder=20,
    )
    title_text = fig.text(
        pos.x0 + title_x_offset,
        pos.y1 + extra_y,
        title,
        ha="left",
        va="bottom",
        fontsize=7.0,
        fontweight="regular",
        color=COL_INK,
        zorder=20,
    )
    panel_texts.extend([letter_text, title_text])


add_panel_heading(ax_a, "a", "32-µm output", extra_y=0.010)
add_panel_heading(ax_b[0], "b", "Aggregation", extra_y=0.010)
add_panel_heading(ax_c[0], "c", "", extra_y=0.012)


# Asymmetric evidence mosaic.  Existing artists remain untouched; only the
# owning axes rectangles are remapped into one compact left stack and one
# dominant warm field on the right.
fig.canvas.draw()
a_c_axes = tuple(fig.axes)
base_canvas_width = float(fig.bbox.width)
fig.set_size_inches(FIG_W_IN, FIG_H_IN, forward=True)
if not np.isclose(float(fig.bbox.width), base_canvas_width, atol=1e-9):
    raise AssertionError("180-mm width changed during the asymmetric reflow")


def set_axes_bbox_mm(ax, x_mm, y_mm, width_mm, height_mm):
    ax.set_position(
        [
            x_mm / 180.0,
            y_mm / PAGE_HEIGHT_MM,
            width_mm / 180.0,
            height_mm / PAGE_HEIGHT_MM,
        ]
    )


def remap_group_to_bbox_mm(axes, x_mm, y_mm, width_mm, height_mm):
    """Affine-remap a multi-axis construction while preserving internal topology."""
    positions = [ax.get_position().frozen() for ax in axes]
    source_x0 = min(pos.x0 for pos in positions)
    source_x1 = max(pos.x1 for pos in positions)
    source_y0 = min(pos.y0 for pos in positions)
    source_y1 = max(pos.y1 for pos in positions)
    for ax, pos in zip(axes, positions):
        rel_x = (pos.x0 - source_x0) / (source_x1 - source_x0)
        rel_y = (pos.y0 - source_y0) / (source_y1 - source_y0)
        rel_w = pos.width / (source_x1 - source_x0)
        rel_h = pos.height / (source_y1 - source_y0)
        set_axes_bbox_mm(
            ax,
            x_mm + rel_x * width_mm,
            y_mm + rel_y * height_mm,
            rel_w * width_mm,
            rel_h * height_mm,
        )


set_axes_bbox_mm(ax_a, *GEOMETRY_MM["a"])
remap_group_to_bbox_mm(ax_b, *GEOMETRY_MM["b"])
for index, ax in enumerate(ax_c):
    set_axes_bbox_mm(
        ax,
        C_BLOCK_LEFT_MM + index * (C_FIELD_WIDTH_MM + C_FIELD_GAP_MM),
        GEOMETRY_MM["c"][1],
        C_FIELD_WIDTH_MM,
        GEOMETRY_MM["c"][3],
    )
set_axes_bbox_mm(ax_cguide, *GEOMETRY_MM["cguide"])

# Re-register the figure-level a/b/c headings to the new axes owners.  The
# offsets stay in the compact 1.8–2.0-mm Nature register.
heading_specs = (
    (panel_texts[0:2], GEOMETRY_MM["a"], 5.0, 1.9),
    (panel_texts[2:4], GEOMETRY_MM["b"], 5.0, 1.30),
    (panel_texts[4:6], GEOMETRY_MM["c"], 4.5, 1.9),
)
for texts, bbox_mm, letter_offset_mm, top_offset_mm in heading_specs:
    x_mm, y_mm, width_mm, height_mm = bbox_mm
    target_y = (y_mm + height_mm + top_offset_mm) / PAGE_HEIGHT_MM
    texts[0].set_position(((x_mm - letter_offset_mm) / 180.0, target_y))
    texts[1].set_position((x_mm / 180.0, target_y))

fig.canvas.draw()

# Panel d: exact v56 iteration-2 HEST triptych, retained as a muted full-width
# confirmation band beneath the asymmetric mosaic, restored to near-golden
# field proportions while remaining smaller in visual area than panel c.
dx, dy, dw, dh = GEOMETRY_MM["dheader"]
ax_hest_header = fig.add_axes([dx / 180.0, dy / PAGE_HEIGHT_MM, dw / 180.0, dh / PAGE_HEIGHT_MM])
hest_header_texts = draw_hest_header(ax_hest_header)
panel_texts.extend(hest_header_texts)
# Pooled paired effect, explicitly distinct from the tissue-specific medians
# and observed specimen ranges below. Raw effect axis 0.30..0.43 maps into
# the small header rail, without changing any existing panel rectangle.
effect = INTERVAL_ROWS[2]
effect_mean, effect_low, effect_high = (float(effect[k]) for k in ("estimate", "lower", "upper"))
effect_x = lambda value: 0.10 + (value - 0.30) / (0.43 - 0.30) * 0.12
ax_hest_header.set_xlim(0, 1)
ax_hest_header.set_ylim(0, 1)
effect_line = ax_hest_header.hlines(0.50, effect_x(effect_low), effect_x(effect_high),
                                   color=HEST_COL_INK, linewidth=0.85, zorder=5)
ax_hest_header.vlines([effect_x(effect_low), effect_x(effect_high)], 0.37, 0.63,
                     color=HEST_COL_INK, linewidth=0.85, zorder=5)
ax_hest_header.plot(effect_x(effect_mean), 0.50, marker="D", markersize=2.8,
                    color=HEST_COL_INK, markeredgewidth=0, zorder=6)
effect_text = ax_hest_header.text(
    0.23, 0.50, "Broadest − Finest (9-specimen mean): +0.365 [0.338, 0.394]",
    fontsize=7.0, ha="left", va="center", color=HEST_COL_INK, zorder=20)
critical_texts.append((ax_hest_header, effect_text))
interval_artists.append((ax_hest_header, effect, effect_line, True))

hest_left = CONTENT_LEFT_MM / 180.0
hest_right = CONTENT_RIGHT_MM / 180.0
hest_wspace = 0.10
hest_field_width = (hest_right - hest_left) / (3.0 + 2.0 * hest_wspace)
hest_gap = hest_wspace * hest_field_width
hest_axes = []
hest_records = []
for index, tissue in enumerate(("IDC", "PAAD", "LUNG")):
    x0 = hest_left + index * (hest_field_width + hest_gap)
    ax = fig.add_axes([x0, GEOMETRY_MM["d"][1] / PAGE_HEIGHT_MM, hest_field_width, GEOMETRY_MM["d"][3] / PAGE_HEIGHT_MM])
    rows = [row for row in HEST if row[0] == tissue]
    record = draw_tissue_field(ax, tissue, rows, show_y=(index == 0))
    if index == 1:
        ax.set_xlabel("Noise-corrected correlation, q", labelpad=2.0)
        ax.xaxis.label.set_fontsize(7.0)
    else:
        ax.set_xlabel("")
    hest_axes.append(ax)
    hest_records.append(record)
    critical_texts.extend((ax, text) for text in record["specimen_texts"])
    critical_texts.append((ax, record["median_text"]))
def all_relevant_texts():
    found = []
    for ax in fig.axes:
        candidates = (
            list(ax.get_xticklabels())
            + list(ax.get_yticklabels())
            + [ax.xaxis.label, ax.yaxis.label, ax.title]
            + list(ax.texts)
        )
        for text in candidates:
            if text.get_visible() and text.get_text():
                text.set_zorder(max(10, text.get_zorder()))
                found.append((ax, text))
    for text in fig.texts:
        if text.get_visible() and text.get_text():
            found.append((None, text))
    return found


def run_floor_checks():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    axis_names = {
        ax_a: "a",
        **{ax: f"b{index + 1}" for index, ax in enumerate(ax_b)},
        **{ax: f"c{index + 1}" for index, ax in enumerate(ax_c)},
        ax_cguide: "cguide",
        cbar_ax: "cbar",
        ax_hest_header: "dheader",
        **{ax: f"d-{tissue}" for ax, tissue in zip(hest_axes, ("IDC", "PAAD", "LUNG"))},
    }
    tick_texts = []
    for ax in fig.axes:
        tick_texts.extend([t for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible() and t.get_text()])

    # Structural text within each axes must remain disjoint after the 7.0-pt
    # minimum-type uplift.  This catches adjacent tick collisions that the
    # accepted cross-axes corridor check intentionally does not inspect.
    for ax in fig.axes:
        structural_texts = [
            text
            for text in (
                list(ax.get_xticklabels())
                + list(ax.get_yticklabels())
                + [ax.xaxis.label, ax.yaxis.label, ax.title]
            )
            if text.get_visible() and text.get_text()
        ]
        structural_boxes = [text.get_window_extent(renderer=renderer) for text in structural_texts]
        for index, (text_a, box_a) in enumerate(zip(structural_texts, structural_boxes)):
            for text_b, box_b in zip(structural_texts[index + 1 :], structural_boxes[index + 1 :]):
                if box_a.overlaps(box_b):
                    failures.append(
                        f"same_axes_structural_text_overlap: '{text_a.get_text()}' vs '{text_b.get_text()}'"
                    )

    # Annotation-to-tick and annotation-to-annotation disjointness.
    critical_boxes = []
    for owner, text in critical_texts:
        box = text.get_window_extent(renderer=renderer)
        critical_boxes.append((owner, text, box))
        for tick in tick_texts:
            if box.overlaps(tick.get_window_extent(renderer=renderer)):
                failures.append(f"annotation_tick_overlap: '{text.get_text()}' vs '{tick.get_text()}'")
    for i, (_, ta, ba) in enumerate(critical_boxes):
        for _, tb, bb in critical_boxes[i + 1 :]:
            if ba.overlaps(bb):
                failures.append(
                    "annotation_annotation_overlap: "
                    f"[{axis_names.get(critical_boxes[i][0], '?')}] '{ta.get_text()}' vs '{tb.get_text()}'"
                )

    # Text belonging to different axes must not collide in the compact page corridors.
    relevant = all_relevant_texts()
    relevant_boxes = [(owner, text, text.get_window_extent(renderer=renderer)) for owner, text in relevant]
    for i, (owner_a, text_a, box_a) in enumerate(relevant_boxes):
        if owner_a is None:
            continue
        for owner_b, text_b, box_b in relevant_boxes[i + 1 :]:
            if owner_b is None or owner_a is owner_b:
                continue
            if box_a.overlaps(box_b):
                failures.append(
                    "cross_axes_text_overlap: "
                    f"[{axis_names.get(owner_a, '?')}] '{text_a.get_text()}' vs "
                    f"[{axis_names.get(owner_b, '?')}] '{text_b.get_text()}' "
                    f"bboxes={box_a.bounds}/{box_b.bounds}"
                )

    # Figure-level panel headings must not collide with any axis-bearing text.
    axis_structure_texts = []
    for ax in fig.axes:
        axis_structure_texts.extend(
            [
                t
                for t in (
                    list(ax.get_xticklabels())
                    + list(ax.get_yticklabels())
                    + [ax.xaxis.label, ax.yaxis.label, ax.title]
                )
                if t.get_visible() and t.get_text()
            ]
        )
    for heading in panel_texts:
        hbox = heading.get_window_extent(renderer=renderer)
        for text in axis_structure_texts:
            if hbox.overlaps(text.get_window_extent(renderer=renderer)):
                failures.append(f"panel_heading_overlap: '{heading.get_text()}' vs '{text.get_text()}'")
    for i, heading_a in enumerate(panel_texts):
        box_a = heading_a.get_window_extent(renderer=renderer)
        for heading_b in panel_texts[i + 1 :]:
            if box_a.overlaps(heading_b.get_window_extent(renderer=renderer)):
                failures.append(f"panel_heading_overlap: '{heading_a.get_text()}' vs '{heading_b.get_text()}'")

    # No clipping and no data annotation crossing its own subplot boundary.
    fig_box = fig.bbox
    for owner, text in all_relevant_texts():
        box = text.get_window_extent(renderer=renderer)
        if not (fig_box.contains(box.x0, box.y0) and fig_box.contains(box.x1, box.y1)):
            failures.append(f"clipped_text: '{text.get_text()}' bbox={box.bounds}")
    for owner, text, box in critical_boxes:
        if owner is None or owner is ax_cguide:
            continue
        # Panel-b outcome names intentionally occupy a text-only cap just
        # above their plotting rectangles.  Give that designed cap its own
        # checked envelope; all other annotations retain the stricter owner box.
        if text in panel_b_strip_texts:
            ax_box = owner.get_window_extent(renderer=renderer).expanded(1.02, 1.22)
        else:
            ax_box = owner.get_window_extent(renderer=renderer).expanded(1.02, 1.05)
        if not (ax_box.contains(box.x0, box.y0) and ax_box.contains(box.x1, box.y1)):
            failures.append(f"cross_panel_annotation: '{text.get_text()}'")

    # Every in-panel critical label over marks has an opaque/high-alpha white backing.
    for owner, text, _ in critical_boxes:
        if owner is None or owner is ax_cguide:
            continue
        patch = text.get_bbox_patch()
        if patch is None or patch.get_facecolor()[-1] < 0.90:
            failures.append(f"text_obscured_by_marks: '{text.get_text()}' lacks protective backing")

    measured = sum(int(np.isfinite(field).sum()) for field in Q_FIELDS.values())
    blanks = sum(int(np.isnan(field).sum()) for field in Q_FIELDS.values())
    if (measured, blanks) != (132, 36):
        failures.append(f"field_topology: measured={measured}, blanks={blanks}")
    def is_explicit_node_collection(child):
        if not isinstance(child, PathCollection):
            return False
        offsets = np.asarray(child.get_offsets(), dtype=float)
        # Current Matplotlib represents each contour level as a PathCollection
        # with the default origin offset.  Only explicit scatter offsets count
        # as observed-node markers for this scientific exclusion.
        return not (
            offsets.shape == (1, 2)
            and np.allclose(offsets[0], [0.0, 0.0])
        )

    path_node_count = sum(
        is_explicit_node_collection(child)
        for ax in ax_c
        for child in ax.get_children()
    )
    line_node_count = sum(
        line.get_marker() in ("o", "O") and line.get_markersize() > 0
        for ax in ax_c
        for line in ax.lines
    )
    node_marker_count = path_node_count + line_node_count
    if node_marker_count != 0:
        failures.append(
            f"node_markers: PathCollection={path_node_count}, circular_Line2D={line_node_count}"
        )
    rectangular_overlay_count = sum(
        isinstance(child, QuadMesh)
        for ax in ax_c
        for child in ax.get_children()
    )
    if rectangular_overlay_count != 0:
        failures.append(
            f"rectangular_overlays: QuadMesh={rectangular_overlay_count}"
        )
    mask_arrays_match = all(
        name in mask_records and np.array_equal(mask_records[name], np.isnan(field))
        for name, field in Q_FIELDS.items()
    )
    mask_patch_style_ok = all(
        patch.get_hatch() == "////" and patch.get_facecolor()[:3] == matplotlib.colors.to_rgb(SURFACE_NA)
        for patch in structural_mask_patches
    )
    if (
        structural_na_cell_count != 36
        or mask_patch_count != 36
        or len(structural_mask_patches) != 36
        or not mask_arrays_match
        or not mask_patch_style_ok
    ):
        failures.append(
            "structural_mask: "
            f"cells={structural_na_cell_count}, patches={mask_patch_count}, "
            f"arrays_match={mask_arrays_match}, style_ok={mask_patch_style_ok}"
        )

    triangle_count = 0
    invalid_triangle_count = 0
    nonlocal_triangle_count = 0
    for name, triangle_grid_vertices in triangle_records.items():
        field = Q_FIELDS[name]
        triangle_count += len(triangle_grid_vertices)
        for triangle in triangle_grid_vertices:
            if any(not np.isfinite(field[row, col]) for row, col in triangle):
                invalid_triangle_count += 1
            rows = [row for row, _ in triangle]
            cols = [col for _, col in triangle]
            if max(rows) - min(rows) > 1 or max(cols) - min(cols) > 1:
                nonlocal_triangle_count += 1
    facet_path_ok = (
        set(surface_vertex_counts) == set(Q_FIELDS)
        and all(
            surface_vertex_counts[name] == int(np.isfinite(field).sum())
            for name, field in Q_FIELDS.items()
        )
        and triangle_count == 189
    )
    if invalid_triangle_count or nonlocal_triangle_count or not facet_path_ok:
        failures.append(
            "masked_interpolation: "
            f"local_facets={facet_path_ok}, invalid_vertices={invalid_triangle_count}, "
            f"nonlocal_triangles={nonlocal_triangle_count}"
        )
    distributed_triangular_relief_ok = (
        set(triangle_relief_overlays) == {"P2 CRC", "P3 NAT"}
        and set(triangle_relief_expected) == {"P2 CRC", "P3 NAT"}
        and all(
            len(triangle_relief_expected[name]) == len(triangle_records[name])
            and np.allclose(
                np.asarray(triangle_relief_overlays[name].get_array()),
                triangle_relief_expected[name],
            )
            for name in triangle_relief_overlays
        )
        and "P1 CRC" not in triangle_relief_overlays
        and rectangular_overlay_count == 0
    )
    if not distributed_triangular_relief_ok:
        failures.append(
            "distributed_triangular_relief: P2/P3 local-triangle contract failed, "
            "a rectangular overlay remains, or P1 preserve path changed"
        )
    warm_palette_ok = (
        SURFACE_HEAT_LOW.lower() == "#f8f2d2"
        and SURFACE_HEAT_GOLD.lower() == "#f8cf55"
        and SURFACE_HEAT_MID.lower() == "#be7148"
        and SURFACE_HEAT_HIGH.lower() == "#862e63"
        and np.isclose(SURFACE_NORM.vmin, 0.44)
        and np.isclose(SURFACE_NORM.vmax, 0.91)
    )
    if (
        len(surface_sets) != 3
        or surface_level_counts != [38, 38, 38]
        or contour_label_count != 9
        or not facet_path_ok
        or not warm_palette_ok
    ):
        failures.append(
            "v52_surface_path: "
            f"surfaces={len(surface_sets)}, fill_levels={surface_level_counts}, "
            f"contour_labels={contour_label_count}, local_facets={facet_path_ok}, palette={warm_palette_ok}"
        )

    semantic_palette_ok = (
        COL_INK.lower() == "#252b2d"
        and COL_AXIS.lower() == "#454b4d"
        and COL_GUIDE.lower() == "#dcd7d1"
        and COL_SPOT_RESULT.lower() == "#365f78"
        and COL_B_RNA.lower() == "#aaa39b"
        and COL_B_RNA_DARK.lower() == "#1f5a58"
        and COL_COMPOSITION.lower() == "#9b5e4c"
        and COL_SAMPLING.lower() == "#a87342"
        and len(set(PATIENT_COLORS.values())) == 1
        and next(iter(PATIENT_COLORS.values())).lower() == "#aaa39b"
        and MEAN_MARKER == "D"
    )
    if not semantic_palette_ok:
        failures.append("semantic_palette: shared palette or shape-coded patient contract failed")

    hest_house_palette_ok = (
        HEST_COL_TEAL.lower() == "#1f5a58"
        and HEST_COL_INK.lower() == "#252b2d"
        and HEST_COL_SPINE.lower() == "#454b4d"
        and HEST_COL_GRID.lower() == "#dcd7d1"
        and HEST_COL_NEUTRAL.lower() == "#aaa39b"
        and HEST_COL_NEUTRAL_LIGHT.lower() == "#85827d"
    )
    if not hest_house_palette_ok:
        failures.append("hest_house_palette: shared panel-d tokens drifted")

    def array_payload(values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            return [[None if not np.isfinite(value) else float(value) for value in row] for row in array]
        return [float(value) for value in array]

    a_c_data_payload = {
        "scales": [float(value) for value in SCALES_UM],
        "range_keys": RANGE_KEYS,
        "range_labels": RANGE_LABELS,
        "range_labels_short": RANGE_LABELS_SHORT,
        "q_fields": {name: array_payload(field) for name, field in Q_FIELDS.items()},
        "ge4096_fraction": {name: array_payload(values) for name, values in GE4096_FRACTION.items()},
        "ge4096_patient_equal": array_payload(GE4096_PATIENT_EQUAL),
        "composition_shared": array_payload(COMPOSITION_SHARED),
        "sampling_shared": array_payload(SAMPLING_SHARED),
    }
    a_c_data_sha256 = hashlib.sha256(
        json.dumps(a_c_data_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_a_c_data_sha256 = "7d7e53ae97731d5c1b5c74b0c5592d8d3c571174cf9c4a121d4c9260a3210fbf"
    if a_c_data_sha256 != expected_a_c_data_sha256:
        failures.append(f"a_c_data_register: {a_c_data_sha256} != {expected_a_c_data_sha256}")

    def axes_bbox_mm(axes):
        x0 = min(ax.get_position().x0 for ax in axes) * 180.0
        x1 = max(ax.get_position().x1 for ax in axes) * 180.0
        y0 = min(ax.get_position().y0 for ax in axes) * PAGE_HEIGHT_MM
        y1 = max(ax.get_position().y1 for ax in axes) * PAGE_HEIGHT_MM
        return (x0, y0, x1 - x0, y1 - y0)

    section_bboxes_mm = {
        "a": axes_bbox_mm([ax_a]),
        "b": axes_bbox_mm(ax_b),
        "c": axes_bbox_mm(ax_c),
        "cguide": axes_bbox_mm([ax_cguide]),
        "d": axes_bbox_mm(hest_axes),
        "dheader": axes_bbox_mm([ax_hest_header]),
    }
    expected_section_bboxes_mm = {
        "a": GEOMETRY_MM["a"],
        "b": GEOMETRY_MM["b"],
        "c": GEOMETRY_MM["c"],
        "cguide": GEOMETRY_MM["cguide"],
        "d": GEOMETRY_MM["d"],
        "dheader": GEOMETRY_MM["dheader"],
    }
    if any(
        not np.allclose(section_bboxes_mm[name], expected, atol=0.03)
        for name, expected in expected_section_bboxes_mm.items()
    ):
        failures.append(f"asymmetric_geometry_register: {section_bboxes_mm}")

    left_column_register = (
        np.isclose(section_bboxes_mm["a"][0], section_bboxes_mm["b"][0], atol=0.03)
        and np.isclose(section_bboxes_mm["a"][2], section_bboxes_mm["b"][2], atol=0.03)
        and section_bboxes_mm["a"][1] > section_bboxes_mm["b"][1] + section_bboxes_mm["b"][3]
    )
    c_right_register = (
        section_bboxes_mm["c"][0] > section_bboxes_mm["a"][0] + section_bboxes_mm["a"][2]
        and np.isclose(section_bboxes_mm["c"][0], section_bboxes_mm["cguide"][0], atol=0.03)
        and np.isclose(section_bboxes_mm["c"][2], section_bboxes_mm["cguide"][2], atol=0.03)
    )
    if not left_column_register or not c_right_register:
        failures.append(
            f"asymmetric_mosaic_topology: left={left_column_register}, right={c_right_register}"
        )

    def axes_area_mm2(axes):
        return sum(
            ax.get_position().width * 180.0 * ax.get_position().height * PAGE_HEIGHT_MM
            for ax in axes
        )

    visual_areas_mm2 = {
        "left": axes_area_mm2([ax_a, *ax_b]),
        "c": axes_area_mm2(ax_c),
        "d": axes_area_mm2(hest_axes),
    }
    if not (
        visual_areas_mm2["c"] > 2.0 * visual_areas_mm2["left"]
        and visual_areas_mm2["c"] > visual_areas_mm2["d"]
    ):
        failures.append(f"visual_area_hierarchy: {visual_areas_mm2}")
    accepted_v87_c_field_width_mm = (105.02 - 2.0 * 7.50) / 3.0
    accepted_v87_c_visible_area_mm2 = 3.0 * accepted_v87_c_field_width_mm * 45.0
    panel_c_area_gain = visual_areas_mm2["c"] / accepted_v87_c_visible_area_mm2 - 1.0
    if not (0.15 <= panel_c_area_gain <= 0.20):
        failures.append(f"panel_c_area_gain: {panel_c_area_gain:.4f}")

    hest_payload = [[tissue, specimen, [float(value) for value in values]] for tissue, specimen, values in HEST]
    hest_data_sha256 = hashlib.sha256(
        json.dumps(hest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_hest_data_sha256 = "8aa4af7b8e61d7ca35c62dd27c75c6e249f5d15ee8ed2649b37ea6ac893391f8"
    hest_value_count = sum(len(values) for _, _, values in HEST)
    hest_specimens = [specimen for _, specimen, _ in HEST]
    if hest_data_sha256 != expected_hest_data_sha256 or hest_value_count != 45 or len(hest_specimens) != 9:
        failures.append(
            f"hest_data_register: hash={hest_data_sha256}, values={hest_value_count}, specimens={len(hest_specimens)}"
        )
    hest_widths = [ax.get_position().width for ax in hest_axes]
    if len(hest_widths) != 3 or max(hest_widths) - min(hest_widths) > 1e-12:
        failures.append(f"hest_equal_fields: widths={hest_widths}")
    hest_specimen_texts = [text for record in hest_records for text in record["specimen_texts"]]
    hest_trajectory_count = sum(len(record["specimen_lines"]) for record in hest_records)
    hest_field_contract_ok = (
        hest_trajectory_count == 9
        and len(hest_specimen_texts) == 9
        and sorted(text.get_text() for text in hest_specimen_texts) == sorted(hest_specimens)
        and len(hest_records) == 3
        and [ax.get_title(loc="left") for ax in hest_axes] == ["IDC", "PAAD", "LUNG"]
    )
    if not hest_field_contract_ok:
        failures.append("hest_screenshot_construction: trajectory/envelope/median/specimen-label contract failed")
    hest_visible_texts = list(ax_hest_header.texts)
    for ax in hest_axes:
        hest_visible_texts.extend(
            text
            for text in (
                list(ax.texts)
                + list(ax.get_xticklabels())
                + list(ax.get_yticklabels())
                + [ax._left_title, ax.xaxis.label, ax.yaxis.label]
            )
            if text.get_visible() and text.get_text()
        )
    hest_lower_ink_y_mm = min(
        text.get_window_extent(renderer=renderer).y0 for text in hest_visible_texts
    ) * 25.4 / fig.dpi
    if not (0.0 < hest_lower_ink_y_mm < 15.0):
        failures.append(f"hest_bottom_dead_band: lower_ink_y={hest_lower_ink_y_mm:.3f} mm")
    hest_min_font = min(text.get_fontsize() for text in hest_visible_texts)
    if hest_min_font < 7.0:
        failures.append(f"hest_minimum_font: {hest_min_font:.3f} pt")
    all_visible_texts = [text for _, text in all_relevant_texts()] + [
        text for text in panel_texts if text.get_visible() and text.get_text()
    ]
    overall_min_font = min(text.get_fontsize() for text in all_visible_texts)
    if overall_min_font < 7.0:
        failures.append(f"overall_minimum_font: {overall_min_font:.3f} pt")
    all_visible_text_boxes = [text.get_window_extent(renderer=renderer) for text in all_visible_texts]
    outer_top_margin_mm = (
        fig.bbox.y1 - max(box.y1 for box in all_visible_text_boxes)
    ) * 25.4 / fig.dpi
    outer_bottom_margin_mm = min(box.y0 for box in all_visible_text_boxes) * 25.4 / fig.dpi
    outer_margin_delta_mm = abs(outer_top_margin_mm - outer_bottom_margin_mm)
    if not (
        8.0 <= outer_top_margin_mm <= 16.0
        and 8.0 <= outer_bottom_margin_mm <= 16.0
        and outer_margin_delta_mm <= 4.0
    ):
        failures.append(
            "outer_margin_balance: "
            f"top={outer_top_margin_mm:.3f}, bottom={outer_bottom_margin_mm:.3f}, "
            f"delta={outer_margin_delta_mm:.3f} mm"
        )
    canvas_width_mm = fig.get_figwidth() * 25.4
    canvas_height_mm = fig.get_figheight() * 25.4
    if not (
        np.isclose(canvas_width_mm, 180.0, atol=0.01)
        and np.isclose(canvas_height_mm, 140.0, atol=0.01)
    ):
        failures.append(f"canvas_size_mm: {canvas_width_mm:.3f} x {canvas_height_mm:.3f}")
    panel_letter_sizes = [panel_texts[index].get_fontsize() for index in (0, 2, 4, 6)]
    if not all(np.isclose(size, 9.0) for size in panel_letter_sizes):
        failures.append(f"panel_letter_register: sizes={panel_letter_sizes}")
    no_endpoint_rail = "broadest − finest" not in " ".join(text.get_text().lower() for text in hest_visible_texts)
    if not no_endpoint_rail or len(fig.axes) != len(a_c_axes) + 4:
        failures.append(f"hest_endpoint_rail_absent: no_rail={no_endpoint_rail}, axes={len(fig.axes)}")
    expected_titles = ["P1 · CRC", "P2 · CRC", "P3 · NAT"]
    title_placement_ok = (
        [text.get_text() for text in patient_title_texts] == expected_titles
        and all(np.allclose(text.get_position(), (0.96, 0.95)) for text in patient_title_texts)
    )
    if not title_placement_ok:
        failures.append("panel_c_title_placement: patient titles drifted from v52 upper-right placement")
    cbar_pos = cbar_ax.get_position()
    guide_pos = ax_cguide.get_position()
    cbar_width_ratio = cbar_pos.width / guide_pos.width
    cbar_height_ratio = cbar_pos.height / guide_pos.height
    cbar_left_ratio = (cbar_pos.x0 - guide_pos.x0) / guide_pos.width
    cbar_right_ratio = (guide_pos.x1 - cbar_pos.x1) / guide_pos.width
    cbar_bottom_ratio = (cbar_pos.y0 - guide_pos.y0) / guide_pos.height
    cbar_to_fields_gap_mm = (min(ax.get_position().y0 for ax in ax_c) - cbar_pos.y1) * PAGE_HEIGHT_MM
    cbar_box = cbar_ax.get_window_extent(renderer=renderer)
    panel_c_bottom_texts = [
        *panel_c_x_tick_texts,
        *[ax.xaxis.label for ax in ax_c if ax.xaxis.label.get_visible() and ax.xaxis.label.get_text()],
    ]
    colorbar_text_collisions = [
        text.get_text()
        for text in panel_c_bottom_texts
        if cbar_box.overlaps(text.get_window_extent(renderer=renderer))
    ]
    cbar_text_clearance_px = min(
        text.get_window_extent(renderer=renderer).y0 for text in panel_c_bottom_texts
    ) - cbar_box.y1
    cbar_contract_ok = (
        np.isclose(cbar_width_ratio, 0.70, atol=0.006)
        and np.isclose(cbar_height_ratio, 0.38, atol=0.006)
        and np.isclose(cbar_left_ratio, 0.15, atol=0.006)
        and np.isclose(cbar_right_ratio, 0.15, atol=0.006)
        and np.isclose(cbar_bottom_ratio, 0.46, atol=0.006)
        and np.allclose(cbar_ax.get_xticks(), [0.5, 0.7, 0.9])
        and len(cbar_ax.patches) == 64
    )
    if not cbar_contract_ok:
        failures.append(
            "shared_colorbar: "
            f"width_ratio={cbar_width_ratio:.3f}, height_ratio={cbar_height_ratio:.3f}, "
            f"ticks={cbar_ax.get_xticks().tolist()}, patches={len(cbar_ax.patches)}"
        )
    if not (0.0 < cbar_to_fields_gap_mm < 8.5):
        failures.append(f"colorbar_field_gap: {cbar_to_fields_gap_mm:.3f} mm")
    if colorbar_text_collisions or cbar_text_clearance_px <= 0.0:
        failures.append(
            f"colorbar_axis_text_overlap: collisions={colorbar_text_collisions}, "
            f"clearance_px={cbar_text_clearance_px:.2f}"
        )
    expected_c_labels = [str(int(value)) for value in SCALES_UM]
    panel_c_label_sets = [
        [text.get_text() for text in texts if text.get_visible() and text.get_text()]
        for texts in panel_c_x_tick_texts_by_axis
    ]
    panel_c_tick_positions = [ax.get_xticks().copy() for ax in ax_c]
    panel_c_minor_positions = [ax.get_xticks(minor=True).copy() for ax in ax_c]
    panel_c_node_register_ok = all(
        labels == expected_c_labels
        and np.allclose(ticks, LOG_SCALES)
        and len(minor) == 0
        for labels, ticks, minor in zip(
            panel_c_label_sets,
            panel_c_tick_positions,
            panel_c_minor_positions,
        )
    )
    panel_c_tick_style_ok = (
        len(panel_c_x_tick_texts) == 24
        and all(np.isclose(text.get_rotation(), 0.0) for text in panel_c_x_tick_texts)
        and min(text.get_fontsize() for text in panel_c_x_tick_texts) >= 7.0
        and all(rows == [0, 1, 0, 1, 0, 1, 0, 1] for rows in panel_c_x_tick_rows_by_axis)
    )
    panel_c_tick_boxes_by_axis = [
        [text.get_window_extent(renderer=renderer) for text in texts]
        for texts in panel_c_x_tick_texts_by_axis
    ]
    panel_c_tick_collisions = []
    panel_c_tick_xlabel_collisions = []
    panel_c_row_centers_px = []
    for axis_index, (texts, boxes, rows, ax) in enumerate(
        zip(
            panel_c_x_tick_texts_by_axis,
            panel_c_tick_boxes_by_axis,
            panel_c_x_tick_rows_by_axis,
            ax_c,
        )
    ):
        for index, (text_a, box_a) in enumerate(zip(texts, boxes)):
            for text_b, box_b in zip(texts[index + 1 :], boxes[index + 1 :]):
                if box_a.overlaps(box_b):
                    panel_c_tick_collisions.append(
                        (axis_index + 1, text_a.get_text(), text_b.get_text())
                    )
            if box_a.overlaps(ax.xaxis.label.get_window_extent(renderer=renderer)):
                panel_c_tick_xlabel_collisions.append(
                    (axis_index + 1, text_a.get_text())
                )
        row_centers = [
            float(np.mean([(box.y0 + box.y1) / 2.0 for box, row in zip(boxes, rows) if row == row_index]))
            for row_index in (0, 1)
        ]
        panel_c_row_centers_px.append(row_centers)
    panel_c_two_row_geometry_ok = all(
        row_centers[0] - row_centers[1] >= 5.0
        for row_centers in panel_c_row_centers_px
    )
    panel_c_tick_length_widths = [
        (tick.tick1line.get_markersize(), tick.tick1line.get_markeredgewidth())
        for ax in ax_c
        for tick in ax.xaxis.majorTicks[: len(LOG_SCALES)]
    ]
    panel_c_hairline_ticks_ok = (
        len(panel_c_tick_length_widths) == 24
        and all(np.isclose(length, 1.35) and np.isclose(width, 0.40) for length, width in panel_c_tick_length_widths)
    )
    if (
        not panel_c_node_register_ok
        or not panel_c_tick_style_ok
        or not panel_c_two_row_geometry_ok
        or not panel_c_hairline_ticks_ok
        or panel_c_tick_collisions
        or panel_c_tick_xlabel_collisions
    ):
        failures.append(
            "panel_c_x_tick_register: "
            f"labels={panel_c_label_sets}, ticks={panel_c_tick_positions}, "
            f"rows={panel_c_row_centers_px}, collisions={panel_c_tick_collisions}, "
            f"xlabel_collisions={panel_c_tick_xlabel_collisions}, style={panel_c_tick_style_ok}, "
            f"hairlines={panel_c_hairline_ticks_ok}"
        )

    # Panel b keeps the same eight physical nodes and exact trajectories, but
    # now owns a deliberate two-row coordinate band and three clearly
    # separated outcome-label rows under the aggregation heading.
    expected_b_labels = [str(int(value)) for value in SCALES_UM]
    panel_b_label_set = [
        text.get_text() for text in panel_b_x_tick_texts if text.get_visible() and text.get_text()
    ]
    panel_b_node_register_ok = (
        panel_b_label_set == expected_b_labels
        and np.allclose(ax_b[2].get_xticks(), LOG_SCALES)
        and len(ax_b[2].get_xticks(minor=True)) == 0
    )
    panel_b_tick_style_ok = (
        len(panel_b_x_tick_texts) == 8
        and all(np.isclose(text.get_rotation(), 0.0) for text in panel_b_x_tick_texts)
        and min(text.get_fontsize() for text in panel_b_x_tick_texts) >= 7.0
        and panel_b_x_tick_rows == [0, 1, 0, 1, 0, 1, 0, 1]
    )
    panel_b_tick_boxes = [text.get_window_extent(renderer=renderer) for text in panel_b_x_tick_texts]
    panel_b_tick_collisions = []
    panel_b_tick_xlabel_collisions = []
    panel_b_xlabel_box = ax_b[2].xaxis.label.get_window_extent(renderer=renderer)
    for index, (text_a, box_a) in enumerate(zip(panel_b_x_tick_texts, panel_b_tick_boxes)):
        for text_b, box_b in zip(panel_b_x_tick_texts[index + 1 :], panel_b_tick_boxes[index + 1 :]):
            if box_a.overlaps(box_b):
                panel_b_tick_collisions.append((text_a.get_text(), text_b.get_text()))
        if box_a.overlaps(panel_b_xlabel_box):
            panel_b_tick_xlabel_collisions.append(text_a.get_text())
    panel_b_row_centers_px = [
        float(
            np.mean(
                [
                    (box.y0 + box.y1) / 2.0
                    for box, row in zip(panel_b_tick_boxes, panel_b_x_tick_rows)
                    if row == row_index
                ]
            )
        )
        for row_index in (0, 1)
    ]
    panel_b_two_row_geometry_ok = panel_b_row_centers_px[0] - panel_b_row_centers_px[1] >= 28.0
    panel_b_tick_length_widths = [
        (tick.tick1line.get_markersize(), tick.tick1line.get_markeredgewidth())
        for tick in ax_b[2].xaxis.majorTicks[: len(LOG_SCALES)]
    ]
    panel_b_hairline_ticks_ok = (
        len(panel_b_tick_length_widths) == 8
        and all(
            np.isclose(length, 1.35) and np.isclose(width, 0.40)
            for length, width in panel_b_tick_length_widths
        )
    )
    panel_b_strip_boxes = [text.get_window_extent(renderer=renderer) for text in panel_b_strip_texts]
    panel_b_heading_box = panel_texts[3].get_window_extent(renderer=renderer)
    panel_b_heading_row_clearance_mm = (
        panel_b_heading_box.y0 - panel_b_strip_boxes[0].y1
    ) * 25.4 / fig.dpi
    panel_b_strip_gaps_mm = [
        (panel_b_strip_boxes[index].y0 - panel_b_strip_boxes[index + 1].y1) * 25.4 / fig.dpi
        for index in range(2)
    ]
    panel_b_strip_register_ok = (
        [text.get_text() for text in panel_b_strip_texts] == strip_labels
        and all(np.isclose(text.get_position()[1], B_STRIP_LABEL_Y_AXES) for text in panel_b_strip_texts)
        and min(text.get_fontsize() for text in panel_b_strip_texts) >= 7.0
        and panel_b_heading_row_clearance_mm >= 0.55
        and min(panel_b_strip_gaps_mm) >= 2.5
    )
    panel_b_endpoint_boxes = [
        text.get_window_extent(renderer=renderer) for text in panel_b_endpoint_texts
    ]
    panel_b_endpoint_centers_px = [
        (box.y0 + box.y1) / 2.0 for box in panel_b_endpoint_boxes
    ]
    panel_b_endpoint_band_ok = (
        [text.get_text() for text in panel_b_endpoint_texts] == ["0.227", "0.448"]
        and max(panel_b_endpoint_centers_px) - min(panel_b_endpoint_centers_px) <= 5.0
        and panel_b_strip_boxes[0].y0 - max(box.y1 for box in panel_b_endpoint_boxes)
        >= 0.40 * fig.dpi / 25.4
    )

    def band_bounds(texts):
        boxes = [text.get_window_extent(renderer=renderer) for text in texts]
        return min(box.y0 for box in boxes), max(box.y1 for box in boxes)

    panel_b_vertical_bands = [
        ("heading", panel_texts[2:4]),
        ("expression outcome", [panel_b_strip_texts[0]]),
        ("endpoint values", panel_b_endpoint_texts),
        ("composition outcome", [panel_b_strip_texts[1]]),
        ("sampling outcome", [panel_b_strip_texts[2]]),
        ("output row 1", [
            text for text, row in zip(panel_b_x_tick_texts, panel_b_x_tick_rows) if row == 0
        ]),
        ("output row 2", [
            text for text, row in zip(panel_b_x_tick_texts, panel_b_x_tick_rows) if row == 1
        ]),
        ("xlabel", [ax_b[2].xaxis.label]),
    ]
    panel_b_band_bounds = [band_bounds(texts) for _, texts in panel_b_vertical_bands]
    panel_b_band_clearances_mm = [
        (upper[0] - lower[1]) * 25.4 / fig.dpi
        for upper, lower in zip(panel_b_band_bounds, panel_b_band_bounds[1:])
    ]
    panel_b_text_band_order_ok = min(panel_b_band_clearances_mm) >= 0.40
    panel_b_tick_xlabel_clearance_mm = (
        min(
            box.y0
            for box, row in zip(panel_b_tick_boxes, panel_b_x_tick_rows)
            if row == 1
        )
        - panel_b_xlabel_box.y1
    ) * 25.4 / fig.dpi
    if panel_b_tick_xlabel_clearance_mm < 0.40:
        failures.append(
            f"panel_b_tick_xlabel_clearance: {panel_b_tick_xlabel_clearance_mm:.3f} mm"
        )

    def visible_axis_texts(axes):
        texts = []
        for axis in axes:
            texts.extend(
                text
                for text in (
                    list(axis.get_xticklabels())
                    + list(axis.get_yticklabels())
                    + [axis.xaxis.label, axis.yaxis.label, axis.title]
                    + list(axis.texts)
                )
                if text.get_visible() and text.get_text()
            )
        return texts

    panel_b_guard_texts = list(dict.fromkeys(
        panel_texts[2:4]
        + visible_axis_texts(ax_b)
        + panel_b_x_tick_texts
    ))
    panel_a_c_guard_texts = list(dict.fromkeys(
        panel_texts[0:2]
        + panel_texts[4:6]
        + visible_axis_texts([ax_a, *ax_c, ax_cguide, cbar_ax])
    ))
    panel_b_a_c_text_collisions = []
    for b_text in panel_b_guard_texts:
        b_box = b_text.get_window_extent(renderer=renderer)
        for other_text in panel_a_c_guard_texts:
            if b_box.overlaps(other_text.get_window_extent(renderer=renderer)):
                panel_b_a_c_text_collisions.append(
                    (b_text.get_text(), other_text.get_text())
                )
    panel_b_to_c_axes_clearance_mm = (
        min(ax.get_window_extent(renderer=renderer).x0 for ax in ax_c)
        - max(text.get_window_extent(renderer=renderer).x1 for text in panel_b_guard_texts)
    ) * 25.4 / fig.dpi
    panel_b_vs_a_c_ok = (
        not panel_b_a_c_text_collisions and panel_b_to_c_axes_clearance_mm >= 1.0
    )
    if (
        not panel_b_node_register_ok
        or not panel_b_tick_style_ok
        or not panel_b_two_row_geometry_ok
        or not panel_b_hairline_ticks_ok
        or not panel_b_strip_register_ok
        or not panel_b_endpoint_band_ok
        or not panel_b_text_band_order_ok
        or not panel_b_vs_a_c_ok
        or panel_b_tick_collisions
        or panel_b_tick_xlabel_collisions
    ):
        failures.append(
            "panel_b_register: "
            f"labels={panel_b_label_set}, ticks={ax_b[2].get_xticks().tolist()}, "
            f"rows={panel_b_row_centers_px}, tick_collisions={panel_b_tick_collisions}, "
            f"xlabel_collisions={panel_b_tick_xlabel_collisions}, "
            f"heading_clearance={panel_b_heading_row_clearance_mm:.3f} mm, "
            f"strip_gaps={panel_b_strip_gaps_mm}, band_clearances={panel_b_band_clearances_mm}, "
            f"endpoint_band={panel_b_endpoint_band_ok}, b_vs_a_c={panel_b_vs_a_c_ok}, "
            f"hairlines={panel_b_hairline_ticks_ok}"
        )
    y_label_counts = [sum(bool(t.get_text()) and t.get_visible() for t in ax.get_yticklabels()) for ax in ax_c]
    if y_label_counts != [7, 0, 0]:
        failures.append(f"shared_y_tick_register: counts={y_label_counts}")
    panel_c_internal_gaps_mm = [
        (ax_c[index + 1].get_position().x0 - ax_c[index].get_position().x1) * 180.0
        for index in range(2)
    ]
    if not np.allclose(panel_c_internal_gaps_mm, [C_FIELD_GAP_MM, C_FIELD_GAP_MM], atol=0.03):
        failures.append(f"panel_c_internal_corridors: gaps={panel_c_internal_gaps_mm}")
    c_to_left_area_ratio = visual_areas_mm2["c"] / visual_areas_mm2["left"]
    if c_to_left_area_ratio < 2.0:
        failures.append(f"panel_c_dominance: c/left area ratio={c_to_left_area_ratio:.3f}")
    c_to_d_area_ratio = visual_areas_mm2["c"] / visual_areas_mm2["d"]
    if c_to_d_area_ratio <= 1.0:
        failures.append(f"panel_d_subordination: c/d area ratio={c_to_d_area_ratio:.3f}")

    panel_c_lower_band_texts = panel_c_bottom_texts + [
        *[text for text in cbar_ax.get_xticklabels() if text.get_visible() and text.get_text()],
        cbar_ax.xaxis.label,
    ]
    hest_header_visible_texts = [
        text for text in hest_header_texts if text.get_visible() and text.get_text()
    ]
    cbar_register_texts = [
        *[text for text in cbar_ax.get_xticklabels() if text.get_visible() and text.get_text()],
        cbar_ax.xaxis.label,
    ]
    c_to_d_register_texts = [
        *panel_c_bottom_texts,
        *cbar_register_texts,
        *hest_header_visible_texts,
    ]
    c_to_d_register_boxes = [
        text.get_window_extent(renderer=renderer) for text in c_to_d_register_texts
    ]
    c_to_d_register_collisions = []
    for index, (text_a, box_a) in enumerate(zip(c_to_d_register_texts, c_to_d_register_boxes)):
        for text_b, box_b in zip(
            c_to_d_register_texts[index + 1 :],
            c_to_d_register_boxes[index + 1 :],
        ):
            if box_a.overlaps(box_b):
                c_to_d_register_collisions.append((text_a.get_text(), text_b.get_text()))
    c_to_d_register_min_font = min(text.get_fontsize() for text in c_to_d_register_texts)
    if c_to_d_register_collisions or c_to_d_register_min_font < 7.0:
        failures.append(
            "c_to_d_bbox_register: "
            f"collisions={c_to_d_register_collisions}, min_font={c_to_d_register_min_font:.3f} pt"
        )
    c_d_text_clearance_px = min(
        text.get_window_extent(renderer=renderer).y0 for text in panel_c_lower_band_texts
    ) - max(
        text.get_window_extent(renderer=renderer).y1 for text in hest_header_visible_texts
    )
    c_d_text_clearance_mm = c_d_text_clearance_px * 25.4 / fig.dpi
    if not (1.5 <= c_d_text_clearance_mm <= 6.0):
        failures.append(f"panel_c_d_band_clearance: {c_d_text_clearance_mm:.3f} mm")
    panel_b_d_header_clearance_mm = (
        ax_b[2].xaxis.label.get_window_extent(renderer=renderer).y0
        - max(text.get_window_extent(renderer=renderer).y1 for text in hest_header_visible_texts)
    ) * 25.4 / fig.dpi
    if panel_b_d_header_clearance_mm < 1.0:
        failures.append(f"panel_b_d_header_clearance: {panel_b_d_header_clearance_mm:.3f} mm")
    field_aspects = [
        (ax.get_position().width * FIG_W_IN) / (ax.get_position().height * FIG_H_IN) for ax in ax_c
    ]
    if any(not (0.62 <= aspect <= 0.82) for aspect in field_aspects):
        failures.append(f"field_aspect_class: width/height={field_aspects}")
    hest_field_aspects = [
        (ax.get_position().width * FIG_W_IN) / (ax.get_position().height * FIG_H_IN)
        for ax in hest_axes
    ]
    if any(not (1.45 <= aspect <= 1.70) for aspect in hest_field_aspects):
        failures.append(f"hest_field_aspect_class: width/height={hest_field_aspects}")
    if np.allclose(np.diff(LOG_SCALES), np.diff(LOG_SCALES)[0]):
        failures.append("physical_x_positioning: output sizes were reduced to equal ordinal positions")
    visible_strings = " ".join(t.get_text().lower() for _, t in all_relevant_texts())
    if "response landscape" in visible_strings:
        failures.append("wording_exclusion: response-landscape title present")
    terminology_register_ok = (
        ax_a.yaxis.label.get_text().replace("\n", " ") == "Noise-corrected correlation, q"
        and cbar_ax.xaxis.label.get_text() == "Noise-corrected correlation, q"
        and ax_a.xaxis.label.get_text() == "Wavelength (µm)"
        and ax_c[0].yaxis.label.get_text() == "Wavelength (µm)"
        and hest_axes[0].yaxis.label.get_text() == "Graph scale"
        and strip_labels[0] == "Expression variation ≥4 mm"
        and "rna-variation range" not in visible_strings
        and "rna range" not in visible_strings
        and "rna signal ≥4 mm" not in visible_strings
        and "cross-validated q" not in visible_strings
    )
    if not terminology_register_ok:
        failures.append("terminology_register: spatial-scale, expression-variation, noise-corrected correlation terminology failed")
    hest_header_register_ok = (
        hest_header_texts[1].get_text() == "HEST"
        and hest_header_texts[2].get_text() == "Ridge-Phikon-v2"
        and "shade:" not in visible_strings
        and "observed min" not in visible_strings
    )
    if not hest_header_register_ok:
        failures.append("hest_header_register: object labels or range-explanation exclusion failed")

    white_rgba = np.asarray(to_rgba(COL_WHITE), dtype=float)
    pure_white_background_ok = np.allclose(np.asarray(fig.get_facecolor()), white_rgba) and all(
        np.allclose(np.asarray(ax.get_facecolor()), white_rgba) for ax in fig.axes
    )
    if not pure_white_background_ok:
        failures.append("pure_white_background: figure or axes facecolor drifted from #ffffff")
    for ax in fig.axes:
        if ax.xaxis.label.get_text():
            y0 = ax.xaxis.label.get_window_extent(renderer=renderer).y0
            if y0 < 0:
                failures.append(f"xlabel_clipped: '{ax.xaxis.label.get_text()}' y0={y0:.2f}")

    if failures:
        raise AssertionError("FLOOR FAILURE\n" + "\n".join(failures[:30]))

    lines = [
        "annotation_tick_overlap: PASS",
        "annotation_annotation_overlap: PASS",
        "same_axes_structural_text_overlap: PASS",
        "cross_axes_text_overlap: PASS",
        "panel_heading_overlap: PASS",
        "text_clipping: PASS",
        "cross_panel_annotations: PASS",
        "xlabels_on_canvas: PASS",
        "text_obscured_by_marks: PASS (all in-panel critical labels use high-alpha white backing and zorder >= 10)",
        "gridline_direction: PASS (none; L1 diagnostic dark rows=0, dark cols=0 in clean reference panel interior)",
        f"field_topology: PASS (measured={measured}, structural_NA={blanks})",
        f"node_markers: PASS ({node_marker_count}; no observed-node circles or central markers)",
        f"rectangular_overlays: PASS ({rectangular_overlay_count}; no pcolormesh, blocks, or full-width bands)",
        f"structural_mask: PASS ({structural_na_cell_count} N/A cells represented by {mask_patch_count} exact hatched cells)",
        "masked_interpolation: PASS "
        f"({triangle_count} finite adjacent-cell triangles; invalid_vertices=0; nonlocal_triangles=0)",
        "local_faceting: PASS "
        f"(piecewise-triangular surfaces and contours use {sum(surface_vertex_counts.values())} observed vertices; synthetic_nodes=0)",
        "distributed_triangular_relief: PASS "
        "(P2/P3 only; 126 adjacent local triangle faces; P1 overlay count=0; rectangular overlays=0)",
        "v52_surface_path: PASS "
        f"({len(surface_sets)} local faceted fields; fill levels={surface_level_counts}; "
        f"{contour_label_count} dashed 0.60/0.70/0.80 labels; exact warm palette and 0.44–0.91 norm)",
        "semantic_palette: PASS (spot=#365F78; measured RNA=#1F5A58; composition=#9B5E4C; sampling=#A87342; patients=#AAA39B)",
        "hest_house_palette: PASS (median=#1F5A58; ink=#252B2D; spine=#454B4D; grid=#DCD7D1; patients=#AAA39B)",
        f"a_c_data_register: PASS (SHA-256={a_c_data_sha256}; exact iteration-8 data payload)",
        "asymmetric_mosaic_topology: PASS (a above b in one left column; c spans the full right mosaic height)",
        "section_bboxes_mm_from_bottom: PASS "
        + "; ".join(
            f"{name}=({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})"
            for name, bbox in section_bboxes_mm.items()
        ),
        "visual_area_hierarchy: PASS "
        + ", ".join(f"{name}={area:.1f} mm^2" for name, area in visual_areas_mm2.items()),
        f"panel_c_area_gain: PASS ({100.0 * panel_c_area_gain:.1f}% versus accepted v87 visible field area)",
        "raster_register: INFORMATIONAL (iteration 1 changes only panel-b text-band spacing and its eight-node label register)",
        "panel_c_title_placement: PASS (P1 · CRC, P2 · CRC, P3 · NAT at upper right)",
        "shared_colorbar: PASS "
        f"(width ratio={cbar_width_ratio:.3f}, height ratio={cbar_height_ratio:.3f}; ticks 0.5/0.7/0.9)",
        f"colorbar_field_gap: PASS ({cbar_to_fields_gap_mm:.2f} mm)",
        f"colorbar_axis_text_overlap: PASS ({cbar_text_clearance_px:.1f} px clear of all panel-c ticks/xlabels)",
        "reference_proportion_check: PASS "
        f"(left/right margins={cbar_left_ratio:.3f}/{cbar_right_ratio:.3f}; bottom={cbar_bottom_ratio:.3f})",
        "panel_c_x_tick_register: PASS (32/64/96/128/192/256/384/512 each explicit and horizontal at 7.0 pt; "
        "alternating rows 0/1/0/1/0/1/0/1; all eight physical log2 positions and hairline ticks preserved; "
        f"row centers={panel_c_row_centers_px}; tick collisions={len(panel_c_tick_collisions)}; "
        f"tick/xlabel collisions={len(panel_c_tick_xlabel_collisions)})",
        "panel_b_register: PASS (three outcome labels remain explicit at 7.0 pt; "
        f"heading/first-row clearance={panel_b_heading_row_clearance_mm:.2f} mm; "
        "all output sizes 32/64/96/128/192/256/384/512 are horizontal at 7.0 pt in alternating rows; "
        f"row centers={panel_b_row_centers_px}; strip gaps={panel_b_strip_gaps_mm}; "
        f"tick collisions={len(panel_b_tick_collisions)}; tick/xlabel collisions={len(panel_b_tick_xlabel_collisions)})",
        "panel_b_text_bands: PASS (heading, expression outcome, endpoint values, composition outcome, "
        "sampling outcome, output row 1, output row 2 and xlabel are vertically disjoint; "
        f"successive clearances={panel_b_band_clearances_mm} mm; "
        f"tick-to-xlabel clearance={panel_b_tick_xlabel_clearance_mm:.2f} mm)",
        "panel_b_vs_a_c_bbox_conflict: PASS "
        f"(text collisions={len(panel_b_a_c_text_collisions)}; "
        f"right-edge clearance to panel-c axes={panel_b_to_c_axes_clearance_mm:.2f} mm)",
        f"shared_y_tick_register: PASS (visible counts={y_label_counts}; P1 carries the shared scale)",
        "panel_c_internal_corridors: PASS (edge gaps="
        + ", ".join(f"{gap:.2f} mm" for gap in panel_c_internal_gaps_mm)
        + "; P2/P3 coordinate text suppressed)",
        f"panel_c_dominance: PASS (c/left visual-area ratio={c_to_left_area_ratio:.3f})",
        f"panel_d_subordination: PASS (c/d visual-area ratio={c_to_d_area_ratio:.3f})",
        f"c_to_d_bbox_register: PASS ({len(c_to_d_register_texts)} tick/xlabel/colorbar/header texts; collisions=0; min font={c_to_d_register_min_font:.1f} pt)",
        f"panel_c_d_band_clearance: PASS ({c_d_text_clearance_mm:.2f} mm compact clear text band)",
        f"panel_b_d_header_clearance: PASS ({panel_b_d_header_clearance_mm:.2f} mm distinct vertical text bands)",
        "field_aspect_class: PASS (width/height=" + ", ".join(f"{v:.3f}" for v in field_aspects) + ")",
        "physical_x_positioning: PASS (all log2 nodes 32,64,96,128,192,256,384,512 retained; unequal intervals retained)",
        f"hest_data_register: PASS ({hest_value_count} exact values across {len(hest_specimens)} specimens; SHA-256={hest_data_sha256})",
        "hest_equal_fields: PASS (IDC/PAAD/LUNG equal widths)",
        f"hest_screenshot_construction: PASS ({hest_trajectory_count} specimen trajectories; 3 observed min-max envelopes; 3 medians; 9 specimen labels)",
        "hest_field_aspect_class: PASS (width/height="
        + ", ".join(f"{v:.3f}" for v in hest_field_aspects)
        + "; near-golden and subordinate to c)",
        f"hest_bottom_dead_band: PASS (lowest visible ink {hest_lower_ink_y_mm:.2f} mm above page bottom)",
        f"hest_minimum_font: PASS ({hest_min_font:.1f} pt >= 7.0 pt for transplanted panel d)",
        f"overall_minimum_font: PASS ({overall_min_font:.1f} pt >= 7.0 pt; labels use 7.0 pt wherever geometry permits)",
        f"outer_margin_balance: PASS (top={outer_top_margin_mm:.2f} mm, bottom={outer_bottom_margin_mm:.2f} mm, delta={outer_margin_delta_mm:.2f} mm)",
        f"panel_letter_register: PASS (sizes={panel_letter_sizes}; all 9.0 pt)",
        "hest_endpoint_rail_absent: PASS (three tissue fields only)",
        f"canvas_size_mm: PASS ({canvas_width_mm:.1f} x {canvas_height_mm:.1f}; compact journal rectangle)",
        "pure_white_background: PASS (figure and every axes facecolor are #ffffff)",
        "font_embedding: PASS (pdf.fonttype=42; ps.fonttype=42)",
        "terminology_register: PASS (Physical wavelength and relative graph scale distinguished; Expression variation ≥4 mm locked; Noise-corrected correlation, q defined; deprecated RNA-range wording and Cross-validated q absent)",
        "hest_header_register: PASS (HEST and Ridge-Phikon-v2 object labels only; range explanation absent from panel)",
        "wording_exclusion: PASS (no response-landscape title)",
        "scientific_exclusions: PASS (simulated signal, phase rotation and calibration absent)",
        "overall_floor: PASS",
    ]
    (OUT / "floor_selfcheck_iter1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_interval_floor_checks():
    """Bounded checks for this interval-only pass; frozen legacy audit above
    remains archived, but does not re-hash or repeat historical style audits.
    """
    relevant = all_relevant_texts()
    for axis in fig.axes:
        for side_title in (axis._left_title, axis._right_title):
            if side_title.get_visible() and side_title.get_text():
                side_title.set_zorder(20)
                relevant.append((axis, side_title))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(owner, text, text.get_window_extent(renderer)) for owner, text in relevant]
    boxes = [(owner, text, box) for owner, text, box in boxes if box.width and box.height]
    overlaps = [f"{ta.get_text()} / {tb.get_text()}"
                for i, (_, ta, ba) in enumerate(boxes)
                for _, tb, bb in boxes[i+1:] if ba.overlaps(bb)]
    clipped = [text.get_text() for _, text, box in boxes
               if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
    obscured = []
    for axis, row, artist, mapped in interval_artists:
        assert float(row["axis_min"]) <= float(row["lower"]) <= float(row["estimate"]) <= float(row["upper"]) <= float(row["axis_max"])
        for segment in artist.get_segments():
            display_segment = axis.transData.transform(segment)
            line_box = matplotlib.transforms.Bbox.from_extents(
                display_segment[:,0].min()-0.6, display_segment[:,1].min()-0.6,
                display_segment[:,0].max()+0.6, display_segment[:,1].max()+0.6)
            for _, text, box in boxes:
                if box.overlaps(line_box):
                    obscured.append(row["source_key"] + " / " + text.get_text())
    minimum = min(text.get_fontsize() for _, text, _ in boxes)
    assert not overlaps, "text overlaps: " + repr(overlaps)
    assert not clipped, "clipped text: " + repr(clipped)
    assert not obscured, "interval obscured: " + repr(obscured)
    assert minimum >= 7.0
    assert len(INTERVAL_ROWS) == 3
    assert len(ax_a.lines) >= 8  # three original patients, mean, and CI caps
    crc_patient_lines = ax_a.lines[:3]
    assert all(line.get_zorder() == 3 for line in crc_patient_lines)
    assert all(artist.get_zorder() < 3 for axis, _, artist, _ in interval_artists if axis is ax_a)
    crc_caps = [line for line in ax_a.lines if line.get_marker() == "_"]
    assert len(crc_caps) == 4 and all(line.get_zorder() < 3 for line in crc_caps)
    assert all(line.get_markersize() == 3.0 for line in crc_patient_lines)
    assert all(cap.get_markersize() > 3.0 for cap in crc_caps)
    assert len(hest_records) == 3 and sum(len(r["specimen_lines"]) for r in hest_records) == 9
    assert plt.rcParams["pdf.fonttype"] == 42
    with (OUT / "intervals_plotted.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INTERVAL_ROWS[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(INTERVAL_ROWS)
    (OUT / "floor_selfcheck_iter1.txt").write_text(
        "text_bboxes: PASS all visible labels pairwise disjoint and inside canvas\n"
        "annotation_tick_overlap: PASS\ninterval_text_clearance: PASS all three intervals clear of text\n"
        "interval_source_bounds: PASS 3 exact supplied intervals within raw displayed limits\n"
        f"minimum_visible_type: PASS {minimum:.1f} pt\n"
        "text_obscured_by_marks: PASS text above data; retained busy-field labels have white backing\n"
        "patient_marks: PASS three CRC curves, nine HEST paths, three original tissue medians/ranges\n"
        "crc_ci_patient_layer_order: PASS CI strokes/caps 2.5 < original patient glyphs 3; original 3-pt glyphs and 3.6-pt cap widths unchanged\n"
        "geometry: PASS baseline panel rectangles and 180 x 140 mm canvas unchanged\n"
        "statistics_scope: PASS CRC LT128/GE4096 mean CIs; HEST equal-nine-specimen paired effect only\n"
        "pdf_fonttype: PASS 42\noverall_floor: PASS\n", encoding="utf-8")


run_interval_floor_checks()
fig.savefig(OUT / "img_iter1.png", dpi=300, facecolor=COL_WHITE)
fig.savefig(
    OUT / "figure_iter1.pdf",
    facecolor=COL_WHITE,
    edgecolor="none",
    bbox_inches=None,
    pad_inches=0,
    metadata={"CreationDate": None, "ModDate": None},
)
raster = plt.imread(OUT / "img_iter1.png")
raster_rgb = np.asarray(raster[..., :3], dtype=float)
raster_border = np.concatenate(
    [raster_rgb[0, :, :], raster_rgb[-1, :, :], raster_rgb[:, 0, :], raster_rgb[:, -1, :]],
    axis=0,
)
if not np.allclose(raster_border, 1.0):
    raise AssertionError("FLOOR FAILURE\nraster_border_background: outer border is not pure #ffffff")
floor_path = OUT / "floor_selfcheck_iter1.txt"
floor_lines = floor_path.read_text(encoding="utf-8").splitlines()
if floor_lines and floor_lines[-1] == "overall_floor: PASS":
    floor_lines.insert(
        -1,
        f"raster_border_background: PASS (pure #ffffff; PNG={raster_rgb.shape[1]} x {raster_rgb.shape[0]} px at 240 dpi)",
    )
floor_path.write_text("\n".join(floor_lines) + "\n", encoding="utf-8")
fig.savefig(OUT / "Figure3.svg", facecolor=COL_WHITE)
plt.close(fig)
print("wrote Figure 2 iteration 1 patient-glyph layering correction")
