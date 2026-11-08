from __future__ import annotations

import base64
import csv
import gzip
import io
import pathlib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


# ============================== DATA SECTOR ==============================
# Four source-exact, native malignant combined B4+B5 interval rows from
# SCALE_MEAN_INTERVALS.tsv (2026-09-13). No inference or new scoring is run.
MEAN_CI_TSV = """model\tsource_band_key\tsource_value_column\tmetric\testimate\tlower\tupper
sCellST\tB4+B5\trna_accuracy\tnoise_corrected_correlation_q\t0.01593064903481916\t0.0076682735692956997\t0.029165563552934461
sCellST\tB4+B5\trna_repeatability\trna_split_half_correlation_r\t0.17141793815765238\t0.12954334711169244\t0.21334077244217933
DeepSpot2Cell\tB4+B5\trna_accuracy\tnoise_corrected_correlation_q\t0.031090483499586097\t0.017076199665951499\t0.04205842891504076
DeepSpot2Cell\tB4+B5\trna_repeatability\trna_split_half_correlation_r\t0.15883888853317571\t0.1171861468333264\t0.20169456314166467
"""
CI_ROWS = list(csv.DictReader(io.StringIO(MEAN_CI_TSV), delimiter="\t"))
CI_LOOKUP = {(row["model"], row["source_value_column"]): row for row in CI_ROWS}
# Exact frozen TSV is inserted below as gzip-compressed base64.
DATA_TSV_GZ_B64 = (
    "H4sIAAAAAAAACq29S48tyZHfuT76KgMF3N5my5G01kIaaEtwugsCIb6mu6iZjz/4WWQVOwv3NvOccxcEyapblZ4R4e5m/5f99fc//+GnP//8+NNf/vmnPz5+"
    "/pff/9P/evzvP/z0/z7++pe//u2Pv//5D3/58+Mvf/v5r3/7+Xf/81/+8M+/+9ufHv/37//8z4//5/E/fvff/uv/+fiXP//+d//61z/+4eff/f5//stPP/2J"
    "f9d//t39b/sfH//959/9619/+qc//OmnP//rf/jr/fN+J4//8tNPf/3vf/3Lz/qff/rjHx//309//sPf/vS7f/rpzz//y0///NifJdp/+9Pj93/840O0r/P4"
    "T/I41+mSqfZMURk//oi8dNxDKo6cPO2Pc434lNcRaz1l9ajrtJTbcZ/OPP4Q6Uuy3dxdpU39oe+tT1nflJV1a/LzXR9yLj9t0efoGevDnyptrTgnWKfIPOSy"
    "iVBV19LTo/6QuXosqyTGvVLs3fXZPr82zZq2mhJ+adGrxNL6RPu0eD/OlZpa5mpuJhnxkMvn1JF0CRs7VQ/VK6TdxyvHzCPeXSBv7lR25oSUn2l/SF4WrVlZ"
    "1TYSj3OFyOjkqEpk8TtcUWodI9b8YamHxpWqNlMV3q2ab6/u//hP/PCTGV0VZX7MQvNRep0MjRotiTjzOJdbneTJaMZkhj/iyux0tVYzr7GHnHO5DH9fnX/V"
    "21/gvT6z7h7lkZzsfERcLGaO+hhfE+uTOdnV2aPaFg+/zqRGerZXhumj/OIbmWzv9jr65Oo08mN1Gvnr/s0jLXFqpjXc4pF2TcRMuWRFJF/p2BnvEgurORmP"
    "uM4xUY8TnScyHyLnEjNvE2vJyX53eWxMORUqFR7uZ1yS7eESqaZdrZ6V7N8RadZiaWVHHnJlZmtmWlYLX5vMZZ3sF53Wox3vLpAnI+fwTZ9MiYnTD/Gr6hjP"
    "8/Dfztutk6fOSR2v6aqHXC0lVnmmMvxos33VTkVmHRXLfHt9u327q1u7RzqP9UP6isoQD7UQKX6JmPaZzGbRExx/fGhu2nLExuOhyeFiI6npKRX19uo+tm+1"
    "ibh7uzjv59Hn0hrbN5yHZ8oJE8mW5iVb9qMuyxPRnNEunf2QM5f6iOkRq+iydxd4ry5OW+vk4VCofKRcdbRPV/oZM/6UTxyLOelWaiKPuKw7JbREtUbt0XqV"
    "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"
    "1E+ONRwoeaeKU5b/h/8fksCfn/zdAAA="
)
df = pd.read_csv(
    io.StringIO(gzip.decompress(base64.b64decode(DATA_TSV_GZ_B64)).decode("utf-8")),
    sep="\t",
)
assert len(df) == 360
assert df[["patient", "model", "view", "band"]].drop_duplicates().shape[0] == 360
assert set(df["patient"]) == {f"patient_{i}" for i in range(1, 6)}
assert set(df["model"]) == {"sCellST", "DeepSpot2Cell"}
# ========================================================================

OUTDIR = pathlib.Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)
PDF_OUT = OUTDIR / "figure_iter0.pdf"
PNG_OUT = OUTDIR / "img_iter0.png"

MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.0,
    "axes.titlesize": 7.4,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#272727"
GRID = "#D3D1CB"
STRIPE = "#F3F3F3"
HEADER = "#FFFFFF"
SCELL = "#315A73"
DEEP = "#9A5A45"
RNA = "#19766D"
NEG = "#AA594A"
MODELS = ["sCellST", "DeepSpot2Cell"]
MCOL = {"sCellST": SCELL, "DeepSpot2Cell": DEEP}
PATIENTS = [f"patient_{i}" for i in range(1, 6)]
PLAB = {p: f"P{i}" for i, p in enumerate(PATIENTS, 1)}
PMARK = ["o", "s", "^", "D", "P"]
VIEWS = ["native_neoplastic", "native_all", "grid_32um", "grid_64um", "grid_128um", "grid_256um"]
VLAB = {
    "native_neoplastic": "Malignant cells",
    "native_all": "All cells",
    "grid_32um": "32 µm",
    "grid_64um": "64 µm",
    "grid_128um": "128 µm",
    "grid_256um": "256 µm",
}
BANDS = ["B1", "B2", "B3", "B4", "B5", "B4+B5"]
BLAB = ["Broadest", "Broad", "Mid", "Fine", "Finest", "Fine + Finest"]
QMIN, QMAX = -0.05, 0.45
NORM = Normalize(QMIN, QMAX, clip=True)
CMAP = {
    "sCellST": LinearSegmentedColormap.from_list("scell", ["#F6F5F1", "#A9BCC5", SCELL]),
    "DeepSpot2Cell": LinearSegmentedColormap.from_list("deep", ["#F6F5F1", "#D4AA97", DEEP]),
}


def letter(fig, x, y, s):
    fig.text(x, y, s, ha="left", va="top", fontsize=10.0, fontweight="bold", color=INK)


def title_bar(ax, x0, x1, y0, h, text, color=HEADER, text_color=INK):
    # Non-data titles use a plain white ground; categorical ink retains model color.
    ax.text((x0 + x1) / 2, y0 + h / 2, text, ha="center", va="center", fontsize=7.0,
            fontweight="bold", color=color if text_color == "white" else text_color)


def q_to_local(q, y0, h):
    return y0 + 0.13 * h + ((q - QMIN) / (QMAX - QMIN)) * 0.72 * h


fig = plt.figure(figsize=(180 * MM, 220 * MM))
drawn_points = []

# ---------------------------------------------------------------------
# a | Benchmark-style score plate, all 360 estimates
# ---------------------------------------------------------------------
letter(fig, 0.014, 0.982, "a")
ax = fig.add_axes([0.075, 0.595, 0.905, 0.355])
ax.set_title("Noise-corrected correlation, q", loc="left", pad=4, fontweight="bold")

cell_w = 1.0
# The attached mean rail is deliberately wider than the band cells.  At final
# 180-mm width this keeps the vertical "Fine + Finest" and "Mean" labels from
# touching without changing the six-band data layout.
mean_w = 0.86
group_gap = 0.45
group_w = 6 * cell_w + mean_w + group_gap
row_h = 1.05
header_h = 0.50
model_gap = 0.62
yrow = {"sCellST": 2.05, "DeepSpot2Cell": 2.05 + row_h + model_gap}

for vi, view in enumerate(VIEWS):
    gx = vi * group_w
    title_bar(ax, gx, gx + 6 * cell_w + mean_w, 0.0, header_h, VLAB[view])
    for bi, label in enumerate(BLAB):
        ax.add_patch(Rectangle((gx + bi + 0.04, 0.58), 0.92, 1.20,
                               facecolor="#FFFFFF", edgecolor="white", linewidth=0.55))
        ax.text(gx + bi + 0.5, 1.18, label, rotation=90, ha="center", va="center",
                fontsize=7.0, color="#4D4B47")
    ax.add_patch(Rectangle((gx + 6.04, 0.58), mean_w - 0.04, 1.20, facecolor="#FFFFFF", edgecolor="white", linewidth=0.55))
    ax.text(gx + 6.04 + (mean_w - 0.04) / 2, 1.18, "Mean", rotation=90,
            ha="center", va="center", fontsize=7.0, color="#4D4B47")

    for model in MODELS:
        yy = yrow[model]
        vals_for_group = []
        for bi, band in enumerate(BANDS):
            cx = gx + bi
            vals = (df[(df.model == model) & (df.view == view) & (df.band == band)]
                    .set_index("patient").loc[PATIENTS, "q"].to_numpy())
            vals_for_group.extend(vals.tolist())
            mean = float(np.mean(vals))
            # restrained cell tint encodes cell mean; five marks remain directly readable
            fc = CMAP[model](0.18 + 0.55 * NORM(mean))
            ax.add_patch(Rectangle((cx + 0.04, yy), 0.92, row_h, facecolor=fc, edgecolor="white", linewidth=0.7))
            zero_y = q_to_local(0, yy, row_h)
            for pi, q in enumerate(vals):
                px = cx + 0.15 + pi * 0.175
                py = q_to_local(float(q), yy, row_h)
                col = MCOL[model] if q >= 0 else NEG
                ax.plot([px, px], [zero_y, py], color=col, lw=0.55, alpha=0.7, solid_capstyle="round")
                point = ax.scatter(px, py, s=8.5, marker=PMARK[pi], facecolor=col, edgecolor="white", linewidth=0.35, zorder=3)
                drawn_points.append(('a', point, px, q_to_local(float(q), yy, row_h)))
            my = q_to_local(mean, yy, row_h)
            ax.plot([cx + 0.09, cx + 0.91], [my, my], color=INK, lw=0.8, zorder=4)

        # attached magnitude rail, one per model × view
        mx = gx + 6.04
        m = float(np.mean(vals_for_group))
        ax.add_patch(Rectangle((mx, yy), mean_w - 0.04, row_h, facecolor="#F3F3F3", edgecolor="white", linewidth=0.7))
        y0 = q_to_local(0, yy, row_h)
        ym = q_to_local(m, yy, row_h)
        ax.add_patch(Rectangle((mx + 0.10, min(y0, ym)), mean_w - 0.24, abs(ym - y0),
                               facecolor=MCOL[model] if m >= 0 else NEG, edgecolor="none"))
        ax.plot([mx + 0.06, mx + mean_w - 0.10], [ym, ym], color=INK, lw=0.75)

for model in MODELS:
    yy = yrow[model]
    ax.text(-8.45, yy + row_h / 2, model, ha="left", va="center", color=MCOL[model],
            fontsize=7.0, fontweight="bold")
    # One true, common row scale for all six views; the original y direction stays.
    ax.plot([-0.60, -0.60], [q_to_local(0, yy, row_h), q_to_local(.4, yy, row_h)],
            color=INK, lw=.5)
    for value in (0.0, 0.2, 0.4):
        yp = q_to_local(value, yy, row_h)
        ax.plot([-0.72, -0.46], [yp, yp], color=INK, lw=.5)
        ax.text(-0.87, yp, f'{value:g}', ha='right', va='center', fontsize=7.0)
        ax.plot([0, len(VIEWS) * group_w - group_gap], [yp, yp],
                color=GRID, lw=.35, zorder=2)

# patient key and compact local scale occupy the same visual rail as the mother figure legends
key_y = max(yrow.values()) + row_h + 0.37
ax.text(-8.70, key_y, "Patient", ha="left", va="center", fontsize=7.0, color="#555555")
for pi, p in enumerate(PATIENTS):
    xx = -1.55 + pi * 2.00
    ax.scatter(xx, key_y, s=13, marker=PMARK[pi], facecolor="white", edgecolor="#64615D", linewidth=0.6)
    ax.text(xx + 0.24, key_y, PLAB[p], ha="left", va="center", fontsize=7.0)

ax.text(22, key_y, "Noise-corrected correlation, q\n(shared scale within rows)", ha="center", va="center",
        fontsize=7.0, color="#555555")

ax.set_xlim(-8.95, len(VIEWS) * group_w - group_gap + 0.05)
ax.set_ylim(key_y + 0.7, -0.12)
ax.axis("off")

# ---------------------------------------------------------------------
# b | Integrated aggregation response plate
# ---------------------------------------------------------------------
letter(fig, 0.014, 0.565, "b")
ax_b = fig.add_axes([0.075, 0.280, 0.47, 0.260])
ax_b.set_title("Aggregation of the same cell predictions", loc="left", pad=8, fontweight="bold")
agg_views = ["native_all", "grid_32um", "grid_64um", "grid_128um", "grid_256um"]
agg_lab = ["Cells", "32", "64", "128", "256"]
cw, rh, mgap = 1.0, 0.78, 0.52
yy = 1.85
for model in MODELS:
    title_bar(ax_b, 0, 5.95, yy - 0.55, 0.46, model, color=MCOL[model], text_color="white")
    for pi, p in enumerate(PATIENTS):
        ry = yy + pi * rh
        if pi % 2:
            ax_b.add_patch(Rectangle((-0.72, ry), 6.47, rh, facecolor=STRIPE, edgecolor="none"))
        ax_b.text(-0.15, ry + rh / 2, PLAB[p], ha="right", va="center", fontsize=7.0)
        vals = []
        for vi, v in enumerate(agg_views):
            q = float(df[(df.model == model) & (df.patient == p) & (df.view == v) & (df.band == "B4+B5")].q.iloc[0])
            vals.append(q)
            fc = CMAP[model](0.15 + 0.68 * NORM(q))
            ax_b.add_patch(Rectangle((vi + 0.04, ry + 0.05), 0.92, rh - 0.10, facecolor=fc, edgecolor="white", linewidth=0.65))
            h = q / 0.35 * (rh - 0.20)
            base = ry + rh - 0.10
            ax_b.plot([vi + 0.5, vi + 0.5], [base, base - h], color=MCOL[model] if q >= 0 else NEG, lw=1.1)
            point = ax_b.scatter(vi + 0.5, base - h, s=11, marker=PMARK[pi], facecolor=MCOL[model] if q >= 0 else NEG,
                         edgecolor="white", linewidth=0.35)
            drawn_points.append(('b', point, vi + .5, base - q / .35 * (rh - .20)))
        m = float(np.mean(vals))
        ax_b.add_patch(Rectangle((5.15, ry + 0.05), 0.76, rh - 0.10, facecolor="#F3F3F3", edgecolor="white", linewidth=0.65))
        mh = m / 0.35 * (rh - 0.20)
        base = ry + rh - 0.10
        ax_b.add_patch(Rectangle((5.40, base - mh), 0.26, mh, facecolor=MCOL[model], edgecolor="none"))
    yy += 5 * rh + mgap + 0.46

for i, lab in enumerate(agg_lab):
    ax_b.add_patch(Rectangle((i + 0.04, 0.28), 0.92, 0.72, facecolor="#FFFFFF", edgecolor="white", linewidth=0.55))
    ax_b.text(i + 0.5, 0.64, lab, ha="center", va="center", fontsize=7.0, fontweight="bold")
ax_b.add_patch(Rectangle((5.15, 0.28), 0.76, 0.72, facecolor="#FFFFFF", edgecolor="white", linewidth=0.55))
ax_b.text(5.53, 0.64, "Mean", ha="center", va="center", fontsize=7.0, fontweight="bold")
ax_b.text(1.7, yy + .55, "Aggregation (µm)", ha="center", va="center", fontsize=7.0)
ax_b.set_xlim(-0.75, 6.03)
ax_b.set_ylim(yy + 1.8, -0.05)
ax_b.axis("off")

# ---------------------------------------------------------------------
# c | Malignant-cell spectrum as a compact companion score plate
# ---------------------------------------------------------------------
letter(fig, 0.565, 0.565, "c")
# Expand 5 mm into the existing b/c gutter; keep c's right edge at 0.980.
ax_c = fig.add_axes([0.62 - 5 / 180, 0.280, 0.36 + 5 / 180, 0.260])
ax_c.set_title("Malignant-cell RNA scales", loc="left", pad=8, fontweight="bold")
yy = 1.85
spec_bands = BANDS[:5]
for model in MODELS:
    title_bar(ax_c, 0, 5.95, yy - 0.55, 0.46, model, color=MCOL[model], text_color="white")
    for pi, p in enumerate(PATIENTS):
        ry = yy + pi * rh
        if pi % 2:
            ax_c.add_patch(Rectangle((-0.72, ry), 6.47, rh, facecolor=STRIPE, edgecolor="none"))
        ax_c.text(-0.15, ry + rh / 2, PLAB[p], ha="right", va="center", fontsize=7.0)
        vals = []
        for bi, b in enumerate(spec_bands):
            q = float(df[(df.model == model) & (df.patient == p) & (df.view == "native_neoplastic") & (df.band == b)].q.iloc[0])
            vals.append(q)
            fc = CMAP[model](0.15 + 0.68 * NORM(q))
            ax_c.add_patch(Rectangle((bi + 0.04, ry + 0.05), 0.92, rh - 0.10, facecolor=fc, edgecolor="white", linewidth=0.65))
            h = q / 0.15 * (rh - 0.20)
            base = ry + rh - 0.10
            ax_c.plot([bi + 0.5, bi + 0.5], [base, base - h], color=MCOL[model] if q >= 0 else NEG, lw=1.0)
            point = ax_c.scatter(bi + 0.5, base - h, s=11, marker=PMARK[pi], facecolor=MCOL[model] if q >= 0 else NEG,
                         edgecolor="white", linewidth=0.35)
            drawn_points.append(('c', point, bi + .5, base - q / .15 * (rh - .20)))
        m = float(np.mean(vals))
        ax_c.add_patch(Rectangle((5.15, ry + 0.05), 0.76, rh - 0.10, facecolor="#F3F3F3", edgecolor="white", linewidth=0.65))
        mh = m / 0.15 * (rh - 0.20)
        base = ry + rh - 0.10
        ax_c.add_patch(Rectangle((5.40, base - mh), 0.26, mh, facecolor=MCOL[model], edgecolor="none"))
    yy += 5 * rh + mgap + 0.46
for i, lab in enumerate(BLAB[:5]):
    ax_c.add_patch(Rectangle((i + 0.04, 0.28), 0.92, .72, facecolor="#FFFFFF", edgecolor="white", linewidth=0.55))
    ax_c.text(i + 0.5, .64, lab, ha="center", va="center", fontsize=7.0, fontweight="bold")
ax_c.add_patch(Rectangle((5.15, 0.28), 0.76, .72, facecolor="#FFFFFF", edgecolor="white", linewidth=0.55))
ax_c.text(5.53, .64, "Mean", ha="center", va="center", fontsize=7.0, fontweight="bold")
ax_c.set_xlim(-0.75, 6.03)
ax_c.set_ylim(yy + 1.8, -0.05)
ax_c.axis("off")

# These are physical-height keys, not colorbars: a full-height stem matches
# exactly one miniature cell's 0-to-upper prediction-correlation displacement.
for axis, upper in ((ax_b, .35), (ax_c, .15)):
    base = yy + .6
    axis.plot([4.1, 4.1], [base, base - (rh - .20)], color=INK, lw=.55)
    for value in (0.0, upper):
        yp = base - value / upper * (rh - .20)
        axis.plot([3.98, 4.22], [yp, yp], color=INK, lw=.55)
        axis.text(4.38, yp, f'{value:g}', ha='left', va='center', fontsize=7.0)
    axis.text(4.40, yy + 1.45, "Noise-corrected correlation, q", ha='center', va='center', fontsize=7.0)
    for model_start in (1.85, 1.85 + 5 * rh + mgap + .46):
        for pi in range(5):
            zero = model_start + pi * rh + rh - .10
            axis.plot([.04, 5.0], [zero, zero], color=GRID, lw=.35, zorder=1.5)

# ---------------------------------------------------------------------
# d | Aligned endpoint rails, separate axes/scales by construction
# ---------------------------------------------------------------------
letter(fig, 0.014, 0.275, "d")
fig.text(0.075, 0.264, "Matched malignant-cell measurements", ha="left", va="bottom", fontsize=7.4, fontweight="bold")
fig.text(0.96, 0.249, "Lines: five-patient range; black ticks: mean; capped bars: 95% mean CI", ha="right", va="center", fontsize=7.0, color=INK)
native = df[(df.view == "native_neoplastic") & (df.band == "B4+B5")]
ci_records = []
endpoint_points = []

def endpoint_plate(pos, metric, xlim, ticks, header, xlabel):
    axp = fig.add_axes(pos)
    title_bar(axp, xlim[0], xlim[1], -0.64, 0.38, header)
    for mi, model in enumerate(MODELS):
        y = mi
        vals = native[native.model == model].set_index("patient").loc[PATIENTS, metric].to_numpy()
        axp.add_patch(Rectangle((xlim[0], y - 0.31), xlim[1] - xlim[0], 0.62,
                                facecolor=STRIPE if mi else "white", edgecolor="none"))
        axp.plot([vals.min(), vals.max()], [y, y], color=MCOL[model], lw=1.2, alpha=0.65)
        for pi, v in enumerate(vals):
            point = axp.scatter(v, y, s=19, marker=PMARK[pi], facecolor="white", edgecolor=MCOL[model], linewidth=0.8, zorder=3)
            endpoint_points.append((point, v, y))
        mean = float(np.mean(vals))
        axp.plot([mean, mean], [y - 0.25, y + 0.25], color=INK, lw=1.5, zorder=4)
        source = CI_LOOKUP[(model, "rna_accuracy" if metric == "q" else "rna_repeatability")]
        estimate, low, high = (float(source[key]) for key in ("estimate", "lower", "upper"))
        assert np.isclose(mean, estimate, rtol=0, atol=1e-14)
        # Separate vertical lane preserves every patient symbol/range line.
        ci_line = axp.hlines(y + 0.17, low, high, color=INK, linewidth=0.85, zorder=2.5)
        axp.vlines([low, high], y + 0.125, y + 0.215,
                   color=INK, linewidth=0.85, zorder=2.5)
        ci_records.append((axp, source, ci_line))
        axp.text(xlim[0] - 0.03 * (xlim[1] - xlim[0]), y, model, ha="right", va="center", fontsize=7.0, color=MCOL[model], fontweight="bold")
    axp.set_xlim(xlim)
    axp.set_ylim(1.55, -0.75)
    axp.set_xticks(ticks)
    axp.set_yticks([])
    axp.set_xlabel(xlabel)
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_visible(False)
    axp.spines["left"].set_visible(False)
    axp.grid(axis="x", color=GRID, lw=0.45)
    return axp

endpoint_plate([0.19, 0.090, 0.30, 0.135], "rna_split_agreement", (0.05, 0.62), [0.1, 0.3, 0.5], "Measured RNA", "RNA split-half correlation")
endpoint_plate([0.66, 0.090, 0.30, 0.135], "q", (-0.035, 0.065), [-0.02, 0, 0.02, 0.04, 0.06], "Predictions", "Noise-corrected correlation, q")

texts = list(fig.texts)
for axis in fig.axes:
    texts += list(axis.texts) + [axis.title, axis._left_title, axis._right_title]
    if axis.axison:
        texts += list(axis.get_xticklabels()) + list(axis.get_yticklabels()) + [axis.xaxis.label, axis.yaxis.label]
    if axis.get_legend():
        texts += list(axis.get_legend().get_texts())
for legend in fig.legends:
    texts += list(legend.get_texts())
texts = [t for t in texts if t.get_visible() and t.get_text()]
assert min(t.get_fontsize() for t in texts) >= 7.0
for text in texts:
    text.set_zorder(20)
fig.set_dpi(300)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
failures = []
c_headers = sorted(
    [t for t in ax_c.texts if t.get_text() in BLAB[:5] + ['Mean']
     and np.isclose(t.get_position()[1], .64)],
    key=lambda t: t.get_position()[0],
)
assert len(c_headers) == 6
c_header_gaps_mm = [
    (right.get_window_extent(renderer).x0 - left.get_window_extent(renderer).x1)
    * 25.4 / fig.dpi
    for left, right in zip(c_headers, c_headers[1:])
]
if min(c_header_gaps_mm) < 1.0:
    failures.append(f'c adjacent-header gap below 1 mm: {c_header_gaps_mm}')
for t in texts:
    bb = t.get_window_extent(renderer)
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
        failures.append('clipped: ' + t.get_text())
for i, t in enumerate(texts):
    for u in texts[i + 1:]:
        if t.get_window_extent(renderer).overlaps(u.get_window_extent(renderer)):
            failures.append(f'text overlap: {t.get_text()} / {u.get_text()}')
assert [sum(tag == p for tag, *_ in drawn_points) for p in ('a', 'b', 'c')] == [360, 50, 50]
for _, point, px, py in drawn_points:
    assert np.allclose(point.get_offsets()[0], [px, py])
assert len(endpoint_points) == 20 and len(ci_records) == 4
for point, value, yy_value in endpoint_points:
    assert np.array_equal(point.get_offsets()[0], [value, yy_value])
ci_export = []
for axis, source, collection in ci_records:
    low, estimate, high = (float(source[key]) for key in ("lower", "estimate", "upper"))
    axis_min, axis_max = axis.get_xlim()
    assert axis_min <= low <= estimate <= high <= axis_max
    segment = collection.get_segments()[0]
    assert np.allclose(segment[:,0], [low, high], rtol=0, atol=1e-15)
    mapped = axis.transData.transform(segment)
    interval_box = mpl.transforms.Bbox.from_extents(mapped[:,0].min()-0.5,
        mapped[:,1].min()-0.5, mapped[:,0].max()+0.5, mapped[:,1].max()+0.5)
    for text in texts:
        if interval_box.overlaps(text.get_window_extent(renderer)):
            failures.append("interval text overlap: " + source["model"] + " / " + text.get_text())
    ci_export.append(dict(
        source_file="SCALE_MEAN_INTERVALS.tsv",
        source_key="|".join(["Breast_Xenium", source["model"], source["metric"], source["source_band_key"]]),
        estimate=source["estimate"], lower=source["lower"], upper=source["upper"],
        displayed_unit_factor="1", panel="ED04d", axis_min=str(axis_min), axis_max=str(axis_max),
        cohort="Breast_Xenium", model=source["model"], metric=source["metric"],
        source_band_key=source["source_band_key"], contrast_id="",
        source_value_column=source["source_value_column"],
        condition="xenium_centred/native_neoplastic/Neoplastic/output_grid_um=0"))
with (OUTDIR / "intervals_plotted.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ci_export[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(ci_export)
fig.savefig(PDF_OUT, dpi=300)
fig.savefig(PNG_OUT, dpi=300)
assert not failures, '\n'.join(failures)
(OUTDIR / 'floor_selfcheck_iter0.txt').write_text(
    'overall: PASS\nall 360 source rows; a=360 b=50 c=50 points\nposition_mapping: PASS exact linear coordinates; b upper scale 0.35; c 0.15\nshared_rulers: a each model 0/0.2/0.4 across six views; b/c literal miniature-stem height keys\n'
    'text_bboxes: disjoint and inside canvas\ntext_obscured_by_marks: PASS all text zorder=20\n'
    'minimum_visible_type: PASS 7 pt\nmetric_labels: PASS separate Prediction correlation, q and RNA split-half correlation\n'
    'patient_range: PASS existing min-to-max lines explicitly labelled five-patient range; not CI\n'
    f'c_header_spacing: PASS minimum_gap={min(c_header_gaps_mm):.3f}mm c_left_shift=5mm right_edge=0.980 font=7pt\n'
    'decorative_headers: white/plain; model identity retained as colored text\n'
    'decorative_row_and_mean_rail_fill: neutral #F3F3F3; all quantitative cell colormaps unchanged\n'
    'native_malignant_mean_ci: PASS 4 supplied B4+B5 intervals; q/RNA split-half independent for each model\n'
    'interval_bounds_and_text_clearance: PASS exact source endpoints within original axes and clear of text\n'
    'endpoint_patient_symbols: PASS all 20 retained at exact original coordinates\n'
    'interval_scope: PASS d only; all-cell and aggregation panels untouched\n'
    'statistical_recalculation: none\npdf_fonttype: 42\n', encoding='utf-8')
plt.close(fig)
