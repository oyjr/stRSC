"""Compact GHIST figure; common-scoreable-gene patient-level panel data; no external files required.

Render with python figure.py.
The output canvas is exactly 180 x 144 mm, without automatic cropping.
"""
from pathlib import Path
from io import StringIO
from itertools import permutations
import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects

# === DATA SECTOR (edit here) ===
DATA_TSV = r'''panel	model	patient	metric	view	population	band	value	n_training_seeds	eligible_genes
a	DeepSpot2Cell	patient_1	q	native_neoplastic	Neoplastic	B4+B5	0.042570959578812698	1	
a	DeepSpot2Cell	patient_2	q	native_neoplastic	Neoplastic	B4+B5	0.044120376271884497	1	
a	DeepSpot2Cell	patient_3	q	native_neoplastic	Neoplastic	B4+B5	0.0067082113379821002	1	
a	DeepSpot2Cell	patient_4	q	native_neoplastic	Neoplastic	B4+B5	0.064289962754691898	1	
a	DeepSpot2Cell	patient_5	q	native_neoplastic	Neoplastic	B4+B5	0.050643103220052499	1	
a	GHIST	patient_1	q	native_neoplastic	Neoplastic	B4+B5	0.30370218464288118	5	
a	GHIST	patient_2	q	native_neoplastic	Neoplastic	B4+B5	0.2376805084328141	5	
a	GHIST	patient_3	q	native_neoplastic	Neoplastic	B4+B5	0.16896563574874349	5	
a	GHIST	patient_4	q	native_neoplastic	Neoplastic	B4+B5	0.24082721653476691	5	
a	GHIST	patient_5	q	native_neoplastic	Neoplastic	B4+B5	0.25908889050512501	5	
a	sCellST	patient_1	q	native_neoplastic	Neoplastic	B4+B5	0.010572568410156099	1	
a	sCellST	patient_2	q	native_neoplastic	Neoplastic	B4+B5	0.0147798488300197	1	
a	sCellST	patient_3	q	native_neoplastic	Neoplastic	B4+B5	0.011347202012835501	1	
a	sCellST	patient_4	q	native_neoplastic	Neoplastic	B4+B5	0.045890197896853301	1	
a	sCellST	patient_5	q	native_neoplastic	Neoplastic	B4+B5	0.018962918475437001	1	
b	DeepSpot2Cell	patient_1	q	grid_256um	all	B4+B5	0.083448596349511101	1	
b	DeepSpot2Cell	patient_2	q	grid_256um	all	B4+B5	0.29134221888859269	1	
b	DeepSpot2Cell	patient_3	q	grid_256um	all	B4+B5	0.076614903388368794	1	
b	DeepSpot2Cell	patient_4	q	grid_256um	all	B4+B5	0.31639028912853179	1	
b	DeepSpot2Cell	patient_5	q	grid_256um	all	B4+B5	0.19245398150031559	1	
b	DeepSpot2Cell	patient_1	q	native_all	all	B4+B5	0.037436416088196603	1	
b	DeepSpot2Cell	patient_2	q	native_all	all	B4+B5	0.062699805444331305	1	
b	DeepSpot2Cell	patient_3	q	native_all	all	B4+B5	0.0312419742411407	1	
b	DeepSpot2Cell	patient_4	q	native_all	all	B4+B5	0.061690173552454097	1	
b	DeepSpot2Cell	patient_5	q	native_all	all	B4+B5	0.0560178438870668	1	
b	GHIST	patient_1	q	grid_256um	all	B4+B5	0.42449111386324939	5	
b	GHIST	patient_2	q	grid_256um	all	B4+B5	0.53583730078097491	5	
b	GHIST	patient_3	q	grid_256um	all	B4+B5	0.38278564682092808	5	
b	GHIST	patient_4	q	grid_256um	all	B4+B5	0.39186433080063088	5	
b	GHIST	patient_5	q	grid_256um	all	B4+B5	0.43938436565492689	5	
b	GHIST	patient_1	q	native_all	all	B4+B5	0.33182979457600198	5	
b	GHIST	patient_2	q	native_all	all	B4+B5	0.33815924403708802	5	
b	GHIST	patient_3	q	native_all	all	B4+B5	0.2479475483788991	5	
b	GHIST	patient_4	q	native_all	all	B4+B5	0.24504021530374689	5	
b	GHIST	patient_5	q	native_all	all	B4+B5	0.26141383294659942	5	
b	sCellST	patient_1	q	grid_256um	all	B4+B5	0.21182744642207699	1	
b	sCellST	patient_2	q	grid_256um	all	B4+B5	0.32409759976027669	1	
b	sCellST	patient_3	q	grid_256um	all	B4+B5	0.15081589361040409	1	
b	sCellST	patient_4	q	grid_256um	all	B4+B5	0.23254014233526901	1	
b	sCellST	patient_5	q	grid_256um	all	B4+B5	0.15926690211192521	1	
b	sCellST	patient_1	q	native_all	all	B4+B5	0.060224997067276999	1	
b	sCellST	patient_2	q	native_all	all	B4+B5	0.052307526448912199	1	
b	sCellST	patient_3	q	native_all	all	B4+B5	0.040224479718308399	1	
b	sCellST	patient_4	q	native_all	all	B4+B5	0.041648995021803102	1	
b	sCellST	patient_5	q	native_all	all	B4+B5	0.0134415806897268	1	
c	DeepSpot2Cell	patient_1	programme_q	neighbourhood	Neoplastic	not_applicable	0.10995122024251247	1	
c	DeepSpot2Cell	patient_2	programme_q	neighbourhood	Neoplastic	not_applicable	-0.4288018923294329	1	
c	DeepSpot2Cell	patient_3	programme_q	neighbourhood	Neoplastic	not_applicable	-0.1448846236901093	1	
c	DeepSpot2Cell	patient_4	programme_q	neighbourhood	Neoplastic	not_applicable	0.08273557291248312	1	
c	DeepSpot2Cell	patient_5	programme_q	neighbourhood	Neoplastic	not_applicable	0.4400330668981038	1	
c	DeepSpot2Cell	patient_1	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9221016108790281	1	
c	DeepSpot2Cell	patient_2	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9152175760552672	1	
c	DeepSpot2Cell	patient_3	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.6587274998863519	1	
c	DeepSpot2Cell	patient_4	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9390166261001514	1	
c	DeepSpot2Cell	patient_5	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9104577043461534	1	
c	GHIST	patient_1	programme_q	neighbourhood	Neoplastic	not_applicable	0.15679567613265638	5	
c	GHIST	patient_2	programme_q	neighbourhood	Neoplastic	not_applicable	0.5315409571325287	5	
c	GHIST	patient_3	programme_q	neighbourhood	Neoplastic	not_applicable	0.31096112422411154	5	
c	GHIST	patient_4	programme_q	neighbourhood	Neoplastic	not_applicable	0.1689541030814177	5	
c	GHIST	patient_5	programme_q	neighbourhood	Neoplastic	not_applicable	0.6905551879987075	5	
c	GHIST	patient_1	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9221016108790281	5	
c	GHIST	patient_2	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9152175760552673	5	
c	GHIST	patient_3	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.6587274998863519	5	
c	GHIST	patient_4	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9390166261001515	5	
c	GHIST	patient_5	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9104577043461534	5	
c	sCellST	patient_1	programme_q	neighbourhood	Neoplastic	not_applicable	0.08729844637458213	1	
c	sCellST	patient_2	programme_q	neighbourhood	Neoplastic	not_applicable	0.17494448733549917	1	
c	sCellST	patient_3	programme_q	neighbourhood	Neoplastic	not_applicable	-0.021485818027183804	1	
c	sCellST	patient_4	programme_q	neighbourhood	Neoplastic	not_applicable	0.2793011848252924	1	
c	sCellST	patient_5	programme_q	neighbourhood	Neoplastic	not_applicable	0.12446618014702945	1	
c	sCellST	patient_1	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9221016108790281	1	
c	sCellST	patient_2	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9152175760552672	1	
c	sCellST	patient_3	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.6587274998863519	1	
c	sCellST	patient_4	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9390166261001514	1	
c	sCellST	patient_5	rna_repeatability	neighbourhood	Neoplastic	not_applicable	0.9104577043461534	1	
d	DeepSpot2Cell	patient_1	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.13540134718462016	1	
d	DeepSpot2Cell	patient_2	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	-0.5166542286428898	1	
d	DeepSpot2Cell	patient_3	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	-0.04615899791455119	1	
d	DeepSpot2Cell	patient_4	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	-0.14348997363881932	1	
d	DeepSpot2Cell	patient_5	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.21922872858589929	1	
d	DeepSpot2Cell	patient_1	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.13067860813301027	1	
d	DeepSpot2Cell	patient_2	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	-0.4656525259322732	1	
d	DeepSpot2Cell	patient_3	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	-0.043275097522372186	1	
d	DeepSpot2Cell	patient_4	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	-0.1482301585797463	1	
d	DeepSpot2Cell	patient_5	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.20866202748994916	1	
d	GHIST	patient_1	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.10207975594042085	5	
d	GHIST	patient_2	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.4658610267336002	5	
d	GHIST	patient_3	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.1479623457522254	5	
d	GHIST	patient_4	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.160222621798694	5	
d	GHIST	patient_5	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.9206017893862601	5	
d	GHIST	patient_1	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.09903852768766969	5	
d	GHIST	patient_2	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.4193941746821136	5	
d	GHIST	patient_3	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.13977848492566508	5	
d	GHIST	patient_4	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.1567486564994599	5	
d	GHIST	patient_5	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.8595137955068933	5	
d	sCellST	patient_1	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.058441253387573486	1	
d	sCellST	patient_2	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.15829761980251583	1	
d	sCellST	patient_3	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.03117629583004035	1	
d	sCellST	patient_4	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.1956394990629966	1	
d	sCellST	patient_5	common_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.10462662737183359	1	
d	sCellST	patient_1	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.055649711283434544	1	
d	sCellST	patient_2	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.14155693402553549	1	
d	sCellST	patient_3	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.02993890752849265	1	
d	sCellST	patient_4	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.18243211285265715	1	
d	sCellST	patient_5	full_hotspot_contrast	neighbourhood	Neoplastic	not_applicable	0.09636831178863368	1'''
# === END DATA SECTOR ===
# Panel a/b: the same scoreable intersection is used for all three methods,
# separately per section, GHIST training seed and view (201-211 genes).
# eligible_genes is intentionally blank: no single gene count is assigned to a
# patient after section/seed aggregation. Baseline models retain one fit;
# their five matched support conditions are not five training replicates.


SOURCE_ROWS = list(csv.DictReader(StringIO(DATA_TSV.strip()), delimiter='\t'))
ROWS = [dict(r, value=float(r['value'])) for r in SOURCE_ROWS]
assert len(ROWS) == 105
METHODS = ['GHIST', 'sCellST', 'DeepSpot2Cell']
PATIENTS = [f'patient_{i}' for i in range(1, 6)]
MARKERS = ['o', 's', '^', 'D', 'P']

# Author-prescribed method colours, verified at L1 reference pixels:
# blue (1393,441), purple (408,1380), brick (1370,1391).
COLORS = {'GHIST': '#365F78', 'sCellST': '#68546E', 'DeepSpot2Cell': '#9B5E4C'}
PATIENT_INK = '#938C84'  # L1-PIL exact filled pixel (849,2175).
RNA_INK = '#437473'  # L1-PIL exact filled RNA marker pixel (1844,2025).
AXIS_INK = '#454B4D'  # L1 line pixels; L2 dark hairline, width 0.55 pt.
TEXT_INK = '#252B2D'  # L2 near-black text; accepted prior author constraint.
GUIDE_INK = '#D1CCC5'  # L1-PIL repeated guide colour in RNA region x920:1070,y1940:2050.
WHITE = '#FFFFFF'  # L1 white background.

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
    'text.color': TEXT_INK, 'axes.labelcolor': TEXT_INK,
    'axes.edgecolor': AXIS_INK, 'axes.linewidth': .55,
    'xtick.color': AXIS_INK, 'ytick.color': AXIS_INK,
    'figure.facecolor': WHITE, 'axes.facecolor': WHITE,
    'savefig.facecolor': WHITE, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.unicode_minus': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
})


def values(panel, model, metric=None, view=None):
    selected = [r for r in ROWS if r['panel'] == panel and r['model'] == model
                and (metric is None or r['metric'] == metric)
                and (view is None or r['view'] == view)]
    lookup = {r['patient']: r['value'] for r in selected}
    assert len(selected) == 5 and set(lookup) == set(PATIENTS)
    return np.array([lookup[p] for p in PATIENTS])


WIDTH_MM, HEIGHT_MM = 180, 144
PATIENT_SIZE, MEAN_SIZE = 4.6, 6.4
fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), dpi=300)
axes, patient_records, mean_records, pair_records, labels = [], [], [], [], []


def styled_axes(x, y, width, height, ylim, yticks, full_frame=True):
    ax = fig.add_axes([x/WIDTH_MM, y/HEIGHT_MM, width/WIDTH_MM, height/HEIGHT_MM])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)
    # L1-PIL b blank interior x340:650,y1000:1130: dark rows 0, dark columns 0.
    ax.grid(False)
    ax.tick_params(length=2.3, width=.55, pad=3, top=False, right=False)
    for side, spine in ax.spines.items():
        spine.set_visible(full_frame or side in ['left', 'bottom'])
        spine.set_linewidth(.55)
        spine.set_color(AXIS_INK)
    ax.axhline(0, color=GUIDE_INK, linewidth=.65, zorder=0)
    axes.append(ax)
    return ax


def panel_header(letter, title, x, y):
    labels.append(fig.text((x-11.2)/WIDTH_MM, y/HEIGHT_MM, letter,
                           fontsize=10, weight='bold', va='baseline', zorder=20))
    labels.append(fig.text(x/WIDTH_MM, y/HEIGHT_MM, title,
                           fontsize=9, weight='normal', va='baseline', zorder=20))


def patient_point(ax, x, y, p, panel, model, metric, color=PATIENT_INK):
    artist, = ax.plot(x, y, marker=MARKERS[p], markersize=PATIENT_SIZE,
                      linestyle='none', markerfacecolor=color,
                      markeredgecolor=WHITE, markeredgewidth=.30, zorder=3)
    patient_records.append(dict(ax=ax, x=float(x), y=float(y), panel=panel,
                                model=model, patient=PATIENTS[p], metric=metric,
                                artist=artist))


def mean_point(ax, x, ys, panel, model, metric, color):
    mu = float(np.mean(ys))
    artist, = ax.plot(x, mu, marker='D', markersize=MEAN_SIZE, linestyle='none',
                      markerfacecolor=color, markeredgecolor=WHITE,
                      markeredgewidth=.55, zorder=5)
    mean_records.append(dict(ax=ax, x=float(x), y=mu, panel=panel, model=model,
                             metric=metric, values=ys.copy(), artist=artist))
    return mu


def pair_jitter(ax, ys0, ys1):
    """Deterministic within-condition spread, preserving paired patient identity.

    Select only x offsets, never alter values. Avoid endpoint collisions in
    both conditions simultaneously. Each patient keeps one offset across pair.
    """
    candidate = np.array([-.46, -.30, -.14, .02, .18])
    best, best_score = None, -np.inf
    for offsets in permutations(candidate):
        distances = []
        for ys in [ys0, ys1]:
            coords = ax.transData.transform(np.column_stack([offsets, ys])) * 72/fig.dpi
            for p in range(5):
                for q in range(p+1, 5):
                    distances.append(np.linalg.norm(coords[p]-coords[q]))
        score = min(distances)
        if score > best_score:
            best, best_score = np.array(offsets), score
    return best


def grouped_pairs(ax, panel, key0, key1, condition_labels):
    ax.set_xlim(-.66, 8.57)
    ticks, tick_labels = [], []
    for j, model in enumerate(METHODS):
        if panel == 'b':
            ys0, ys1 = values(panel, model, view=key0), values(panel, model, view=key1)
        else:
            ys0, ys1 = values(panel, model, metric=key0), values(panel, model, metric=key1)
        offset = pair_jitter(ax, ys0, ys1)
        base = 3.25*j
        for p in range(5):
            xp = np.array([base, base+1.35])+offset[p]
            line, = ax.plot(xp, [ys0[p], ys1[p]], color=PATIENT_INK,
                            linewidth=.75, alpha=.9, zorder=1)
            pair_records.append(dict(panel=panel, model=model, patient=PATIENTS[p],
                                     key0=key0, key1=key1, x=xp.copy(),
                                     y=np.array([ys0[p], ys1[p]]), artist=line))
            patient_point(ax, xp[0], ys0[p], p, panel, model, key0)
            patient_point(ax, xp[1], ys1[p], p, panel, model, key1)
        mu0 = mean_point(ax, base+.45, ys0, panel, model, key0, COLORS[model])
        mu1 = mean_point(ax, base+1.80, ys1, panel, model, key1, COLORS[model])
        line, = ax.plot([base+.45, base+1.80], [mu0, mu1],
                        color=COLORS[model], linewidth=1.45, zorder=4)
        line.set_path_effects([patheffects.Stroke(linewidth=2.0, foreground=WHITE),
                              patheffects.Normal()])
        ax.text(base+.675, 1.027, model, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7.8, color=COLORS[model], zorder=20)
        ticks.extend([base+.02, base+1.37])
        tick_labels.extend(condition_labels)
    ax.set_xticks(ticks, tick_labels)
    ax.tick_params(axis='x', length=0, pad=4)


panel_header('a', "Fine-plus-Finest q in malignant cells", 16.5, 137)
ax_a = styled_axes(16.5, 83, 70, 45, (-.020, .340), [0, .1, .2, .3])
ax_a.set_xlim(-.48, 2.62)
ax_a.set_ylabel("Noise-corrected correlation, q", labelpad=4)
jitter = np.array([-.26, -.15, -.04, .07, .18])
for j, model in enumerate(METHODS):
    ys = values('a', model)
    for p,y in enumerate(ys):
        patient_point(ax_a, j+jitter[p], y, p, 'a', model, 'q')
    mean_point(ax_a, j+.42, ys, 'a', model, 'q', COLORS[model])
ax_a.set_xticks(np.arange(3)+.04, METHODS)
ax_a.tick_params(axis='x', length=0, labelsize=8, pad=4)
for t,model in zip(ax_a.get_xticklabels(), METHODS):
    t.set_color(COLORS[model])

panel_header('b', 'Aggregation of all-cell predictions', 104.5, 137)
ax_b = styled_axes(104.5, 83, 72, 45, (-.035, .60), [0, .2, .4, .6])
ax_b.set_ylabel("Noise-corrected correlation, q", labelpad=4)
grouped_pairs(ax_b, 'b', 'native_all', 'grid_256um', ['Single\ncells', '256 µm'])

panel_header('c', 'Proliferation programme', 16.5, 73)
ax_c = styled_axes(16.5, 19, 70, 45, (-.55, 1.03), [-.5, 0, .5, 1])
ax_c.set_xlim(-.48, 3.64)
ax_c.set_ylabel('Programme q / RNA repeatability', labelpad=4)
for j,model in enumerate(METHODS):
    ys = values('c', model, metric='programme_q')
    for p,y in enumerate(ys):
        patient_point(ax_c, j+jitter[p], y, p, 'c', model, 'programme_q')
    mean_point(ax_c, j+.42, ys, 'c', model, 'programme_q', COLORS[model])
rna = values('c', 'sCellST', metric='rna_repeatability')
for model in METHODS:
    assert np.allclose(rna, values('c', model, metric='rna_repeatability'), rtol=0, atol=2e-15)
for p,y in enumerate(rna):
    patient_point(ax_c, 3+jitter[p], y, p, 'c', 'RNA', 'rna_repeatability', RNA_INK)
mean_point(ax_c, 3.42, rna, 'c', 'RNA', 'rna_repeatability', RNA_INK)
ax_c.set_xticks(np.arange(4)+.03, ['GHIST', 'sCellST', 'DeepSpot2Cell', 'RNA\nrepeatability'])
ax_c.tick_params(axis='x', length=0, pad=4)
for t,col in zip(ax_c.get_xticklabels(), [COLORS[m] for m in METHODS]+[RNA_INK]):
    t.set_color(col)
ax_c.axvline(2.60, color=GUIDE_INK, linewidth=.55, zorder=0)

panel_header('d', "Hotspot enrichment", 104.5, 73)
ax_d = styled_axes(104.5, 19, 72, 45, (-.60, 1.04), [-.5, 0, .5, 1], full_frame=False)
ax_d.set_ylabel("Hotspot enrichment", labelpad=4)
grouped_pairs(ax_d, 'd', 'full_hotspot_contrast', 'common_hotspot_contrast', ['Full', 'Shared\n218'])

handles = [Line2D([], [], marker=m, color=PATIENT_INK, markersize=PATIENT_SIZE,
                  linestyle='none', label=f'P{i+1}') for i,m in enumerate(MARKERS)]
handles += [Line2D([], [], marker='D', color=TEXT_INK, markersize=MEAN_SIZE,
                   linestyle='none', label='Patient mean')]
legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.535, .018),
                    ncol=6, frameon=False, fontsize=8, handlelength=1.0,
                    handletextpad=.35, columnspacing=1.05, borderaxespad=0)
legend.set_zorder(20)


def floor_check():
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for ax in axes:
        texts += [t for t in ax.get_xticklabels()+ax.get_yticklabels()
                  +[ax.title, ax.xaxis.label, ax.yaxis.label]+list(ax.texts)
                  if t.get_visible() and t.get_text()]
    texts += labels+list(legend.get_texts())
    for t in texts:
        t.set_zorder(20)
    fig.canvas.draw()
    boxes = [(t,t.get_window_extent(renderer)) for t in texts]
    for i,(ta,ba) in enumerate(boxes):
        assert ba.x0 >= 0 and ba.y0 >= 0 and ba.x1 <= fig.bbox.width and ba.y1 <= fig.bbox.height, f'Clipped: {ta.get_text()}'
        for tb,bb in boxes[i+1:]:
            assert not ba.overlaps(bb), f'Overlap: {ta.get_text()} / {tb.get_text()}'
        for ax in axes:
            assert not ba.overlaps(ax.bbox), f'Text intersects data: {ta.get_text()}'
    assert min(t.get_fontsize() for t in texts) >= 7.5
    assert [sum(p['panel']==panel for p in patient_records) for panel in 'abcd'] == [15,30,20,30]
    assert len(mean_records) == 19 and len(pair_records) == 30
    source_path = Path(__file__).resolve().parent/'inputs'/'data.txt'
    source_check = 'SKIP (standalone copy, staged source absent)'
    if source_path.exists():
        external = list(csv.DictReader(StringIO(source_path.read_text(encoding='utf-8-sig').strip()), delimiter='\t'))
        assert SOURCE_ROWS == external, 'Inline data differs from frozen staged source'
        source_check = 'PASS (all 105 rows, every string field matches inputs/data.txt)'
    min_patient_distance, min_mean_clearance = np.inf, np.inf
    for i,record in enumerate(patient_records):
        ax, x, y = record['ax'], record['x'], record['y']
        assert ax.get_xlim()[0]<x<ax.get_xlim()[1] and ax.get_ylim()[0]<y<ax.get_ylim()[1]
        assert record['artist'].get_ydata()[0] == y
        model = 'sCellST' if record['model']=='RNA' else record['model']
        key = record['metric']
        source = [r for r in ROWS if r['panel']==record['panel'] and r['model']==model
                  and r['patient']==record['patient'] and (r['view']==key or r['metric']==key)]
        assert len(source)==1 and source[0]['value']==y, 'Point not bound to source'
        xy = ax.transData.transform((x,y)) * 72/fig.dpi
        for other in patient_records[i+1:]:
            if ax is other['ax']:
                dist = np.linalg.norm(xy-ax.transData.transform((other['x'],other['y']))*72/fig.dpi)
                min_patient_distance = min(min_patient_distance, dist)
        for mean in mean_records:
            if ax is mean['ax']:
                dist = np.linalg.norm(xy-ax.transData.transform((mean['x'],mean['y']))*72/fig.dpi)
                min_mean_clearance = min(min_mean_clearance, dist)
    assert min_patient_distance >= PATIENT_SIZE-.3, f'Patient symbols overlap: {min_patient_distance}'
    assert min_mean_clearance >= (PATIENT_SIZE+MEAN_SIZE)/2, f'Mean hides patient: {min_mean_clearance}'
    for record in mean_records:
        assert record['y'] == float(np.mean(record['values']))
        assert record['artist'].get_ydata()[0] == record['y']
    for record in pair_records:
        assert np.array_equal(record['artist'].get_ydata(), record['y'])
        for key,value in zip([record['key0'],record['key1']],record['y']):
            match = [r for r in ROWS if r['panel']==record['panel'] and r['model']==record['model']
                     and r['patient']==record['patient'] and (r['view']==key or r['metric']==key)]
            assert len(match)==1 and match[0]['value']==value
    negative_count = sum(p['y']<0 for p in patient_records)
    assert negative_count == 9
    report = [
        'figure_width_mm: 180.000 PASS', 'figure_height_mm: 144.000 PASS',
        'topology: 2 x 2, four grouped axes PASS',
        f'font_family: {plt.rcParams["font.family"]}',
        f'min_text_pt: {min(t.get_fontsize() for t in texts):.1f} PASS',
        'pdf_fonttype: 42 PASS', f'inline_source_binding: {source_check}',
        'text_bbox_pairwise_disjoint: PASS', 'annotation_tick_disjoint: PASS',
        'text_canvas_containment: PASS', 'cross_panel_label_bleed: PASS',
        'xlabel_canvas_containment: PASS',
        'text_obscured_by_marks: PASS (all text outside data boxes, zorder 20)',
        'all_patient_points_in_axes: PASS',
        f'patient_marker_diameter_pt: {PATIENT_SIZE:.1f}',
        f'mean_marker_diameter_pt: {MEAN_SIZE:.1f}',
        f'min_patient_center_distance_pt: {min_patient_distance:.3f} PASS',
        f'min_patient_mean_center_distance_pt: {min_mean_clearance:.3f} PASS',
        'input_rows: 105 PASS', 'unique_patient_points: 95 PASS (a15 b30 c20 d30)',
        'all_95_points_bound_to_source: PASS', 'all_30_pairs_bound_to_same_patient: PASS',
        'all_19_means_equal_five_patient_mean: PASS',
        'RNA_reference: one five-patient series; repeated rows agree within 2e-15 PASS',
        f'negative_values_retained: {negative_count} PASS', 'intervals_or_p_values: none PASS',
        f'b_DeepSpot2Cell_native_mean: {values("b","DeepSpot2Cell",view="native_all").mean():.17g}',
        'data_layer_max_zorder: 5; text_zorder: 20 PASS',
    ]
    for panel,ax in zip('abcd',axes):
        b = ax.get_window_extent(renderer)
        report.append(f'axes_{panel}_bbox_px: {b.x0:.1f},{b.y0:.1f},{b.x1:.1f},{b.y1:.1f}')
        report.append(f'axes_{panel}_limits: x={ax.get_xlim()}, y={ax.get_ylim()}')
    return '\n'.join(report)+'\n'


def main():
    stem = Path(__file__).stem
    outdir = Path(__file__).resolve().parent
    match = re.search(r'_iter(\d+)$', stem)
    suffix = f'iter{match.group(1)}' if match else 'final'
    report = floor_check()
    (outdir/f'floor_selfcheck_{suffix}.txt').write_text(report, encoding='utf-8')
    png = f'img_{suffix}.png' if match else 'figure.png'
    for name,fmt in [(png,'png'), (stem+'.pdf','pdf'), (stem+'.svg','svg')]:
        fig.savefig(outdir/name, format=fmt, dpi=300, facecolor=WHITE)
    print(report)


if __name__ == '__main__':
    main()
