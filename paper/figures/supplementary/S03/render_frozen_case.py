"""Render the existing fixed case only; plotting block preserved from build_case.py."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
selection = json.loads((HERE/'data/CASE_SELECTION.json').read_text(encoding='utf-8'))
summary = json.loads((HERE/'data/CASE_CONTRAST_SUMMARY.json').read_text(encoding='utf-8'))
sites = pd.read_csv(HERE/'data/CASE_SPATIAL_VALUES.tsv', sep='\t')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'svg.fonttype': 'none',
                     'axes.titleweight': 'semibold', 'savefig.facecolor': 'white'})
fig, axs = plt.subplots(1, 4, figsize=(13.8, 7.8))
fig.subplots_adjust(left=.028, right=.985, bottom=.11, top=.9, wspace=.08)
keys = ['measured', 'old_standard', 'old_local', 'non_neighbour']
titles = ['Measured RNA', 'Standard training', 'Neighbour supervision', 'Non-neighbour supervision']
vmin, vmax = selection['colour_limits']
for ax, key, title in zip(axs, keys, titles):
    im = ax.scatter(sites.x, sites.y, c=sites[key], cmap='viridis', vmin=vmin, vmax=vmax,
                    s=1.55, linewidths=0, edgecolors='none')
    ax.set_xlim(summary['full_bbox'][0][0] - 450, summary['full_bbox'][1][0] + 450)
    ax.set_ylim(summary['full_bbox'][1][1] + 450, summary['full_bbox'][0][1] - 450)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, pad=11)
    ax.set_axis_off()
cbax = fig.add_axes([.35, .054, .30, .018])
cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=[0, 2, 4, 6, vmax])
cb.ax.set_xticklabels(['0', '2', '4', '6', '7.40'])
cb.set_label('ERBB2 expression: log(1 + count)', fontsize=10, labelpad=4)
cb.outline.set_linewidth(.5)
fig.suptitle('ST-Net | held-out specimen TENX99 | same 20,761 locations', y=.975, fontsize=13)
fig.savefig(ASSETS / 'ERBB2_same_field.png', dpi=160)
fig.savefig(ASSETS / 'ERBB2_same_field.svg')
plt.close(fig)

