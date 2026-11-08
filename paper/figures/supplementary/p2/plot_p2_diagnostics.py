"""Replot frozen P2 evidence. No simulations, model fitting, or interval estimation.

First export: python plot_p2_diagnostics.py --run-root /path/to/STRSC-EVIDENCE-CLOSURE-20260918
Offline redraw: python plot_p2_diagnostics.py (reads adjacent source_data CSVs).
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_data'
DATA.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path)
args = parser.parse_args()
sources = {
    'A_split_denominators': 'p2_closure/g_denominator/original_splits/ORIGINAL_PATIENT_SPLIT_DENOMINATORS.tsv',
    'B_native_response': 'p2_closure/native_spectrum/ALL_SPECIMEN_CONDITIONS.tsv',
    'C_macro_paired_response': 'p2_closure/model_transfer/results/HISTORICAL_FAMILY_PAIRED_RESPONSES.tsv',
    'D_specimen_model_response': 'p2_closure/model_transfer/results/PAIRED_SPECIMEN_RESPONSES.tsv',
}
if args.run_root:
    for key, rel in sources.items():
        df = pd.read_csv(args.run_root / rel, sep='\t')
        if key == 'C_macro_paired_response':
            df = df[(df.group == 'ALL_NINE') & (df.band == 'FineMean')]
        elif key == 'D_specimen_model_response':
            df = df[(df.band == 'FineMean') & df.in_historical_macro_family.astype(str).str.lower().eq('true')]
        df.to_csv(DATA / f'{key}.csv', index=False)
dfs = {k: pd.read_csv(DATA / f'{k}.csv') for k in sources}
plt.rcParams.update({'font.family':'Arial', 'font.size':8, 'font.weight':'normal', 'axes.titlesize':10,
    'axes.titleweight':'normal', 'axes.labelweight':'normal',
    'axes.labelsize':8, 'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False})
BLUE, TEAL, GOLD, GREY = '#526F92', '#138A86', '#D68B34', '#8D949E'
fig, axes = plt.subplots(2,2,figsize=(180 / 25.4,7.0))
fig.subplots_adjust(left=.15,right=.99,bottom=.22,top=.92,wspace=.45,hspace=.65)
for label, ax in zip('abcd',axes.flat):
    ax.text(-.16,1.09,label,transform=ax.transAxes,weight='bold',fontsize=12)

ax=axes[0,0]; d=dfs['A_split_denominators']
tasks=['Across regions','Similar cell mix','Within regions']
for i,task in enumerate(tasks):
    sub=d[d.task==task]
    for j,(patient,g) in enumerate(sub.groupby('patient',sort=True)):
        x=i+(j-3.5)*.06
        ax.scatter(np.full(len(g),x),g.D,s=8,c=BLUE,alpha=.28,linewidths=0)
        ax.scatter(x,g.D.mean(),s=20,c=BLUE,edgecolors='white',linewidths=.4,zorder=3)
ax.axhline(0,c=GREY,lw=.8)
ax.set(xticks=range(3),xticklabels=['Across\nregions','RNA context +\ndepth matched','Within\nregions'],
       ylabel='RNA repeatability − majority baseline',title='Split-count denominators',ylim=(-.015,.415))
ax.text(.98,.98,'8 patients × 10 splits / task',transform=ax.transAxes,va='top',ha='right',fontsize=7)
ax.text(.98,.88,f'Minimum = {d.D.min():.4f}',transform=ax.transAxes,va='top',ha='right',fontsize=7)

ax=axes[0,1]; d=dfs['B_native_response']
bands=['B1','B2','B3','B4','B5','FineMean']
for i,band in enumerate(bands):
    sub=d[d.band==band].sort_values(['specimen','rho'])
    x=i+np.linspace(-.22,.22,len(sub))
    ax.scatter(x,sub.error_vs_native_oracle,s=10,c=TEAL,alpha=.62,linewidths=0)
ax.axhline(0,c=GREY,lw=.8)
ax.set(xticks=range(6),xticklabels=['B1','B2','B3','B4','B5','Fine\nmean'],
       ylabel='Pooled q − native scorer oracle',title='Native-spectrum conditional response',ylim=(-.015,.015))
ax.text(.02,.97,'9 specimens × 6 signal levels × 6 outputs',transform=ax.transAxes,va='top',fontsize=7)
ax.text(.02,.87,f'Max. absolute error = {d.error_vs_native_oracle.abs().max():.4f}',transform=ax.transAxes,va='top',fontsize=7)

ax=axes[1,0]; d=dfs['C_macro_paired_response']
families=['local_vs_standard','local_vs_non_neighbour']
for i,family in enumerate(families):
    r=d[d.family==family].iloc[0]
    ax.plot([r.native_oracle_delta,r.q_delta],[i,i],c=GREY,lw=2)
    ax.scatter(r.native_oracle_delta,i-.045,s=35,facecolors='white',edgecolors=BLUE,zorder=3,label='Native scorer oracle' if i==0 else None)
    ax.errorbar(r.q_delta,i+.045,xerr=[[r.q_delta-r.MC_delta_low],[r.MC_delta_high-r.q_delta]],fmt='o',ms=5,c=TEAL,capsize=3,label='Conditional noisy q (95% MC CI)' if i==0 else None)
    ax.text(.00025,i+.25,f'Residual bias +{r.total_error_delta:.6f}',fontsize=7)
ax.axvline(0,c=GREY,lw=.8)
ax.set(yticks=range(2),yticklabels=['Local −\nstandard\n8 models','Local −\nnon-neighbour\n11 models'],
       xlabel='Fine-mean paired difference',title='Positive macro effects; residual bias',xlim=(-.0003,.014),ylim=(-.35,1.55))
ax.invert_yaxis()
handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.14,.105),frameon=False,ncol=2)

ax=axes[1,1]; d=dfs['D_specimen_model_response']
for family,color in zip(families,[BLUE,TEAL]):
    sub=d[d.family==family]
    ax.scatter(sub.native_oracle_delta,sub.q_delta,s=16,c=color,alpha=.64,linewidths=0,label=family)
lo=min(d.native_oracle_delta.min(),d.q_delta.min())-.002
hi=max(d.native_oracle_delta.max(),d.q_delta.max())+.002
ax.plot([lo,hi],[lo,hi],c=GREY,lw=.8)
ax.axhline(0,c=GREY,lw=.5);ax.axvline(0,c=GREY,lw=.5)
flips=d[np.sign(d.native_oracle_delta)!=np.sign(d.q_delta)]
ax.scatter(flips.native_oracle_delta,flips.q_delta,s=45,facecolors='none',edgecolors=GOLD,lw=1,zorder=4)
ax.set(xlabel='Native scorer oracle difference',ylabel='Conditional noisy q difference',
       title='All specimen–model paired effects',xlim=(lo,hi),ylim=(lo,hi))
ax.text(.03,.97,f'{len(d)} pairs; {len(flips)} near-zero sign changes',transform=ax.transAxes,va='top',fontsize=7,
        bbox=dict(facecolor='white',edgecolor='none',pad=1))
ax.text(.03,.88,'Blue: standard;\nteal: non-neighbour baseline',transform=ax.transAxes,va='top',fontsize=7,
        bbox=dict(facecolor='white',edgecolor='none',pad=1))
for ext in ['pdf','svg','png']:
    fig.savefig(ROOT / f'Supplement_P2_Conditional_Diagnostics.{ext}',dpi=220,facecolor='white')
plt.close(fig)
manifest={'status':'RENDERED_FROM_FROZEN_TABLES','recomputed_experiments':False,
          'panels':{k:{'source':v,'rows':len(dfs[k])} for k,v in sources.items()},
          'retained_sign_changes':len(flips),'interval':'Existing 95% conditional Monte Carlo intervals; no new intervals estimated.'}
(ROOT/'FIGURE_DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest,indent=2))
