"""Standalone figures from bundled panel CSVs. Run with Python + matplotlib/numpy.

No project data, training library, scipy, network, or stochastic statistics needed.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_data'
COLORS={'S':'#526F92','D':'#138A86','A':'#D68B34','fixed':'#777777'}
INK='#26313B'; MUTED='#66727C'; LIGHT='#C6CDD2'
plt.rcParams.update({'font.family':'Arial','font.size':7.8,'axes.titlesize':8.6,
    'axes.labelsize':7.8,'xtick.labelsize':7.4,'ytick.labelsize':7.4,'axes.linewidth':.65,
    'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.edgecolor':INK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'savefig.facecolor':'white','figure.facecolor':'white'})


def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def natural(name):
    return (''.join(x for x in name if not x.isdigit()),int(''.join(x for x in name if x.isdigit()) or 0))


def style(ax, letter, title):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3,width=.65,pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.text(-.13,1.075,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
    ax.set_title(title,loc='left',pad=10,fontweight='normal')


def legend(fig, ncol=4):
    items=[Line2D([0],[0],color=COLORS['S'],marker='s',ms=4,lw=1.4,label='S  Standard'),
           Line2D([0],[0],color=COLORS['D'],marker='o',ms=4,lw=1.4,label='D  Local difference'),
           Line2D([0],[0],color=COLORS['A'],marker='^',ms=4.5,lw=1.4,label='A  D + ranking'),
           Line2D([0],[0],color=COLORS['fixed'],marker='D',ms=3.7,lw=1.2,ls='--',label='S  Fixed 1,024 positions')]
    fig.legend(handles=items,ncol=ncol,loc='upper center',bbox_to_anchor=(.52,.93),frameon=False,
               columnspacing=1.4,handlelength=1.8,fontsize=7.8)


def curve(ax, rows, metric, letter, title, ylabel):
    style(ax,letter,title)
    for arm,regime,key,marker in [('S','all_sites','S','s'),('D','all_sites','D','o'),
                                   ('A','all_sites','A','^'),('S','fixed_site_budget','fixed','D')]:
        chosen=sorted([r for r in rows if r['metric']==metric and r['arm']==arm and r['regime']==regime],key=lambda r:int(r['n_train_sources']))
        x=[int(r['n_train_sources']) for r in chosen];y=[float(r['value']) for r in chosen]
        ax.plot(x,y,color=COLORS[key],marker=marker,markersize=4.1,lw=1.35,
                ls='--' if key=='fixed' else '-',mfc='white' if key=='fixed' else COLORS[key],
                markeredgewidth=.8,zorder=2 if key=='fixed' else 3)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_xlim(.82,19.4)
    ax.set_xlabel('Training sources');ax.set_ylabel(ylabel)
    ax.grid(axis='y',color='#E8EBED',lw=.5,zorder=0)


def effect_groups(ax, points, ci, groups, labels, colors, letter,title):
    style(ax,letter,title)
    ax.axvline(0,color='#8A949B',lw=.7,ls=(0,(3,3)),zorder=0)
    for j,(group,label,color) in enumerate(zip(groups,labels,colors)):
        rows=sorted([r for r in points if r['comparison']==group],key=lambda r:natural(r['sample']))
        assert len(rows)==24
        values=np.asarray([float(r['delta']) for r in rows])
        y=j+.15+np.linspace(-.09,.09,24)
        ax.scatter(values,y,s=14,color=color,alpha=.6,edgecolors='white',linewidths=.3,zorder=2)
        r=next(r for r in ci if r['comparison']==group)
        mu,lo,hi=(float(r[k]) for k in ('mean','ci95_low','ci95_high'))
        ax.errorbar(mu,j+.40,xerr=[[mu-lo],[hi-mu]],fmt='o',color=color,ms=5.5,capsize=3,
                    elinewidth=1.7,capthick=1.2,zorder=4)
        ax.text(.0,j-.16,label.replace('\n',' · '),transform=ax.get_yaxis_transform(),
                ha='left',va='bottom',fontsize=7.8,fontweight='normal',color=color)
        ax.text(.99,j-.025,f'{mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]',transform=ax.get_yaxis_transform(),
                ha='right',va='bottom',fontsize=7.7,color=color)
    ax.set_yticks([]);ax.set_ylim(len(groups)-.35,-.4)
    ax.set_xlabel('Change in observed neighbourhood Pearson r')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.xaxis.set_major_locator(MaxNLocator(5))


def paired(ax,rows,letter,title,ylabel):
    style(ax,letter,title)
    subjects=sorted({r['sample'] for r in rows},key=natural)
    lookup={(r['sample'],r['arm']):float(r['value']) for r in rows}
    offsets=np.linspace(-.035,.035,len(subjects))
    for i,s in enumerate(subjects):
        values=[lookup[s,a] for a in ['S','D']]
        ax.plot([0+offsets[i],1+offsets[i]],values,color='#A8B1B8',alpha=.65,lw=.7,zorder=1)
        ax.scatter([0+offsets[i],1+offsets[i]],values,c=[COLORS['S'],COLORS['D']],s=18,
                   edgecolors='white',linewidths=.35,zorder=3)
    ax.set_xticks([0,1],['S\nStandard','D\nLocal difference']);ax.set_xlim(-.25,1.25)
    ax.set_ylabel(ylabel);ax.grid(axis='y',color='#E8EBED',lw=.5,zorder=0)
    ax.text(.99,1.015,'24 patients; each line is a pair',transform=ax.transAxes,ha='right',va='bottom',
            fontsize=7.8,color=MUTED)


def individual_effect(ax,points,ci,letter,title,xlabel):
    style(ax,letter,title)
    values=np.asarray([float(r['delta']) for r in sorted(points,key=lambda r:natural(r['sample']))])
    assert len(values)==24
    ax.axvline(0,color='#8A949B',lw=.75,ls=(0,(3,3)),zorder=0)
    ax.scatter(values,np.linspace(.08,.3,24),s=21,color=COLORS['D'],edgecolors='white',linewidths=.4,zorder=2)
    r=ci[0];mu,lo,hi=(float(r[k]) for k in ['mean','ci95_low','ci95_high'])
    ax.errorbar(mu,.72,xerr=[[mu-lo],[hi-mu]],fmt='o',color=INK,ms=6.5,capsize=4,
                elinewidth=2.0,capthick=1.3,zorder=4)
    ax.set_yticks([.19,.72],['Patients','Mean + 95% CI']);ax.set_ylim(.98,-.23)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.set_xlabel(xlabel);ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.text(.0,1.0,f'{mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]',transform=ax.transAxes,
            fontsize=8.3,fontweight='normal',ha='left',va='top')
    ax.text(.99,.02,f"{int(r['positive'])}/24 positive",transform=ax.transAxes,
            fontsize=8.2,ha='right',va='bottom',color=MUTED)


def footer(fig,text):
    fig.text(.06,.025,text,fontsize=7.1,color=MUTED,ha='left',va='bottom',linespacing=1.5)


def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/(name+'.'+ext),dpi=300)
    plt.close(fig)


def p3():
    fig=plt.figure(figsize=(180/25.4,6.15))
    gs=fig.add_gridspec(2,6,left=.075,right=.98,bottom=.15,top=.79,hspace=.9,wspace=1.4)
    fig.suptitle('ccRCC | Training-source scaling and local objectives',x=.06,y=.985,ha='left',fontsize=11,fontweight='normal')
    fig.text(.06,.941,'Fixed MLP · MSE checkpoint selector · equal-weight means across 24 held-out patients',fontsize=8,color=MUTED)
    legend(fig)
    rows=read('P3_abc_source_curves.csv')
    curve(fig.add_subplot(gs[0,0:2]),rows,'pearson','a','Expression prediction','Observed Pearson r')
    curve(fig.add_subplot(gs[0,2:4]),rows,'neighbour_pearson','b','Neighbourhood prediction','Observed neighbourhood\nPearson r')
    curve(fig.add_subplot(gs[0,4:6]),rows,'mse','c','Expression error','Mean squared error')
    effect_groups(fig.add_subplot(gs[1,0:3]),read('P3_d_source_count_patient_differences.csv'),read('P3_d_source_count_conditional_CI.csv'),
        ['standard_all_sources16_minus_1','standard_fixed1024_sources16_minus_1'],['All positions\n16 − 1 sources','Fixed 1,024 positions\n16 − 1 sources'],
        [COLORS['S'],COLORS['fixed']],'d','More sources, with and without a fixed budget')
    effect_groups(fig.add_subplot(gs[1,3:6]),read('P3_e_objective_patient_differences.csv'),read('P3_e_objective_conditional_CI.csv'),
        ['sources16_D_minus_S','sources16_A_minus_D'],['Local difference\nD − S','Added ranking\nA − D'],
        [COLORS['D'],COLORS['A']],'e','Local objectives at 16 training sources')
    footer(fig,'Dots in d–e: all 24 patient differences. Large points: equal-patient means; bars: existing conditional bootstrap 95% CIs.\nIntervals condition on the fixed out-of-fold predictions and do not include retraining variation. No new P values.')
    save(fig,'P3_training_sources_and_local_objectives')


def p4():
    fig=plt.figure(figsize=(180/25.4,6.45))
    gs=fig.add_gridspec(2,2,left=.12,right=.98,bottom=.15,top=.81,hspace=.72,wspace=.47)
    fig.suptitle('ccRCC | Local training and programme-level use',x=.06,y=.985,ha='left',fontsize=11,fontweight='normal')
    fig.text(.06,.937,'MKI67 · TOP2A · CENPF · PCLAF  |  Fixed Ridge / Phikon-v2  |  Whole-tissue spots',fontsize=8,color=MUTED)
    paired(fig.add_subplot(gs[0,0]),read('P4_ac_programme_paired_points.csv'),'a','Programme prediction','Observed programme Pearson r')
    paired(fig.add_subplot(gs[0,1]),read('P4_bd_hotspot_paired_points.csv'),'b','Model-selected top 10% hotspots','Measured programme enrichment')
    individual_effect(fig.add_subplot(gs[1,0]),read('P4_ac_programme_patient_differences.csv'),read('P4_ac_programme_conditional_CI.csv'),
        'c','Paired programme change','D − S: programme Pearson r')
    individual_effect(fig.add_subplot(gs[1,1]),read('P4_bd_hotspot_patient_differences.csv'),read('P4_bd_hotspot_conditional_CI.csv'),
        'd','Paired hotspot-enrichment change','D − S: programme enrichment')
    footer(fig,'Enrichment: mean measured four-gene programme in model-selected top 10% of spots minus the remaining spots.\nAll 24 patients retained. Existing conditional patient-bootstrap 95% CIs use fixed OOF predictions;\nthey exclude retraining variation.')
    save(fig,'P4_programme_and_hotspot_use')


def supplement():
    fig=plt.figure(figsize=(180/25.4,6.5))
    gs=fig.add_gridspec(2,6,left=.10,right=.98,bottom=.15,top=.79,hspace=.93,wspace=1.6)
    fig.suptitle('ccRCC | Selector sensitivity and neighbour direction',x=.06,y=.985,ha='left',fontsize=11,fontweight='normal')
    fig.text(.06,.941,'Local checkpoint selector in a–c · four-gene programme use in d–e · all 24 patients',fontsize=8,color=MUTED)
    legend(fig)
    rows=read('SUPP_abc_source_curves.csv')
    curve(fig.add_subplot(gs[0,0:2]),rows,'pearson','a','Local selector: expression','Observed Pearson r')
    curve(fig.add_subplot(gs[0,2:4]),rows,'neighbour_pearson','b','Local selector: neighbourhood','Observed neighbourhood\nPearson r')
    curve(fig.add_subplot(gs[0,4:6]),rows,'mse','c','Local selector: error','Mean squared error')
    paired(fig.add_subplot(gs[1,0:3]),read('SUPP_de_neighbour_direction_paired_points.csv'),'d','Programme neighbour direction','Neighbour-direction\nconcordance')
    individual_effect(fig.add_subplot(gs[1,3:6]),read('SUPP_de_neighbour_direction_patient_differences.csv'),read('SUPP_de_neighbour_direction_conditional_CI.csv'),
        'e','Paired direction change: interval crosses zero','D − S: neighbour-direction concordance')
    footer(fig,'All negative and zero patient changes are retained. Exact predicted ties receive half credit; RNA ties are ineligible.\nThe existing conditional patient-bootstrap 95% CI in e conditions on fixed OOF predictions and excludes retraining variation.')
    save(fig,'Supplement_ccRCC_selector_and_neighbour_direction')


if __name__=='__main__':
    p3();p4();supplement()
    print(json.dumps({'status':'THREE_FIGURES_RENDERED','formats':['PDF','SVG','PNG'],'source':'bundled panel CSV files','new_statistics':False}))
