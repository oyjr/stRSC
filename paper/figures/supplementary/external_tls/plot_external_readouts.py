from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
d=pd.read_csv(HERE/'SECTION_METRICS.tsv',sep='\t')
samples=['KC1','KC2','KC3','LC1','LC2','LC3','LC4','LC5']
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
colors=['#85827D','#9B5E4C']
markers=['o','s','^','D','P','v','X','>']
fig,axes=plt.subplots(1,3,figsize=(180/25.4,77/25.4),facecolor='white')
panels=[('TLS_vs_INFL','auroc','TLS vs INFL','AUROC',(.1,1.035),[.2,.4,.6,.8,1.0]),('TLS_vs_explicit_non_TLS','tls_score_spearman','Continuous TLS','Spearman rho',(-.2,.52),[-.2,0,.2,.4]),('TLS_vs_explicit_non_TLS','mean_gene_spearman','Gene expression','Mean gene Spearman rho',(-.015,.155),[0,.05,.1,.15])]
records=[]
for k,(ax,(endpoint,metric,title,ylabel,lims,ticks)) in enumerate(zip(axes,panels)):
    p=d[d.endpoint.eq(endpoint)].pivot(index='sample',columns='arm',values=metric).loc[samples,['ordinary','neighbor']]
    values=p.to_numpy()
    assert np.isfinite(values).all() and values.min()>lims[0] and values.max()<lims[1], (metric, values.min(), values.max(), lims)
    for i,(sample,y) in enumerate(zip(samples,values)):
        ax.plot([0,1],y,color='#C8C8C8',lw=.7,zorder=1)
        for j in range(2):
            ax.scatter(j,y[j],s=17,marker=markers[i],facecolor=colors[j],edgecolor='white',linewidth=.35,zorder=3)
            records.append(dict(panel=chr(97+k),sample=sample,arm=['ordinary','neighbor'][j],metric=metric,value=y[j]))
    for j,mean in enumerate(values.mean(axis=0)):
        ax.plot([j-.11,j+.11],[mean,mean],color='#252B2D',lw=1.8,zorder=4)
    ax.axhline(.5 if metric=='auroc' else 0,color='#B7B7B7',lw=.6,ls='--',zorder=0)
    ax.set(xlim=(-.3,1.3),ylim=lims,yticks=ticks,xticks=[0,1],xticklabels=['Ordinary','Neighbour'],ylabel=ylabel)
    ax.set_title(title,loc='left',pad=8)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,pad=2)
    ax.text(-.27,1.08,chr(97+k),transform=ax.transAxes,fontsize=10,fontweight='bold')
handles=[Line2D([],[],marker=m,linestyle='none',color='#777777',markersize=4,label=s) for m,s in zip(markers,samples)]
fig.legend(handles=handles,loc='lower center',ncol=8,frameon=False,handletextpad=.3,columnspacing=.8,bbox_to_anchor=(.52,.01))
fig.subplots_adjust(left=.105,right=.98,top=.80,bottom=.28,wspace=.63)
for ext in ['pdf','png','svg']:
    fig.savefig(HERE.parent/f'SupplementaryFigure12.{ext}',dpi=300,facecolor='white')
pd.DataFrame(records).to_csv(HERE/'SOURCE_DATA.tsv',sep='\t',index=False)
plt.close(fig)
