"""Draw the complete 4-condition × 11-model rank uncertainty from real tables."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parent
with (BASE/'inputs/RANK_INTERVALS.tsv').open(encoding='utf-8-sig',newline='') as f:
    source=list(csv.DictReader(f,delimiter='\t'))
reference=json.loads((BASE/'inputs/FIGURE_INTEGRATION_PLAN.json').read_text(encoding='utf-8'))
models=reference['model_display_order']
conditions=['Overall Pearson','Across regions','RNA context + depth matched','Within regions']
keys={(r['condition'],r['method']):r for r in source}
if len(source)!=44 or len(keys)!=44 or set(keys)!={(c,m) for c in conditions for m in models}:
    raise ValueError('The real rank table must contain every one of the 44 fixed entries.')
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
    'font.size':7.5})
COLORS={'EGNv2':'#365F78','DeepSpaCE':'#9B5E4C'}
TEXT='#252B2D';GREY='#85827D';GRID='#E4E0DB'
fig,axes=plt.subplots(1,4,figsize=(180/25.4,100/25.4),sharey=True,dpi=300,facecolor='white')
fig.subplots_adjust(left=.168,right=.990,top=.800,bottom=.178,wspace=.25)
titles=['Overall\nPearson','Across\nregions','RNA context +\ndepth matched','Within\nregions']
plotted=[]
for j,(ax,condition,title) in enumerate(zip(axes,conditions,titles)):
    ax.set_xlim(.5,11.5);ax.set_ylim(10.6,-.6)
    ax.set_xticks([1,3,5,7,9,11]);ax.set_yticks(range(11))
    if j==0:ax.set_yticklabels(models,fontsize=7.5)
    ax.tick_params(axis='both',which='both',length=0,pad=3,colors=TEXT,labelsize=7.0)
    ax.set_axisbelow(True);ax.grid(axis='x',color=GRID,linewidth=.5)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['left','bottom']:ax.spines[side].set_color('#454B4D');ax.spines[side].set_linewidth(.6)
    ax.set_title(title,fontsize=8,fontweight='normal',color=TEXT,pad=9)
    ax.text(-.02,1.12,chr(ord('a')+j),transform=ax.transAxes,ha='left',va='bottom',
            fontsize=9,fontweight='bold',color=TEXT)
    for y,model in enumerate(models):
        r=keys[(condition,model)]
        point=float(r['observed_rank'])
        low=float(r['attainable_interval_low']);high=float(r['attainable_interval_high'])
        if point!=reference['point_ranks'][condition][model]:
            raise ValueError('Rank differs from original point: '+condition+' / '+model)
        if not 1<=low<=high<=11:
            raise ValueError('Invalid supplied rank limits: '+condition+' / '+model)
        color=COLORS.get(model,GREY)
        ax.hlines(y,low,high,color=color,linewidth=1.05,alpha=.78,zorder=2)
        ax.vlines([low,high],y-.08,y+.08,color=color,linewidth=.70,zorder=2)
        ax.scatter([point],[y],s=18 if model in COLORS else 15,color=color,edgecolors='white',linewidths=.45,zorder=3)
        plotted.append({'condition':condition,'method':model,'observed_rank':point,
            'plotted_interval_low':low,'plotted_interval_high':high,
            'interval_source':'attainable_interval_low / attainable_interval_high',
            'colour':color,'source_file':'RANK_INTERVALS.tsv'})
fig.text(.580,.100,'Rank (1 = highest mean score)',ha='center',va='center',fontsize=8,color=TEXT)
legend=[Line2D([0],[0],marker='o',linestyle='none',markerfacecolor=GREY,markeredgecolor='white',markersize=4,label='Observed rank'),
        Line2D([0],[0],color=GREY,linewidth=1.05,label='95% patient-bootstrap rank interval')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.57,.014),ncol=2,
           frameon=False,fontsize=7.0,handlelength=1.6,columnspacing=1.9)

# One plot-specific layout and input-binding validation, after all artists exist.
fig.canvas.draw();renderer=fig.canvas.get_renderer()
texts=[t for t in fig.findobj(match=matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
boxes=[(t,t.get_window_extent(renderer)) for t in texts]
boxes=[(t,b) for t,b in boxes if b.width and b.height]
outside=[t.get_text() for t,b in boxes if b.x0<-.75 or b.y0<-.75 or b.x1>fig.bbox.width+.75 or b.y1>fig.bbox.height+.75]
overlap=[(a.get_text(),b.get_text()) for i,(a,ab) in enumerate(boxes) for b,bb in boxes[i+1:] if ab.overlaps(bb)]
if outside or overlap:raise AssertionError({'outside':outside,'overlap':overlap})
assert len(plotted)==44 and min(t.get_fontsize() for t in texts)>=7
fig.savefig(BASE/'HER2ST_rank_uncertainty.png',dpi=300,facecolor='white')
fig.savefig(BASE/'HER2ST_rank_uncertainty.pdf',facecolor='white',metadata={'CreationDate':None,'ModDate':None})
with (BASE/'RANKS_PLOTTED.tsv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(plotted[0]),delimiter='\t');w.writeheader();w.writerows(plotted)
(BASE/'RANK_FIGURE_CHECK.json').write_text(json.dumps({'status':'PASS','actual_rank_entries':44,
  'conditions':conditions,'patient_count':8,'rank_one_is_best':True,'source_fields':'attainable_interval_low/high',
  'point_estimates_unchanged':True,'minimum_font_pt':min(t.get_fontsize() for t in texts),
  'out_of_canvas_labels':outside,'overlapping_text':overlap,'statistics_recomputed':False},indent=2),encoding='utf-8')
plt.close(fig)
print('Rendered all 44 original ranks with real supplied patient-bootstrap intervals.')
