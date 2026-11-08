"""Current Figure 5: independent histology support plus retained biological tasks.

Run from any directory. All render inputs are staged beside this script.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import PathCollection
from matplotlib.text import Text
import legacy_panels as old

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Arial','font.size':7.2,'axes.titlesize':8.6,
    'axes.titleweight':'regular','axes.labelsize':7.2,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'text.color':old.COL_TEXT,'axes.labelcolor':old.COL_TEXT,'savefig.facecolor':'white'})
W,H=180/25.4,193/25.4

def read(name):
    with (OUT/name).open(encoding='utf-8',newline='') as h:return list(csv.DictReader(h,delimiter='\t'))

def main():
    config=json.loads((OUT/'FIGURE_INPUTS.json').read_text(encoding='utf8'))
    spatial=read('SOURCE_DATA_SPATIAL.tsv')
    section=read('EXTERNAL_ALL_SECTIONS.tsv')
    fig=plt.figure(figsize=(W,H),facecolor='white',dpi=150)
    def axinch(x,y,w,h,**kwargs):return fig.add_axes([x/W,y/H,w/W,h/H],**kwargs)
    def label(x,y,letter,title):
        fig.text(x/W,y/H,letter,fontsize=10,fontweight='bold',ha='left',va='bottom')
        fig.text((x+.19)/W,(y+.007)/H,title,fontsize=8.6,ha='left',va='bottom')
    left,top,right,bottom=config['crop_left_top_right_bottom']
    image=plt.imread(OUT/'KC2_SOURCE_CROP.png')
    extent=(left,right,bottom,top)
    xs=np.array([float(r['source_image_col']) for r in spatial])
    ys=np.array([float(r['source_image_row']) for r in spatial])
    map_axes=[axinch(x,5.33,2.08,2.05) for x in [.26,2.54,4.82]]
    for ax in map_axes:
        ax.imshow(image,extent=extent,zorder=0)
        ax.set_xlim(left,right);ax.set_ylim(bottom,top);ax.axis('off')
    for x,letter,title in zip([.26,2.54,4.82],'abc',['H&E, KC2','Expert annotation','Ordinary prediction']):label(x,7.43,letter,title)
    scale_length=config['scalebar_nominal_um']/config['nominal_um_per_pixel']
    scale_x=left+350;scale_y=bottom-430
    map_axes[0].plot([scale_x,scale_x+scale_length],[scale_y,scale_y],color=old.COL_TEXT,lw=2)
    map_axes[0].text(scale_x,scale_y-150,'1 mm (nominal)',fontsize=7,ha='left',va='bottom',
        bbox=dict(facecolor='white',edgecolor='none',pad=1.0,alpha=.85))

    label_colors={'TLS':old.COL_NEGATIVE,'INFL':'#B58C3F','TUM':'#7294A8','NOR':'#648B77',
                  'UNASSIGNED':'#D5D1CB','':'#EEECE8','LN':'#68546E'}
    # Circle sizes encode position only; the source spot diameter remains in Source Data.
    for annotation,color in label_colors.items():
        mask=np.array([r['manual_anno']==annotation for r in spatial])
        if mask.any():map_axes[1].scatter(xs[mask],ys[mask],s=5 if annotation=='TLS' else 2.6,
            c=color,linewidths=.15 if annotation=='TLS' else 0,edgecolors='white' if annotation=='TLS' else 'none',
            alpha=1.0 if annotation=='TLS' else .9,zorder=3 if annotation=='TLS' else 2,rasterized=True)
    handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=3.5,markerfacecolor=label_colors[a],
             markeredgecolor='none',label=display) for a,display in [('TLS','TLS'),('INFL','INFL'),('TUM','TUM'),('NOR','NOR'),('UNASSIGNED','Unassigned'),('','No label')]]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(2.52/W,5.095/H),ncol=6,frameon=False,
               handlelength=.5,handletextpad=.35,columnspacing=.85,borderpad=0)

    score=np.array([float(r['ordinary_tls']) for r in spatial])
    cmap=LinearSegmentedColormap.from_list('tls_readout',['#F5F1E9','#B9C8C5','#365F78','#183B49'])
    norm=Normalize(vmin=float(score.min()),vmax=float(score.max()))
    colored=map_axes[2].scatter(xs,ys,c=score,cmap=cmap,norm=norm,s=3.2,linewidths=0,zorder=2,rasterized=True)
    cax=axinch(5.08,5.125,1.62,.06)
    cb=fig.colorbar(colored,cax=cax,orientation='horizontal',ticks=[float(score.min()),0.0,float(score.max())])
    cb.ax.xaxis.set_major_formatter(old.FormatStrFormatter('%.3f'))
    cb.ax.tick_params(labelsize=7,length=2,width=.5,pad=2)
    cb.outline.set_linewidth(.5)
    fig.text(5.89/W,5.265/H,"TLS score",fontsize=7,ha='center')

    # All eight source sections, no inferential grouping by unconfirmed patient ID.
    label(.26,4.74,'d','Independent histological TLS')
    ax_d=axinch(.68,3.38,6.19,1.14)
    method_specs=[('measured_RNA','Measured RNA',old.COL_MEASURED,'o',-.16),
                  ('ordinary','Ordinary',old.COL_NEUTRAL,'s',0),
                  ('neighbor','Neighbor',old.COL_DEEPSPACE,'^',.16)]
    for i,row in enumerate(section):
        ax_d.plot([i,i+.16],[float(row['ordinary']),float(row['neighbor'])],color=old.COL_PATIENT,lw=.7,zorder=1)
    legend=[]
    for key,name,color,marker,offset in method_specs:
        values=np.array([float(r[key]) for r in section])
        ax_d.scatter(np.arange(8)+offset,values,c=color,marker=marker,s=17,
                     edgecolors='white',linewidths=.35,zorder=4)
        legend.append(Line2D([0],[0],marker=marker,linestyle='none',markerfacecolor=color,markeredgecolor='white',
            markeredgewidth=.35,markersize=4,label=f'{name} (mean {values.mean():.3f})'))
    ax_d.axhline(.5,color=old.COL_GRID,lw=.7,linestyle=(0,(3,3)),zorder=0)
    ax_d.set_xlim(-.50,7.50);ax_d.set_ylim(.47,1.015)
    ax_d.set_xticks(range(8),[r['sample'] for r in section]);ax_d.set_yticks([.5,.75,1.0])
    ax_d.set_ylabel('AUROC')
    old.style_axis(ax_d,grid=False)
    ax_d.grid(axis='y',color=old.COL_GRID,lw=.45,zorder=0)
    ax_d.text(1.0,1.07,'n = 8 sections',transform=ax_d.transAxes,
              ha='right',va='bottom',fontsize=7)
    fig.legend(handles=legend,loc='center',bbox_to_anchor=(3.75/W,3.03/H),ncol=3,frameon=False,
        handletextpad=.4,columnspacing=1.7,borderpad=0)

    # The retained original functions preserve every point and the original t intervals.
    ax_e=axinch(.89,.60,1.90,1.66)
    ax_f=axinch(3.03,.60,1.90,1.66,sharey=ax_e)
    ax_g=axinch(5.57,.60,1.36,1.66)
    old.draw_tls_field(ax_e,old.TLS_STATE_MEASURED,old.TLS_STATE_MODELS,'EGNv2',
        (-.22,.50),[-.2,0,.2,.4],'Within-region correlation, r',reference=0)
    old.draw_tls_field(ax_f,old.TLS_HOTSPOT_MEASURED,old.TLS_HOTSPOT_MODELS,'DeepSpaCE',
        (.31,.81),[.4,.5,.6,.7,.8],'RNA-defined hotspot AUC',reference=.5)
    ax_f.tick_params(axis='y',labelleft=False)
    for patient,model,x,y in old.CELL_POINTS:
        i=int(patient.split('_')[1])-1
        ax_g.scatter([x],[y],s=23,marker=old.CELL_PATIENT_MARKERS[i],facecolor=old.CELL_COLORS[model],
                     edgecolor='white',linewidth=.45,zorder=5)
    ax_g.axhline(0,color=old.COL_ZERO,lw=.68,linestyle=(0,(2.3,2.3)),zorder=1)
    ax_g.set_xlim(.35,1.48);ax_g.set_ylim(-.5,.4)
    ax_g.set_xticks([.4,.8,1.2,1.4]);ax_g.set_yticks([-.4,0,.4])
    ax_g.set_xlabel('Measured-expression\nhotspot–background contrast',fontsize=7,labelpad=4)
    ax_g.set_ylabel('Measured-RNA contrast\nat model-defined hotspots',fontsize=7,labelpad=4)
    old.style_axis(ax_g,grid=True)
    model_handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=3.8,markerfacecolor=old.CELL_COLORS[m],
                         markeredgecolor='white',markeredgewidth=.4,label=m) for m in ['sCellST','DeepSpot2Cell']]
    ax_g.legend(handles=model_handles,loc='upper left',bbox_to_anchor=(.01,.99),frameon=True,facecolor='white',edgecolor='none',
                framealpha=1,handlelength=.65,handletextpad=.25,borderpad=.16,labelspacing=.2)
    for x,letter,title in [(.49,'e','TLS state'),(2.83,'f','RNA-defined hotspots'),(5.35,'g','Proliferation hotspots')]:label(x,2.38,letter,title)
    patient_handles=[Line2D([0],[0],marker=m,linestyle='none',markersize=3.5,markerfacecolor=old.COL_PATIENT,
                     markeredgecolor='white',markeredgewidth=.35,label=p) for p,m in zip(old.TLS_PATIENTS,old.TLS_PATIENT_MARKERS)]
    fig.text(1.49/W,2.72/H,'Patient',fontsize=7,va='center')
    fig.legend(handles=patient_handles,loc='center left',bbox_to_anchor=(1.85/W,2.72/H),ncol=8,frameon=False,
        handlelength=.7,handletextpad=.17,columnspacing=.55,borderpad=0)
    cell_handles=[Line2D([0],[0],marker=m,linestyle='none',markersize=3.5,markerfacecolor=old.COL_PATIENT,
                   markeredgecolor='white',markeredgewidth=.35,label=str(i+1)) for i,m in enumerate(old.CELL_PATIENT_MARKERS)]
    fig.text(5.48/W,2.72/H,'Patient',fontsize=7,va='center')
    fig.legend(handles=cell_handles,loc='center left',bbox_to_anchor=(5.85/W,2.72/H),ncol=5,frameon=False,
        handlelength=.5,handletextpad=.08,columnspacing=.20,borderpad=0)
    for ax in [ax_d,ax_e,ax_f,ax_g]:old.raise_text_zorders(ax)
    fig.canvas.draw()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    renderer=fig.canvas.get_renderer()
    clipped=[t.get_text() for t in texts if (lambda b:b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5)(t.get_window_extent(renderer))]
    legacy_counts=[sum(len(c.get_offsets()) for c in ax.collections if isinstance(c,PathCollection)) for ax in [ax_e,ax_f,ax_g]]
    receipt=dict(status='RENDERED',figure_number=5,width_mm=180,height_mm=193,
        minimum_font_pt=min(t.get_fontsize() for t in texts),legacy_patient_point_counts=legacy_counts,
        external_section_points=24,spatial_spot_rows=len(spatial),clipped_text=clipped,
        only_bold_text=[t.get_text() for t in texts if t.get_fontweight()=='bold'],
        score_color_limits=[float(score.min()),float(score.max())],negative_scores_preserved=True,
        source_legacy_intervals='unchanged two-sided t intervals from 9/13 statistical package',
        png_dpi=300,pdf_fonttype=42)
    assert legacy_counts==[96,96,10]
    assert receipt['minimum_font_pt']>=7 and not clipped
    fig.savefig(OUT/'Figure5.pdf',facecolor='white')
    fig.savefig(OUT/'Figure5.svg',facecolor='white')
    fig.savefig(OUT/'Figure5.png',dpi=300,facecolor='white')
    plt.close(fig)
    (OUT/'RENDER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
