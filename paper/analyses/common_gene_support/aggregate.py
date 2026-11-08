"""Aggregate item2 at the original patient/seed units and map old to new."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPECIMENS = ['TENX98','TENX99','TENX95','TENX97','TENX94','TENX96','NCBI784','NCBI785','NCBI783']
SEEDS = list(range(2026092501,2026092506))


def strict_mean(values):
    a = np.asarray(values,dtype=float)
    return float(a.mean()) if np.isfinite(a).all() else np.nan


def ratio(frame, numerator, left, right, tol=0):
    den = np.sqrt(np.maximum(frame[left],0)*np.maximum(frame[right],0))
    return np.where(den>tol,frame[numerator]/den,np.nan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    p.add_argument('--baseline',type=Path,default=here/'baseline')
    p.add_argument('--input',type=Path,default=here/'inputs')
    p.add_argument('--output',type=Path,default=here/'output')
    args=p.parse_args()
    root=args.input
    output=args.output; output.mkdir(parents=True,exist_ok=True)
    panel,programme,support,missing=[],[],[],[]
    for specimen in SPECIMENS:
        for seed in SEEDS:
            folder=root/'results'/specimen/str(seed)
            if not (folder/'COMPLETE.json').is_file():
                missing.append(dict(specimen=specimen,training_seed=seed)); continue
            receipt=json.loads((folder/'COMPLETE.json').read_text(encoding='utf8'))
            if receipt['status']!='COMPLETE' or receipt['identity']['training_seed']!=seed or receipt['identity']['specimen']!=specimen:
                raise ValueError(f'Invalid completion receipt: {folder}')
            pf=pd.read_csv(folder/'PANEL_SECTION_METRICS.tsv',sep='\t')
            gf=pd.read_csv(folder/'PROGRAMME_SECTION_METRICS.tsv',sep='\t')
            if len(pf)!=72 or len(gf)!=6:
                raise ValueError(f'Incomplete endpoint table: {folder}')
            panel.append(pf); programme.append(gf)
            support.append(pd.read_csv(folder/'GENE_SUPPORT.tsv',sep='\t'))
    pd.DataFrame(missing,columns=['specimen','training_seed']).to_csv(output/'MISSING_TASKS.tsv',sep='\t',index=False)
    if missing:
        (output/'STATUS.json').write_text(json.dumps(dict(status='INCOMPLETE',missing_tasks=len(missing),completed_tasks=45-len(missing)),indent=2)+'\n')
        return
    panel=pd.concat(panel,ignore_index=True); programme=pd.concat(programme,ignore_index=True)
    pd.concat(support,ignore_index=True).to_csv(output/'ALL_GENE_SUPPORT.tsv',sep='\t',index=False)
    panel.to_csv(output/'PANEL_SECTION_METRICS.tsv',sep='\t',index=False)
    programme.to_csv(output/'PROGRAMME_SECTION_METRICS.tsv',sep='\t',index=False)
    keys=['variant','model','training_seed','patient','view','band']
    moments=['A_energy','B_energy','V_RNA','C_model','V_model']
    pp=panel.groupby(keys,as_index=False)[moments].agg(strict_mean)
    pp['q']=ratio(pp,'C_model','V_RNA','V_model',1e-12)
    pp.to_csv(output/'PANEL_PATIENT_BY_SEED.tsv',sep='\t',index=False)
    gp=programme.groupby(['model','training_seed','patient','normalization'],as_index=False)[['q','hotspot_contrast']].agg(strict_mean)
    gp.to_csv(output/'PROGRAMME_PATIENT_BY_SEED.tsv',sep='\t',index=False)
    original=args.baseline
    old_panel=pd.read_csv(original/'PANEL_PATIENT_BY_SEED.tsv',sep='\t')
    old_panel=old_panel[old_panel.support.eq('three_model_common')].copy()
    old_panel['original_seed']=old_panel.training_seed.astype(str)
    pp['original_seed']=np.where(pp.model.eq('GHIST'),pp.training_seed.astype(str),'archived')
    match=['model','original_seed','patient','view','band']
    part=pp.merge(old_panel[match+['q']].rename(columns={'q':'old_value'}),on=match,how='left',validate='many_to_one',indicator=True)
    if not part._merge.eq('both').all(): raise ValueError('Missing original panel endpoint')
    part=part.drop(columns='_merge').rename(columns={'q':'new_value'})
    part['family']='panel'; part['metric']='q'
    impacts=[part[['family','metric','variant','model','training_seed','patient','view','band','old_value','new_value']]]
    old_prog=pd.read_csv(original/'PROGRAMME_PATIENT_BY_SEED.tsv',sep='\t')
    old_prog=old_prog[old_prog.support.eq('three_model_common')].copy()
    old_prog['original_seed']=old_prog.training_seed.astype(str)
    gp['original_seed']=np.where(gp.model.eq('GHIST'),gp.training_seed.astype(str),'archived')
    match=['model','original_seed','patient']
    for normalization,column,metric in [('common218','q','programme_q'),
                                        ('common218','hotspot_contrast','common_hotspot_contrast'),
                                        ('full_canonical_panel','hotspot_contrast','full_hotspot_contrast')]:
        part=gp[gp.normalization.eq(normalization)].merge(old_prog[match+[metric]].rename(columns={metric:'old_value'}),
                    on=match,how='left',validate='many_to_one',indicator=True)
        if not part._merge.eq('both').all(): raise ValueError('Missing original programme endpoint')
        part=part.rename(columns={column:'new_value'})
        part['family']='programme'; part['metric']=metric; part['variant']='unchanged_fixed_programme'
        part['view']='neighbourhood'; part['band']='not_applicable'
        impacts.append(part[['family','metric','variant','model','training_seed','patient','view','band','old_value','new_value']])
    impact=pd.concat(impacts,ignore_index=True)
    impact['absolute_change']=impact.new_value-impact.old_value
    impact['relative_change']=np.where(impact.old_value.ne(0),impact.absolute_change/impact.old_value.abs(),np.nan)
    impact.to_csv(output/'PATIENT_IMPACT_BY_SEED.tsv',sep='\t',index=False)
    keys=['family','metric','variant','model','patient','view','band']
    patients=impact.groupby(keys,as_index=False)[['old_value','new_value']].agg(strict_mean)
    patients.to_csv(output/'PATIENT_IMPACT.tsv',sep='\t',index=False)
    cohort=patients.groupby([k for k in keys if k!='patient'],as_index=False)[['old_value','new_value']].agg(strict_mean)
    cohort['absolute_change']=cohort.new_value-cohort.old_value
    cohort['relative_change']=np.where(cohort.old_value.ne(0),cohort.absolute_change/cohort.old_value.abs(),np.nan)
    cohort['old_sign']=np.sign(cohort.old_value)
    cohort['relative_rule']=np.where(cohort.old_value.eq(0),'undefined_old_zero_report_absolute','delta_div_abs_old')
    cohort['main_endpoint']=cohort.family.eq('programme')|cohort.band.eq('B4+B5')
    cohort['over_5pct']=cohort.relative_change.abs().gt(.05)
    cohort.to_csv(output/'COHORT_OLD_TO_NEW.tsv',sep='\t',index=False)
    rank=cohort[(cohort.variant.eq('intersection')|cohort.family.eq('programme'))&cohort.main_endpoint].copy()
    rk=['family','metric','view','band']
    rank['old_rank']=rank.groupby(rk).old_value.rank(method='min',ascending=False)
    rank['new_rank']=rank.groupby(rk).new_value.rank(method='min',ascending=False)
    rank['undefined_rank']=rank.old_rank.isna()|rank.new_rank.isna()
    rank['rank_changed']=~rank.undefined_rank&rank.old_rank.ne(rank.new_rank)
    rank.to_csv(output/'METHOD_RANKS.tsv',sep='\t',index=False)
    # Programme invariance is checked at section/seed before aggregation can hide errors.
    os=pd.read_csv(original/'PROGRAMME_SPECIMEN_BY_SEED.tsv',sep='\t')
    os=os[os.support.eq('three_model_common')].copy()
    os['original_seed']=os.training_seed.astype(str)
    programme['original_seed']=np.where(programme.model.eq('GHIST'),programme.training_seed.astype(str),'archived')
    match=['model','original_seed','specimen']
    checks=[]
    for normalization,column,metric in [('common218','q','programme_q'),('common218','hotspot_contrast','common_hotspot_contrast'),
                                         ('full_canonical_panel','hotspot_contrast','full_hotspot_contrast')]:
        check=programme[programme.normalization.eq(normalization)].merge(os[match+[metric]].rename(columns={metric:'original'}),
                    on=match,validate='many_to_one')
        check['metric']=metric; check['new']=check[column]; check['absolute_error']=(check['new']-check.original).abs()
        checks.append(check[['model','training_seed','specimen','metric','original','new','absolute_error']])
    reproduction=pd.concat(checks,ignore_index=True)
    reproduction.to_csv(output/'PROGRAMME_INVARIANCE.tsv',sep='\t',index=False)
    constant=panel[['specimen','training_seed','view','model','model_constant_genes','common_prediction_genes']].drop_duplicates()
    constant.to_csv(output/'EXCLUDED_GENE_COUNTS.tsv',sep='\t',index=False)
    programme_pass=bool(np.isfinite(reproduction.absolute_error).all() and reproduction.absolute_error.max()<=1e-6)
    decision_rows=cohort.main_endpoint&(cohort.variant.eq('intersection')|cohort.family.eq('programme'))
    extra_rows=cohort.main_endpoint&cohort.variant.eq('all218_retain_rna')
    changed=bool(cohort.loc[decision_rows,'over_5pct'].any() or rank.rank_changed.any())
    status=dict(status='COMPLETE',tasks=45,sections=9,patients=5,training_seeds=5,
                programme_invariance_pass=programme_pass,max_programme_error=float(reproduction.absolute_error.max()),
                intersection_main_relative_change_over_5pct=bool(cohort.loc[decision_rows,'over_5pct'].any()),
                all218_additional_main_relative_change_over_5pct=bool(cohort.loc[extra_rows,'over_5pct'].any()),
                any_method_rank_change=bool(rank.rank_changed.any()),
                recommendation='use_intersection_results' if changed else 'methods_clarification',
                original_results_modified=False)
    (output/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))


if __name__=='__main__': main()
