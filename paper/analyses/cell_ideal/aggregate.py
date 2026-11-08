#!/usr/bin/env python3
"""Collect exact requested cell conditions without dropping missing/undefineds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SPECIMENS = ['TENX98','TENX99','TENX95','TENX97','TENX94','TENX96','NCBI784','NCBI785','NCBI783']
VIEWS = ['native_all','native_neoplastic','grid_32um','grid_64um','grid_128um','grid_256um']


def complete_mean(values):
    a=np.asarray(values,dtype=float)
    return float(a.mean()) if np.isfinite(a).all() else float('nan')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=Path(__file__).resolve().parent/'inputs')
    p.add_argument('--mode',choices=['original','amplitude_matched'],default='original')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'output')
    args=p.parse_args()
    frames=[]
    missing=[]
    for specimen in SPECIMENS:
        for view in VIEWS:
            folder=args.results_root/args.mode/specimen/view
            if not (folder/'COMPLETE.json').is_file():
                missing.append(dict(specimen=specimen,view=view,expected=str(folder)))
                continue
            receipt=json.loads((folder/'COMPLETE.json').read_text(encoding='utf8'))
            if receipt['status']!='COMPLETE' or receipt['mode']!=args.mode:
                raise ValueError(f'INVALID_RECEIPT:{folder}')
            frame=pd.read_csv(folder/'SECTION_METRICS.tsv',sep='\t')
            if len(frame)!=12 or frame[['oracle','band']].duplicated().any():
                raise ValueError(f'INCOMPLETE_OR_DUPLICATE_CONDITIONS:{folder}')
            frames.append(frame)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(missing,columns=['specimen','view','expected']).to_csv(args.output/'MISSING_CONDITIONS.tsv',sep='\t',index=False)
    if missing:
        (args.output/'STATUS.json').write_text(json.dumps({'status':'INCOMPLETE','missing_task_count':len(missing),'completed_task_count':len(frames),'mode':args.mode},indent=2)+'\n')
        print(json.dumps({'status':'INCOMPLETE','missing_task_count':len(missing)}))
        return
    section=pd.concat(frames,ignore_index=True)
    section.to_csv(args.output/'ALL_SECTION_METRICS.tsv',sep='\t',index=False)
    keys=['mode','oracle','view','band']
    patient=section.groupby(keys+['patient'],dropna=False).agg(
        C_model=('C_model',complete_mean),V_RNA=('V_RNA',complete_mean),
        V_model=('V_model',complete_mean),sections=('specimen','size'),
        undefined_sections=('q_ideal',lambda s:int((~np.isfinite(s)).sum())),
        section_min=('q_ideal','min'),section_max=('q_ideal','max')).reset_index()
    denominator=np.sqrt(np.maximum(patient.V_RNA,0)*np.maximum(patient.V_model,0))
    patient['q_ideal']=np.where(denominator>1e-12,patient.C_model/denominator,np.nan)
    patient.to_csv(args.output/'PATIENT_METRICS.tsv',sep='\t',index=False)
    summary=patient.groupby(keys,dropna=False).agg(
        q_ideal_equal_patient_mean=('q_ideal',complete_mean),patients=('patient','size'),
        undefined_patients=('q_ideal',lambda s:int((~np.isfinite(s)).sum())),
        patient_min=('q_ideal','min'),patient_max=('q_ideal','max')).reset_index()
    section_range=section.groupby(keys).agg(section_min=('q_ideal','min'),section_max=('q_ideal','max'),
        undefined_sections=('q_ideal',lambda s:int((~np.isfinite(s)).sum()))).reset_index()
    summary=summary.merge(section_range,on=keys,validate='one_to_one')
    summary.to_csv(args.output/'COHORT_SUMMARY.tsv',sep='\t',index=False)
    primary=section[section.oracle.eq('expected_half_log')]
    failed=primary[~np.isfinite(primary.q_ideal)|primary.q_ideal.lt(.95)]
    failed.to_csv(args.output/'FAILED_PRIMARY_CONDITIONS.tsv',sep='\t',index=False)
    result={'status':'COMPLETE','mode':args.mode,'sections':9,'patients':5,
            'required_views_per_section':6,'required_tasks':54,'primary_conditions':len(primary),
            'ideal_gate_pass':failed.empty,'gate_threshold':.95,'failed_conditions':len(failed),
            'patient_aggregation':'equal section Gram moments within patient then q; equal patient q; nonfinite moments propagate'}
    (args.output/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
