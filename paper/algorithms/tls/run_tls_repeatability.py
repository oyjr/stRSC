from pathlib import Path
import argparse
import concurrent.futures
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[1]
DEFAULT_DATA = None
SAMPLES = ['KC1', 'KC2', 'KC3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5']
ARMS = ['ordinary', 'neighbor']
SPLITS = 20
BASE_SEED = 20260930


def original_module():
    path = RUN / 'source/tls_readout.py'
    spec = importlib.util.spec_from_file_location('tls_readout_original', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corr(x, y, ranked=False):
    if ranked:
        x, y = rankdata(x), rankdata(y)
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    den = np.sqrt(np.mean(x*x) * np.mean(y*y))
    return float(np.mean(x*y)/den) if den > 0 else float('nan')


def one_section(index, data_root):
    start = time.monotonic()
    sample = SAMPLES[index]
    data_root = Path(data_root)
    with threadpool_limits(limits=1):
        original = original_module()
        with np.load(data_root / f'inputs/derived/{sample}/COUNTS_COMMON.npz', allow_pickle=False) as z:
            counts = z['counts']
            barcodes = z['barcodes'].astype(str)
            genes = tuple(z['gene_canonical'].astype(str))
        fixed = pd.read_csv(data_root / f'predicted_tls/folds/{sample}/SPOT_READOUTS.tsv.gz', sep='\t')
        assert not fixed.barcode.duplicated().any() and len(set(barcodes)) == len(barcodes)
        ix_barcode = pd.Index(barcodes).get_indexer(fixed.barcode.astype(str))
        assert (ix_barcode >= 0).all(), f'{sample}:barcode'
        counts = counts[ix_barcode]
        assert np.issubdtype(counts.dtype, np.integer) and counts.min() >= 0
        assert counts.shape == (len(fixed), 748) and (counts.sum(1) > 0).all()
        coefficients = original.load_coefficients(RUN / 'source/coef-full.tsv.gz')
        indices, weights, intercept, overlap = original.coefficient_binding(genes, coefficients)
        assert len(indices) == 747
        binding = pd.read_csv(data_root / 'measured_tls/GENE_COEFFICIENT_BINDING.tsv', sep='\t')
        binding = binding[binding['sample'].eq(sample)].set_index('gene_canonical').loc[list(genes)]
        coefficient_error = float(np.max(np.abs(binding.loc[overlap, 'coefficient'].to_numpy()-weights)))
        assert coefficient_error < 1e-12, f'{sample}:coefficient-mismatch:{coefficient_error}'
        full_measured = original.official_tls_transform(counts, indices, weights, intercept)
        error = float(np.max(np.abs(full_measured-fixed.measured_tls.to_numpy())))
        assert error < 1e-9, f'{sample}:full-measured-mismatch:{error}'
        section_old = pd.read_csv(data_root / f'predicted_tls/folds/{sample}/METRICS.tsv', sep='\t')
        checks = dict(sample=sample, spots=len(fixed), input_genes=748, weighted_genes=747,
                      counts_dtype=str(counts.dtype), min_panel_total=int(counts.sum(1).min()),
                      full_measured_max_abs_difference=error, coefficient_tsv_roundtrip_max_abs_difference=coefficient_error,
                      transform_source='source/tls_readout.py:official_tls_transform',
                      score_support='original fixed prediction barcodes, all retained')
        p = {a: fixed[f'{a}_tls'].to_numpy(dtype=float) for a in ARMS}
        pc = {a: p[a]-p[a].mean() for a in ARMS}
        vp = {a: float(np.mean(pc[a]**2)) for a in ARMS}
        for a in ARMS:
            observed = corr(p[a], full_measured, True)
            expected = float(section_old[section_old.arm.eq(a)].iloc[0].tls_score_spearman)
            assert abs(observed-expected) < 1e-12, f'{sample}:{a}:spearman-mismatch'
            checks[f'{a}_full_spearman_difference'] = observed-expected
        rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, index]))
        moments, halves = [], []
        for split in range(SPLITS):
            na = rng.binomial(counts, .5)
            nb = counts-na
            a = original.official_tls_transform(na, indices, weights, intercept)
            b = original.official_tls_transform(nb, indices, weights, intercept)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            ac, bc = a-a.mean(), b-b.mean()
            va, vb, vr = map(float, [np.mean(ac*ac), np.mean(bc*bc), np.mean(ac*bc)])
            shared = dict(sample=sample, split=split, spots=len(counts), VA=va, VB=vb, VR=vr,
                          rna_half_spearman=corr(a,b,True),
                          zero_total_A=int(np.count_nonzero(na.sum(1)==0)),
                          zero_total_B=int(np.count_nonzero(nb.sum(1)==0)))
            for arm in ARMS:
                c = float(.5*(np.mean(pc[arm]*ac)+np.mean(pc[arm]*bc)))
                moments.append(dict(**shared, arm=arm, VP=vp[arm], C=c,
                    prediction_half_pearson=.5*(corr(p[arm],a)+corr(p[arm],b)),
                    prediction_half_spearman=.5*(corr(p[arm],a,True)+corr(p[arm],b,True))))
            halves.append(np.stack([a,b]))
        frame = pd.DataFrame(moments)
        rows=[]
        for arm in ARMS:
            f=frame[frame.arm.eq(arm)]
            m=f[['VP','VA','VB','VR','C']].mean()
            repeat=float(m.VR/np.sqrt(m.VA*m.VB)) if m.VA>0 and m.VB>0 else float('nan')
            reason='defined' if m.VP>0 and m.VR>0 else ('nonpositive_prediction_variance' if m.VP<=0 else 'nonpositive_split_covariance')
            q=float(m.C/np.sqrt(m.VP*m.VR)) if reason=='defined' else float('nan')
            rows.append(dict(sample=sample,tissue='kidney' if sample.startswith('KC') else 'lung',
                 arm=arm,spots=len(counts),splits=SPLITS,genes=748,weighted_genes=747,
                 rna_repeatability=repeat,rna_half_spearman=float(f.rna_half_spearman.mean()),
                 full_pearson=corr(p[arm],full_measured),full_spearman=corr(p[arm],full_measured,True),
                 half_pearson=float(f.prediction_half_pearson.mean()),half_spearman=float(f.prediction_half_spearman.mean()),
                 programme_q=q,q_status=reason,VP=float(m.VP),VA=float(m.VA),VB=float(m.VB),VR=float(m.VR),C=float(m.C),
                 zero_half_rows_total=int(f.zero_total_A.sum()+f.zero_total_B.sum())))
        out=RUN/'results'/sample
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'RNA_HALF_SCORES.npz',scores=np.stack(halves),barcodes=fixed.barcode.to_numpy(dtype=str),split=np.arange(SPLITS),halves=np.array(['A','B']))
        frame.to_csv(out/'SPLIT_MOMENTS.tsv.gz',sep='\t',index=False)
        pd.DataFrame(rows).to_csv(out/'SECTION_METRICS.tsv',sep='\t',index=False)
        checks['elapsed_seconds']=time.monotonic()-start
        (out/'INPUT_CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
        return rows,checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    assert (RUN/'ANALYSIS_PLAN.md').exists()
    out=RUN/'results'
    out.mkdir(exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Reuse completed results; do not rerun unchanged analysis'
    rows,checks=[],[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers,8)) as pool:
        jobs=[pool.submit(one_section,i,str(args.data_root)) for i in range(8)]
        for future in concurrent.futures.as_completed(jobs):
            rr,cc=future.result(); rows.extend(rr); checks.append(cc)
            print(json.dumps({'sample':cc['sample'],'rna_repeatability':rr[0]['rna_repeatability'],'ordinary_q':rr[0]['programme_q'],'elapsed_seconds':cc['elapsed_seconds']}),flush=True)
    frame=pd.DataFrame(rows).sort_values(['sample','arm'])
    frame.to_csv(out/'SECTION_METRICS.tsv',sep='\t',index=False)
    summary=[]
    metrics=['rna_repeatability','rna_half_spearman','full_pearson','full_spearman','half_pearson','half_spearman','programme_q']
    for cohort in ['all','kidney','lung']:
        f=frame if cohort=='all' else frame[frame.tissue.eq(cohort)]
        for arm in ARMS:
            t=f[f.arm.eq(arm)]
            row=dict(cohort=cohort,arm=arm,sections=len(t),spots=int(t.spots.sum()),defined_q_sections=int(t.programme_q.notna().sum()))
            for metric in metrics:
                row[metric]=float(t[metric].mean()) if t[metric].notna().all() else None
                row[metric+'_min']=float(t[metric].min())
                row[metric+'_max']=float(t[metric].max())
            summary.append(row)
    pd.DataFrame(summary).to_csv(out/'COHORT_SUMMARY.tsv',sep='\t',index=False)
    (out/'INPUT_CHECKS.json').write_text(json.dumps(sorted(checks,key=lambda x:x['sample']),indent=2),encoding='utf8')
    receipt=dict(status='COMPLETE',sections=8,arms=2,splits_per_section=SPLITS,base_seed=BASE_SEED,
                 source_counts_root=str(args.data_root/'inputs/derived'),all_original_spots_retained=True,
                 total_spots=int(frame[frame.arm.eq('ordinary')].spots.sum()),
                 no_training=True,no_new_model_predictions=True,summary=summary)
    (out/'COMPLETE.json').write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding='utf8')
    print(pd.DataFrame(summary)[['cohort','arm','sections','rna_repeatability','rna_half_spearman','full_pearson','full_spearman','programme_q']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
