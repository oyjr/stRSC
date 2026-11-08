"""Item 1 only: new 20-realisation ideal-prediction calibration.

Calls the historical CRC transforms/FFT and HEST transforms/projection. Changes
only counts for the requested conditional simulation and, if necessary, the
prediction amplitude in a separately labelled derived scorer. Never trains.
"""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
import torch  # Windows DLL order; HEST's original projection also uses torch.
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import fft as spfft
from scipy.stats import poisson

REPS = 20
SOURCE_DIR = Path(__file__).resolve().parent / "source"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def q(vr, c, vp):
    return c / math.sqrt(vr * vp) if vr > 0 and vp > 0 else math.nan


def half_expectation(rates, transform):
    """Finite Poisson sum; <=2e-13 tail probability targeted per unique rate."""
    values, inverse = np.unique(rates, return_inverse=True)
    means, omitted = [], 0.
    for value in values:
        mu = float(value) / 2
        if mu == 0:
            means.append(0.); continue
        lo = max(0, int(poisson.ppf(1e-13, mu)))
        hi = int(poisson.ppf(1 - 1e-13, mu))
        k = np.arange(lo, hi + 1, dtype=np.float64)
        prob = np.exp(poisson.logpmf(k, mu))
        omitted = max(omitted, abs(1 - float(prob.sum())))
        f = np.log1p(2*k) if transform == 'log' else np.sqrt(2*k)
        means.append(float(np.sum(prob * f)))
    return np.asarray(means)[inverse].reshape(rates.shape), omitted


def hest_original_prediction_function(source):
    """Compile the actual final score_task prediction operations unchanged."""
    tree = ast.parse(source.read_text(encoding='utf8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score_task')
    names = {'centered_prediction', 'pnorm', 'p', 'pc', 'vpred'}
    body = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in names]
    body.sort(key=lambda n: n.lineno)
    assert [n.targets[0].id for n in body] == ['centered_prediction', 'pnorm', 'p', 'pc', 'vpred']
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=x) for x in ('prediction', 'p_scale', 'cal', 'bands', 'device')], kwonlyargs=[], kw_defaults=[], defaults=[])
    wrapper = ast.FunctionDef(name='original_prediction', args=args, body=body + [ast.Return(value=ast.Name(id='pc', ctx=ast.Load()))], decorator_list=[])
    env = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), 'exec'), env)
    return env['original_prediction']


def crc_original_module(source, fft_workers=1):
    """Execute unchanged pure original functions; avoid unrelated model loading."""
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fnames = {'normalize_prediction', 'split_transform', 'grid_contract', 'fft_fields'}
    const = {'WAVELENGTH_BANDS', 'GENE_BLOCK', 'SPLIT_SEEDS'}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in fnames)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in const for t in n.targets))]
    assert sum(isinstance(n, ast.FunctionDef) for n in nodes) == 4
    # Thread count is an execution setting only. The original FFT arguments,
    # normalisation, masks and transform implementation remain unchanged.
    fft_api = types.SimpleNamespace(fft2=lambda *a, **kw: spfft.fft2(*a, **{**kw, 'workers': fft_workers}))
    env = dict(np=np, pd=pd, sparse=sparse, spfft=fft_api, math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), env)
    return types.SimpleNamespace(**env)


def finish(out, contract, rows, energy_rows):
    raw = pd.DataFrame(rows)
    raw.to_csv(out/'REPLICATE_GRAMS.tsv', sep='\t', index=False)
    pd.DataFrame(energy_rows).to_csv(out/'GENE_AMPLITUDES.tsv', sep='\t', index=False)
    summary = []
    keys = ['scorer', 'prediction', 'band', 'manuscript_used']
    for key, group in raw.groupby(keys, sort=False, dropna=False):
        v = dict(zip(keys, key)); finite = group.q[np.isfinite(group.q)]
        v.update(q=q(group.V_RNA.mean(), group.C_model.mean(), group.V_model.mean()),
                 mean_replicate_q=float(finite.mean()), min_replicate_q=float(finite.min()),
                 max_replicate_q=float(finite.max()), undefined_replicates=int((~np.isfinite(group.q)).sum()),
                 V_RNA=float(group.V_RNA.mean()), C_model=float(group.C_model.mean()), V_model=float(group.V_model.mean()))
        summary.append(v)
    frame = pd.DataFrame(summary)
    # Training endpoint is explicitly the arithmetic mean of separately formed B4/B5 q.
    if contract['pathway'] == 'HEST':
        extra = []
        for key, part in frame.groupby(['scorer', 'prediction'], sort=False):
            bands = part.set_index('band')
            extra.append(dict(scorer=key[0], prediction=key[1], band='FineMean', manuscript_used=True,
                              q=float(bands.loc[['B4','B5'], 'q'].mean()) if bands.loc[['B4','B5'], 'q'].notna().all() else math.nan,
                              undefined_replicates=int(raw.loc[(raw.scorer == key[0]) & (raw.prediction == key[1]) & raw.band.isin(['B4','B5']), 'q'].isna().sum())))
        frame = pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)
    frame.to_csv(out/'SECTION_CONDITIONS.tsv', sep='\t', index=False)
    ideal = frame.loc[(frame.prediction == 'expected_native') & frame.manuscript_used]
    contract['status'] = 'SYNTHETIC_HARNESS_COMPLETE' if contract.get('synthetic_fixture') else 'COMPLETED_NEW_CONDITIONAL_POISSON_CHECK'
    contract['legacy_below_095'] = int(((ideal.scorer == 'legacy') & (ideal.q < .95)).sum())
    contract['legacy_undefined'] = int(ideal.loc[ideal.scorer == 'legacy', 'q'].isna().sum())
    contract['derived_below_095'] = int(((ideal.scorer == 'amplitude_matched') & (ideal.q < .95)).sum())
    contract['derived_undefined'] = int(ideal.loc[ideal.scorer == 'amplitude_matched', 'q'].isna().sum())
    write_json(out/'COMPLETE.json', contract)
    print(json.dumps(clean(contract)), flush=True)


def hest(args):
    out = args.output/args.task/args.specimen; out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir))
    cal = load_module(args.source_dir/'g0047_hest_calibration.py', 'g0047_hest_calibration')
    cal.INPUT, cal.BASIS, cal.MOMENTS = [args.frozen_root/x for x in ('input','basis','moments')]
    counts, basis = cal._load_specimen(args.task, args.specimen)
    bands = cal._device_bands(basis, args.device)
    weights = np.load(cal.MOMENTS/args.task/f'{args.specimen}.weights.npz')
    eligible = weights['eligible'].astype(bool)
    rscale = np.divide(np.sqrt(weights['w']), np.sqrt(weights['d2']), out=np.zeros(counts.shape[1]), where=weights['d2'] > cal.NUMERIC_TOL)
    rscale[~eligible] = 0
    old_amplitude = rscale*np.sqrt(np.maximum(weights['ell2'], 0))
    original = hest_original_prediction_function(args.source_dir/'g0047b_hest_spectrum_score.py')
    expectation, omitted = half_expectation(counts, 'log')
    predictions = {'expected_native': expectation, 'simplified_native': np.log1p(counts.astype(float))}
    # Unit-normalisation and projection execute actual final scorer statements.
    unit = {name: original(values, eligible.astype(float), cal, bands, args.device) for name, values in predictions.items()}
    # Preserve the historical order: gene amplitude is applied before float32
    # projection, rather than moving it past the matrix multiply.
    legacy_pc = {name: original(values, old_amplitude, cal, bands, args.device) for name, values in predictions.items()}
    legacy_vp = {name: np.asarray([np.sum(c*c) for c in pc]) for name, pc in legacy_pc.items()}
    legacy_c = {name: np.empty((REPS,5)) for name in predictions}
    vp_gene = {name: np.asarray([np.sum(c*c, axis=0) for c in pc]) for name, pc in unit.items()}
    c_gene = {name: np.empty((REPS,5,counts.shape[1])) for name in predictions}
    vr_gene = np.empty((REPS,5,counts.shape[1]))
    seeds = []
    for rep in range(REPS):
        seed = cal._seed('FOUR-CHECKS-ITEM1-20261001', args.task, args.specimen, rep)
        seeds.append(seed)
        full = np.random.default_rng(seed).poisson(counts)
        # _halves executes original binomial splitting, log transform and centring.
        aa, bb = cal._halves(full, args.task, args.specimen, seed)
        ac = cal._coefficients(bands, aa*rscale[None,:], args.device)
        bc = cal._coefficients(bands, bb*rscale[None,:], args.device)
        for bi in range(5):
            vr_gene[rep,bi] = np.sum(ac[bi]*bc[bi], axis=0)
            for name in predictions:
                c_gene[name][rep,bi] = .5*np.sum(unit[name][bi]*(ac[bi]+bc[bi]), axis=0)
                legacy_c[name][rep,bi] = .5*np.sum(legacy_pc[name][bi]*(ac[bi]+bc[bi]))
        if rep in (0,9,19): print(json.dumps({'specimen':args.specimen,'replicates':rep+1}), flush=True)
    new_amplitude = np.sqrt(np.maximum(vr_gene.mean(axis=0).sum(axis=0),0))
    legacy_q = [q(vr_gene[:,bi].sum(axis=1).mean(), legacy_c['expected_native'][:,bi].mean(), float(legacy_vp['expected_native'][bi])) for bi in range(5)]
    amplitudes = {'legacy':old_amplitude}
    if args.always_derived or any(np.isfinite(x) and x < .95 for x in legacy_q): amplitudes['amplitude_matched'] = new_amplitude
    rows = []
    for version, amplitude in amplitudes.items():
        for name in predictions:
            for rep in range(REPS):
                for bi in range(5):
                    vr = float(vr_gene[rep,bi].sum())
                    c = float(legacy_c[name][rep,bi]) if version == 'legacy' else float((c_gene[name][rep,bi]*amplitude).sum())
                    vp = float(legacy_vp[name][bi]) if version == 'legacy' else float((vp_gene[name][bi]*amplitude**2).sum())
                    rows.append(dict(scorer=version,prediction=name,replicate=rep,band=f'B{bi+1}',manuscript_used=True,V_RNA=vr,C_model=c,V_model=vp,q=q(vr,c,vp)))
    contract = dict(pathway='HEST',task=args.task,specimen=args.specimen,replicates=REPS,spots=len(counts),genes=counts.shape[1],eligible_genes=int(eligible.sum()),
                    source_dir=str(args.source_dir.resolve()), final_scorer=str((args.source_dir/'g0047b_hest_spectrum_score.py').resolve()), frozen_root=str(args.frozen_root.resolve()),
                    transform='log1p(2*half_count)',expected_prediction='E[log1p(2*Poisson(lambda/2))]',simplified_prediction='log1p(lambda)',
                    prediction_operations='AST-extracted unchanged assignments from score_task',rng_seeds=seeds,expected_mass_omitted_max=omitted,
                    derived_amplitude='sqrt(max(sum_all_five_bands(mean_20_replicates(A_scaled*B_scaled)),0)) per gene',original_rna_scaling_unchanged=True,
                    derived_same_realisation_energy_estimate=True,original_results_modified=False,synthetic_fixture=args.synthetic_fixture)
    contract['zero_projected_ideal_genes'] = int((vp_gene['expected_native'].sum(axis=0) <= cal.NUMERIC_TOL).sum())
    energy = [dict(gene_index=i,eligible=bool(eligible[i]),rna_scale=float(rscale[i]),legacy_amplitude=float(old_amplitude[i]),derived_amplitude=float(new_amplitude[i]),all_band_scaled_crossenergy=float(vr_gene[:,:,i].mean(axis=0).sum()),expected_unit_prediction_energy=float(vp_gene['expected_native'][:,i].sum())) for i in range(counts.shape[1])]
    finish(out,contract,rows,energy)


def crc(args):
    out=args.output/args.specimen/f'{args.scale}um'; out.mkdir(parents=True,exist_ok=True)
    orig=crc_original_module(args.crc_source,args.fft_workers)
    root=args.crc_payload/'cache'/'crc_grids'/args.specimen
    panel=pd.read_csv(args.crc_panel,sep='\t')['gene_index'].astype(int).to_numpy()
    allcounts=sparse.load_npz(root/f'{args.scale}um'/'counts_windows_by_genes.npz').tocsr()
    counts=allcounts[:,panel].tocsr()
    windows=pd.read_parquet(root/f'{args.scale}um'/'window_axis.parquet')
    area=windows.effective_area_um2.to_numpy(float)
    iy,ix,ny,nx,masks=orig.grid_contract(windows,args.scale)
    names=list(masks); nband=len(names); ng=len(panel)
    modes=[int(masks[n].sum()) for n in names]
    used=[low >= 2*args.scale for _,low,_ in orig.WAVELENGTH_BANDS]
    vr_gene=np.empty((REPS,nband,ng))
    prednames=('expected_native','simplified_native')
    c_gene={n:np.empty((REPS,nband,ng)) for n in prednames}
    vp_gene={n:np.empty((nband,ng)) for n in prednames}
    omitted=0.; seed_base=2026100100
    for offset in range(0,ng,orig.GENE_BLOCK):
        stop=min(offset+orig.GENE_BLOCK,ng)
        block=counts[:,offset:stop].tocsr()
        expected_data,err=half_expectation(block.data,'sqrt'); omitted=max(omitted,err)
        # A frozen abundance-ranked panel gives unsorted CSR column indices.
        # astype() may canonicalise them, so attach transformed values to their
        # original indices explicitly rather than assigning after astype().
        expected=sparse.csr_matrix((expected_data.copy(),block.indices.copy(),block.indptr.copy()),shape=block.shape)
        expected=expected.toarray()/np.sqrt(area[:,None])
        simple=np.sqrt(block.toarray()/area[:,None])
        unit={n:orig.fft_fields(orig.normalize_prediction(p),iy,ix,ny,nx) for n,p in zip(prednames,(expected,simple))}
        for n in prednames:
            for bi,b in enumerate(names): vp_gene[n][bi,offset:stop]=np.square(np.abs(unit[n][masks[b]])).sum(axis=0)
        for rep in range(REPS):
            seed=seed_base+rep+offset*1009
            full=block.copy(); full.data=np.random.default_rng(seed).poisson(block.data)
            aa,bb=orig.split_transform(full,area,seed+1000003)
            fa,fb=[orig.fft_fields(x,iy,ix,ny,nx) for x in (aa,bb)]
            for bi,b in enumerate(names):
                mask=masks[b]
                vr_gene[rep,bi,offset:stop]=np.real(np.conj(fa[mask])*fb[mask]).sum(axis=0)
                for n in prednames: c_gene[n][rep,bi,offset:stop]=.5*np.real(np.conj(unit[n][mask])*(fa[mask]+fb[mask])).sum(axis=0)
        print(json.dumps({'specimen':args.specimen,'scale':args.scale,'genes_completed':stop}),flush=True)
    old_amplitude=np.ones(ng)
    new_amplitude=np.sqrt(np.maximum(vr_gene.mean(axis=0).sum(axis=0),0))
    legacy_q=[q(vr_gene[:,bi].sum(axis=1).mean(),c_gene['expected_native'][:,bi].sum(axis=1).mean(),float(vp_gene['expected_native'][bi].sum())) for bi in range(nband)]
    amplitudes={'legacy':old_amplitude}
    if args.always_derived or any(u and np.isfinite(x) and x < .95 for u,x in zip(used,legacy_q)): amplitudes['amplitude_matched']=new_amplitude
    rows=[]
    for version,amp in amplitudes.items():
        for name in prednames:
            for rep in range(REPS):
                for bi,b in enumerate(names):
                    vr=float(vr_gene[rep,bi].sum()); c=float((c_gene[name][rep,bi]*amp).sum()); vp=float((vp_gene[name][bi]*amp**2).sum())
                    rows.append(dict(scorer=version,prediction=name,replicate=rep,band=b,manuscript_used=used[bi],V_RNA=vr,C_model=c,V_model=vp,q=q(vr,c,vp)))
    contract=dict(pathway='CRC',runner_revision='crc_expected_csr_axis_v2',specimen=args.specimen,scale_um=args.scale,replicates=REPS,spots=counts.shape[0],genes=ng,fourier_modes=dict(zip(names,modes)),fft_workers=args.fft_workers,
                  final_scorer=str(args.crc_source.resolve()),crc_payload=str(args.crc_payload.resolve()),crc_panel=str(args.crc_panel.resolve()),
                  transform='sqrt(2*half_count/effective_area)',expected_prediction='E[sqrt(2*Poisson(lambda/2)/effective_area)]',simplified_prediction='sqrt(lambda/effective_area)',
                  original_functions=['normalize_prediction','split_transform','grid_contract','fft_fields'],seed_formula='2026100100 + replicate + gene_block_offset*1009; split seed adds1000003',
                  expected_mass_omitted_max=omitted,derived_amplitude='sqrt(max(sum_all_seven_bands(mean_20_replicates(A_scaled*B_scaled)),0)) per gene',
                  original_rna_scaling_unchanged=True,derived_same_realisation_energy_estimate=True,original_results_modified=False,synthetic_fixture=args.synthetic_fixture)
    contract['zero_projected_ideal_genes'] = int((vp_gene['expected_native'].sum(axis=0) <= 1e-12).sum())
    energy=[dict(gene_index=int(panel[i]),legacy_amplitude=1.,derived_amplitude=float(new_amplitude[i]),all_band_scaled_crossenergy=float(vr_gene[:,:,i].mean(axis=0).sum()),expected_unit_prediction_energy=float(vp_gene['expected_native'][:,i].sum())) for i in range(ng)]
    finish(out,contract,rows,energy)


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='pathway',required=True)
    h=sub.add_parser('hest'); h.add_argument('--source-dir',type=Path,default=SOURCE_DIR,help='Bundled HEST scorer source directory'); h.add_argument('--frozen-root',type=Path,required=True); h.add_argument('--task',required=True); h.add_argument('--specimen',required=True); h.add_argument('--device',default='cpu'); h.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('crc'); c.add_argument('--crc-source',type=Path,default=SOURCE_DIR/'g0048_crc_physical_spectrum.py',help='Bundled CRC physical-spectrum scorer'); c.add_argument('--crc-payload',type=Path,required=True); c.add_argument('--crc-panel',type=Path,required=True); c.add_argument('--specimen',required=True); c.add_argument('--scale',type=int,required=True); c.add_argument('--fft-workers',type=int,default=1); c.add_argument('--output',type=Path,required=True)
    for parser in (h,c):
        parser.add_argument('--always-derived',action='store_true',help='Run after a failure in any item1 condition; retain both outputs')
        parser.add_argument('--synthetic-fixture',action='store_true')
    args=ap.parse_args(); (hest if args.pathway=='hest' else crc)(args)


if __name__=='__main__': main()
