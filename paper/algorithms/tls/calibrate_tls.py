"""Ideal-score calibration using actual TLS readout and original spot support.

Existing scores/data/scorers are read-only. The prescribed ideal is f(lambda).
An independent Monte Carlo approximation E[f(Poisson(lambda/2))] diagnoses
nonlinearity of the independently normalised half-depth readout.
"""
from pathlib import Path
import argparse
import concurrent.futures
import importlib.util
import json
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SAMPLES = ['KC1', 'KC2', 'KC3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5']
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def correlation(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def metrics(p, a, b):
    p, a, b = [x - x.mean() for x in (p, a, b)]
    return dict(VP=float(np.mean(p*p)), VA=float(np.mean(a*a)),
                VB=float(np.mean(b*b)), VR=float(np.mean(a*b)),
                C=float((np.mean(p*a)+np.mean(p*b))/2))


def score(m):
    return float(m['C']/np.sqrt(m['VP']*m['VR'])) if m['VP'] > 0 and m['VR'] > 0 else None


def one(index, args):
    with threadpool_limits(limits=1):
        started = time.monotonic()
        sample = SAMPLES[index]
        out = args.out / sample
        receipt = out/'COMPLETE.json'
        if receipt.exists():
            return json.loads(receipt.read_text(encoding='utf8'))
        out.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location('tls_readout', args.source/'tls_readout.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with np.load(args.data/f'inputs/derived/{sample}/COUNTS_COMMON.npz', allow_pickle=False) as z:
            lam = z['counts']
            ids = z['barcodes'].astype(str)
            genes = tuple(z['gene_canonical'].astype(str))
        fixed = pd.read_csv(args.data/f'predicted_tls/folds/{sample}/SPOT_READOUTS.tsv.gz', sep='\t')
        positions = pd.Index(ids).get_indexer(fixed.barcode.astype(str))
        assert (positions >= 0).all() and len(np.unique(positions)) == len(positions)
        lam = lam[positions]
        assert np.issubdtype(lam.dtype, np.integer) and lam.min() >= 0
        binding = mod.coefficient_binding(genes, mod.load_coefficients(args.source/'coef-full.tsv.gz'))
        gi, weights, intercept, _ = binding
        transform = lambda n: mod.official_tls_transform(n, gi, weights, intercept)
        ideal = transform(lam)
        assert np.max(np.abs(ideal-fixed.measured_tls.to_numpy())) < 1e-9
        # MC is independent of the 20 calibration replicates. Save two independent
        # blocks to quantify approximation stability instead of claiming exact E[f].
        mc_rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 100]))
        block_sums = [np.zeros(len(lam)), np.zeros(len(lam))]
        for j in range(args.mc):
            value = transform(mc_rng.poisson(lam/2))
            block_sums[j//(args.mc//2)] += value
        mc1, mc2 = [x/(args.mc//2) for x in block_sums]
        mc_mean = (mc1+mc2)/2
        ideals = {'transform_of_lambda': ideal, 'mc_halfdepth_expectation': mc_mean,
                  'mc_first_half': mc1, 'mc_second_half': mc2}
        rows = []
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index, 200]))
        for replicate in range(20):
            total = rng.poisson(lam)
            na = rng.binomial(total, .5)
            nb = total-na
            a, b = transform(na), transform(nb)
            for name, pred in ideals.items():
                m = metrics(pred, a, b)
                rows.append(dict(sample=sample, replicate=replicate, ideal=name,
                                 **m, replicate_q=score(m)))
        frame = pd.DataFrame(rows)
        frame.to_csv(out/'REPLICATE_MOMENTS.tsv', sep='\t', index=False)
        np.savez_compressed(out/'IDEAL_SCORES.npz', transform_of_lambda=ideal,
                            mc_halfdepth_expectation=mc_mean, mc_first_half=mc1,
                            mc_second_half=mc2, barcode=fixed.barcode.to_numpy(dtype=str))
        results = []
        for name in ideals:
            f = frame[frame.ideal.eq(name)]
            m = f[['VP', 'VA', 'VB', 'VR', 'C']].mean().to_dict()
            q = score(m)
            results.append(dict(sample=sample, spots=len(lam), ideal=name, q_ideal=q,
                                passes_0p95=q is not None and q >= .95,
                                replicate_q_min=float(f.replicate_q.min()),
                                replicate_q_max=float(f.replicate_q.max()), **m))
        stability = dict(mc_draws=args.mc, block_draws=args.mc//2,
                         block_correlation=correlation(mc1, mc2),
                         mc_block_q_difference=abs(results[2]['q_ideal']-results[3]['q_ideal']),
                         deterministic_vs_mc_correlation=correlation(ideal, mc_mean))
        payload = dict(status='COMPLETE', sample=sample, seed=args.seed,
                       poisson_then_binomial_replicates=20,
                       prescribed_ideal='original_TLS_transform(lambda)',
                       mc_target='E[original_TLS_transform(Poisson(lambda/2))]',
                       rows=results, mc_stability=stability,
                       elapsed_seconds=time.monotonic()-started)
        receipt.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')
        print(json.dumps({'sample':sample, 'q_simple':results[0]['q_ideal'],
                          'q_mc':results[1]['q_ideal'], 'seconds':payload['elapsed_seconds']}), flush=True)
        return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'source')
    parser.add_argument('--out', type=Path, default=ROOT/'tls')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mc', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    assert args.mc >= 4 and args.mc % 2 == 0
    args.out.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        receipts = list(pool.map(one, range(8), [args]*8))
    rows = [r for entry in receipts for r in entry['rows']]
    pd.DataFrame(rows).to_csv(args.out/'SECTION_CALIBRATION.tsv', sep='\t', index=False)
    summary = {}
    for name in ['transform_of_lambda', 'mc_halfdepth_expectation']:
        values = [r['q_ideal'] for r in rows if r['ideal'] == name]
        summary[name] = dict(mean=float(np.mean(values)), min=min(values), max=max(values),
                             passed=sum(x >= .95 for x in values), sections=len(values))
    payload = dict(status='COMPLETE', summary=summary, sections=receipts,
                   original_data_modified=False, original_scorer_modified=False)
    (args.out/'COMPLETE.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
