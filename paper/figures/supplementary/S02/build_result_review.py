"""Render the completed frozen HEST V2 tables for presentation.

This module performs presentation-only arithmetic (unit conversion and the
declared equal-model display means). It never scores predictions, bootstraps,
tests a hypothesis, selects models/specimens, or manufactures example results.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL_ID = 'HEST_BACKGROUND_TRANSFER_V2_20260913'
DEFAULT_ROOT = Path(__file__).resolve().parent
MODELS = ['Ridge-Phikon-v2', 'MLP', 'ST-Net', 'HisToGene', 'Hist2ST', 'THItoGene',
          'DeepSpot', 'TCGN', 'DeepSpaCE', 'EGNv1', 'EGNv2']
SPECIMENS = {'IDC': ['TENX99', 'TENX95', 'NCBI785', 'NCBI783'],
             'PAAD': ['TENX116', 'TENX140', 'TENX126'], 'LUNG': ['TENX141', 'TENX118']}
ARENAS = ['BROAD_STRUCTURE_ALLOWED', 'COMPOSITION_MATCHED_ORIGINAL']
ARENA_ZH = {ARENAS[0]: 'No context matching', ARENAS[1]: 'RNA-context and depth matching'}
SUPPORTS = ['paired_common_strata', 'condition_specific']
SUPPORT_ZH = {'paired_common_strata': 'Paired common strata: primary comparison',
              'condition_specific': 'Condition-specific evaluable strata: descriptive'}
COMPONENTS = ['model_accuracy', 'majority_accuracy', 'model_minus_majority',
              'rna_accuracy', 'rna_minus_majority', 'normalized_ordering_gain']
RANK_FIELDS = ['ordinary_pearson', 'unrestricted_model_minus_majority', 'matched_model_minus_majority']
RANK_ZH = {'ordinary_pearson': 'Pearson on the same support',
           'unrestricted_model_minus_majority': 'Ordering gain without context matching',
           'matched_model_minus_majority': 'Ordering gain with context and depth matching'}


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, digits=3, signed=False, multiplier=1.0):
    try:
        x = float(value) * multiplier
    except (TypeError, ValueError):
        return 'NA'
    if not np.isfinite(x):
        return 'NA'
    return f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'


def pvalue(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 'NA'
    if not np.isfinite(x):
        return 'NA'
    return f'{x:.6f}'


def mean_full(values):
    """Display mean preserving missing values; never silently reduce support."""
    return float(np.mean(np.asarray(values, dtype=float)))


def table(headers, rows, classes=''):
    head = ''.join(f'<th>{esc(x)}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row) + '</tr>' for row in rows)
    return f'<div class="table-scroll"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def load_tables(root):
    receipt_path = root / 'COMPLETE.json'
    if not receipt_path.is_file():
        receipt_path = root / 'ESTIMABILITY_RECEIPT.json'
    if not receipt_path.is_file():
        raise SystemExit('No terminal scientific receipt yet. No review or placeholder results were written.')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('protocol_id') != PROTOCOL_ID or receipt.get('status') not in ('COMPLETE', 'TERMINAL_PRIMARY_NOT_ESTIMABLE'):
        raise SystemExit('Expected a completed frozen V2 receipt; no review was written.')
    files = {name: root / 'scores' / f'{name}.tsv' for name in
             ['SUMMARY', 'SPECIMEN_SCORES', 'PAIRED_SPECIMEN_CONTRASTS', 'RANKS',
              'RANK_COMPARISONS', 'BASELINE_SUPPORT', 'PCC_SPECIMEN']}
    missing = [str(p) for p in files.values() if not p.is_file() or not p.stat().st_size]
    if missing:
        raise SystemExit('Required scored tables are missing; no review was written:\n' + '\n'.join(missing))
    data = {name: pd.read_csv(path, sep='\t') for name, path in files.items()}
    # Only binding checks needed to avoid publishing a partial or misjoined page.
    frame = data['SPECIMEN_SCORES']
    keys = ['tissue', 'specimen', 'arena', 'model', 'support_kind']
    expected = {(t, s, a, m, k) for t, members in SPECIMENS.items() for s in members
                for a in ARENAS for m in MODELS for k in SUPPORTS}
    actual = list(frame[keys].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise SystemExit('The complete fixed model/specimen/condition/support matrix is required for this page.')
    summary = data['SUMMARY']
    primary = summary[(summary.record_type == 'MATCHED_MINUS_UNRESTRICTED') &
                      (summary.scope == 'TISSUE_BALANCED') & (summary.model == 'EQUAL_MODEL_MEAN')]
    if len(primary) != 1:
        raise SystemExit('Exactly one prespecified primary contrast is required.')
    secondary = summary[(summary.record_type == 'MATCHED_MINUS_UNRESTRICTED') &
                        (summary.scope == 'TISSUE_BALANCED') & summary.model.isin(MODELS)]
    if len(secondary) != 11 or set(secondary.model) != set(MODELS):
        raise SystemExit('All eleven secondary models must be retained.')
    secondary = secondary.set_index('model').loc[MODELS].reset_index()
    return receipt_path, receipt, files, data, primary.iloc[0], secondary


def component_rows(frame, scope):
    """Equal-model condition means directly from supplied summary rows."""
    rows = []
    for support in SUPPORTS:
        for arena in ARENAS:
            part = frame[(frame.record_type == 'CONDITION_COMPONENTS') & (frame.scope == scope) &
                         (frame.support_kind == support) & (frame.arena == arena)]
            part = part.set_index('model').reindex(MODELS)
            vals = {key: mean_full(part[key + '_mean']) for key in COMPONENTS}
            rows.append([SUPPORT_ZH[support], ARENA_ZH[arena],
                         number(vals['model_accuracy'], multiplier=100),
                         number(vals['majority_accuracy'], multiplier=100),
                         number(vals['model_minus_majority'], signed=True, multiplier=100),
                         number(vals['rna_accuracy'], multiplier=100),
                         number(vals['rna_minus_majority'], signed=True, multiplier=100)])
    return rows


def headline(primary):
    if primary.status != 'ESTIMABLE' or not np.isfinite(primary.effect_pp):
        return ('The prespecified nine-specimen comparison is not fully estimable',
                'Missing models and specimens remain in the analysis. A reduced subset cannot replace the prespecified primary analysis; inspect the support status.',
                'not-estimable')
    if primary.p_two_sided < .05:
        if primary.effect_pp < 0:
            return ('Ordering gain decreases after matching RNA context and depth',
                    'In nine HEST specimens outside HER2ST, expression-ordering assessment changes with the comparison context. This supports transfer of the two-condition comparison; it does not validate within-pathology-region assessment.',
                    'supported-decrease')
        if primary.effect_pp > 0:
            return ('Ordering gain increases after matching RNA context and depth',
                    'The evaluation conditions change the result, but the direction differs from the decrease in HER2ST. This is not a replication of the HER2ST direction.',
                    'supported-increase')
    return ('The transfer analysis does not establish a systematic change after matching',
            'The estimate and interval describe the change in these nine specimens; the prespecified two-sided exact test did not reach 0.05. This establishes neither equivalence of the conditions nor replication of the HER2ST decrease.',
            'uncertain')


def draw_effects(primary, secondary, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'svg.fonttype': 'none'})
    rows = [primary] + [r for _, r in secondary.iterrows()]
    labels = ['Primary: 11-model mean'] + MODELS
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    fig.subplots_adjust(left=.29, right=.97, bottom=.13, top=.90)
    ax.axvline(0, color='#a8b3b8', linewidth=.9, zorder=0)
    ax.axhspan(-.45, .45, color='#edf4f1', zorder=0)
    finite = []
    for i, row in enumerate(rows):
        point, lo, hi = float(row.effect_pp), float(row.ci_lower_pp), float(row.ci_upper_pp)
        if np.isfinite(point):
            colour = '#1e6956' if i == 0 else '#405a6b'
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=colour, linewidth=1.7)
                ax.plot([lo, lo], [i-.075, i+.075], color=colour, linewidth=1)
                ax.plot([hi, hi], [i-.075, i+.075], color=colour, linewidth=1)
                finite.extend([lo, hi])
            ax.plot(point, i, 'D' if i == 0 else 'o', color=colour, markersize=6 if i == 0 else 4.5)
            finite.append(point)
        else:
            ax.text(.98, i, 'Not estimable', transform=ax.get_yaxis_transform(), ha='right', va='center', color='#7b5653', fontsize=9)
    if finite:
        low, high = min([0.] + finite), max([0.] + finite)
        pad = max((high-low)*.08, .05)
        ax.set_xlim(low-pad, high+pad)
    ax.set_ylim(len(rows)-.45, -.55)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel('Matched minus unrestricted: model-minus-majority accuracy (percentage points)')
    ax.set_title('HEST: fixed nine-specimen transfer', loc='left', fontsize=13, fontweight='bold', pad=14)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#c2cbce')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#e9eeef', linewidth=.6)
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'HEST_paired_effects.{suffix}', dpi=170, facecolor='white')
    plt.close(fig)
    return 'HEST_paired_effects.png'


def draw_ranks(ranks, folder):
    import matplotlib.pyplot as plt
    rows = ranks[(ranks.scope == 'TISSUE_BALANCED') & ranks.model.isin(MODELS)].set_index('model').reindex(MODELS)
    fields = [key + '_rank' for key in RANK_FIELDS]
    values = rows[fields].to_numpy(float)
    # Undefined fixed ranks remain a complete table, not a smaller ranked plot.
    if not np.isfinite(values).all():
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    fig.subplots_adjust(left=.25, right=.75, bottom=.15, top=.89)
    colours = plt.get_cmap('tab20')(np.arange(len(MODELS)))
    for index, model in enumerate(MODELS):
        row = values[index]
        ax.plot([0, 1, 2], row, marker='o', markersize=4, linewidth=1.2, color=colours[index], alpha=.9)
    ax.set_xticks([0, 1, 2], ['Same-support PCC', 'Unrestricted gain', 'Matched gain'])
    ax.set_yticks(range(1, 12))
    ax.set_ylim(11.5, .5)
    ax.set_xlim(-.12, 2.12)
    ax.set_ylabel('Rank among the fixed 11 methods')
    ax.set_title('Descriptive rankings; equal tissue weights', fontsize=12, fontweight='bold', pad=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#edf0f1', linewidth=.6)
    ax.legend(MODELS, loc='center left', bbox_to_anchor=(1.05, .5), frameon=False, fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'HEST_descriptive_ranks.{suffix}', dpi=170, facecolor='white')
    plt.close(fig)
    return 'HEST_descriptive_ranks.png'


def draw_before_after(scores, summary, folder):
    import matplotlib.pyplot as plt
    fields = ['model_accuracy', 'majority_accuracy', 'model_minus_majority', 'rna_minus_majority']
    titles = ['Model ordering accuracy', 'Training-majority accuracy',
              'Model minus majority', 'RNA reference minus majority']
    colours = {'IDC':'#4279a3', 'PAAD':'#bd8549', 'LUNG':'#8b74a9'}
    fig, axes = plt.subplots(1,4,figsize=(14.7,4.8))
    fig.subplots_adjust(left=.055,right=.985,bottom=.23,top=.81,wspace=.35)
    handles = {}
    for ax,field,title in zip(axes,fields,titles):
        for tissue,members in SPECIMENS.items():
            for specimen in members:
                values = []
                for arena in ARENAS:
                    part = scores[(scores.tissue==tissue)&(scores.specimen==specimen)&
                                  (scores.arena==arena)&(scores.support_kind=='paired_common_strata')]
                    values.append(mean_full(part.set_index('model').reindex(MODELS)[field])*100)
                if np.isfinite(values).all():
                    handle, = ax.plot([0,1],values,'o-',color=colours[tissue],linewidth=.9,
                                      markersize=3.7,alpha=.65)
                    handles[tissue] = handle
        combined = []
        for arena in ARENAS:
            part = summary[(summary.record_type=='CONDITION_COMPONENTS')&
                           (summary.scope=='TISSUE_BALANCED')&
                           (summary.support_kind=='paired_common_strata')&(summary.arena==arena)]
            combined.append(mean_full(part.set_index('model').reindex(MODELS)[field+'_mean'])*100)
        if np.isfinite(combined).all():
            handle, = ax.plot([0,1],combined,'D-',color='#17372f',linewidth=2.5,markersize=5.5,zorder=5)
            handles['Equal-tissue mean'] = handle
            for x,y in zip([0,1],combined):
                ax.annotate(f'{y:.2f}',(x,y),xytext=(0,9),textcoords='offset points',
                            ha='center',fontsize=10,fontweight='bold',color='#17372f',
                            bbox={'facecolor':'white','edgecolor':'none','pad':1.1,'alpha':.85})
        ax.set_title(title,fontsize=11,pad=13)
        ax.set_xticks([0,1],['Unrestricted','Matched'])
        ax.set_xlim(-.19,1.19)
        ax.set_ylabel('Percent' if field in fields[:2] else 'Percentage points')
        ax.axhline(0,color='#cad3d5',linewidth=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#edf0f1',linewidth=.6)
        ax.margins(y=.17)
    fig.suptitle('Same fixed predictions, paired common strata',fontsize=13,fontweight='bold',y=.955)
    if handles:
        fig.legend(list(handles.values()),list(handles),loc='lower center',bbox_to_anchor=(.5,.03),
                   ncol=4,frameon=False,fontsize=10)
    for suffix in ('png','svg'):
        fig.savefig(folder/f'HEST_before_after_components.{suffix}',dpi=170,facecolor='white')
    plt.close(fig)
    return 'HEST_before_after_components.png'


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    receipt_path, receipt, sources, data, primary, secondary = load_tables(root)
    summary, scores = data['SUMMARY'], data['SPECIMEN_SCORES']
    paired, ranks = data['PAIRED_SPECIMEN_CONTRASTS'], data['RANKS']
    rank_cmp, baseline = data['RANK_COMPARISONS'], data['BASELINE_SUPPORT']
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / 'review_assets'
    assets.mkdir(exist_ok=True)
    effect_image = draw_effects(primary, secondary, assets)
    rank_image = draw_ranks(ranks, assets)
    before_after_image = draw_before_after(scores, summary, assets)
    relative = lambda path: os.path.relpath(path, output.parent).replace('\\', '/')
    source_link = lambda name, title=None: f'<a href="{esc(relative(sources[name]))}">{esc(title or name + ".tsv")}</a>'
    title, interpretation, result_class = headline(primary)
    primary_condition_means = []
    for arena in ARENAS:
        part = summary[(summary.record_type=='CONDITION_COMPONENTS')&(summary.scope=='TISSUE_BALANCED')&
                       (summary.support_kind=='paired_common_strata')&(summary.arena==arena)]
        primary_condition_means.append(mean_full(part.set_index('model').reindex(MODELS).model_minus_majority_mean)*100)
    condition_sentence = (f'Ordering gain above the training-set majority-direction baseline changes from '
                          f'<b>{number(primary_condition_means[0])} percentage points</b> without matching to '
                          f'<b>{number(primary_condition_means[1])} percentage points</b> after matching. Both values are accuracy minus baseline; the next section separates accuracy and reference components.')
    primary_specimens = paired[paired.model=='EQUAL_MODEL_MEAN'].matched_minus_unrestricted_pp
    n_negative_specimens = int((primary_specimens<0).sum())
    n_negative_models = int((secondary.effect_pp<0).sum())
    n_holm = int((secondary.p_holm_eleven<.05).sum())
    consistency_sentence = (f'{n_negative_specimens}/9 specimens have negative paired changes after equal-model averaging; '
                            f'{n_negative_models}/11 models have negative tissue-balanced mean changes, and '
                            f'{n_holm}/11 secondary model comparisons meet the prespecified Holm-adjusted criterion.')
    pooled_rank_cmp = rank_cmp[rank_cmp.scope=='TISSUE_BALANCED']
    before_after_tau = pooled_rank_cmp[(pooled_rank_cmp.left=='unrestricted_model_minus_majority')&
                                      (pooled_rank_cmp.right=='matched_model_minus_majority')].kendall_tau_b.iloc[0]
    pcc_matched_tau = pooled_rank_cmp[(pooled_rank_cmp.left=='ordinary_pearson')&
                                     (pooled_rank_cmp.right=='matched_model_minus_majority')].kendall_tau_b.iloc[0]
    ranking_interpretation = (f'Kendall tau-b between rankings before and after matching is <b>{number(before_after_tau,4)}</b>; '
                              f'between matched rankings and Pearson rankings on the same support it is <b>{number(pcc_matched_tau,4)}</b>. ')
    if np.isfinite(before_after_tau) and before_after_tau>=.8 and np.isfinite(pcc_matched_tau) and pcc_matched_tau>=.8:
        ranking_interpretation += 'Rankings remain broadly concordant in this cohort. A decrease in gain can transfer across cohorts without implying substantial model reordering in every cohort.'
    else:
        ranking_interpretation += 'These rankings describe the fixed model set in this cohort; rank changes alone do not establish a general failure of rankings.'
    effect_text = number(primary.effect_pp, signed=True)
    ci_text = f'{number(primary.ci_lower_pp, signed=True)} to {number(primary.ci_upper_pp, signed=True)}'
    primary_card = f'''<div class="hero-metrics"><div><span>Primary change · percentage points</span><strong>{effect_text}</strong></div><div><span>Reported 95% interval</span><strong class="medium">{ci_text}</strong></div><div><span>Two-sided exact p</span><strong class="medium">{pvalue(primary.p_two_sided)}</strong></div><div><span>Estimable / prespecified specimens</span><strong class="medium">{number(primary.n_specimens_estimable,0)} / 9</strong></div></div>'''

    component_headers = ['Support set', 'Condition', 'Model accuracy (%)', 'Training majority baseline (%)',
                         'Model minus baseline (pp)', 'RNA-reference accuracy (%)', 'RNA minus baseline (pp)']
    component_table = table(component_headers, component_rows(summary, 'TISSUE_BALANCED'))
    secondary_table = table(['Model', 'Paired change (pp)', '95% CI (pp)', 'Two-sided p',
                             'Holm p across 11 models', 'Status'],
                            [[r.model, number(r.effect_pp, signed=True),
                              f'{number(r.ci_lower_pp,signed=True)} to {number(r.ci_upper_pp,signed=True)}',
                              pvalue(r.p_two_sided), pvalue(r.p_holm_eleven), r.status]
                             for r in secondary.itertuples()])

    tissue_rows = []
    for tissue, members in SPECIMENS.items():
        r = summary[(summary.record_type == 'MATCHED_MINUS_UNRESTRICTED') &
                    (summary.scope == tissue) & (summary.model == 'EQUAL_MODEL_MEAN')].iloc[0]
        tissue_rows.append([tissue, len(members), number(r.effect_pp, signed=True),
                            f'Per specimen: {1/(3*len(members)):.6f}', 'Descriptive; no within-tissue significance claim'])
    tissue_table = table(['Tissue', 'Specimen count', 'Mean paired change across 11 models (pp)', 'Primary-comparison weight', 'Interpretation'], tissue_rows)

    specimen_sections = []
    specimen_summary_rows = []
    raw_headers = ['Model', 'Condition', 'Model accuracy (%)', 'Majority baseline (%)', 'Model minus baseline (pp)',
                   'RNA reference (%)', 'RNA minus baseline (pp)', 'Normalized ratio', 'Denominator / support status', 'RNA splits']
    for tissue, members in SPECIMENS.items():
        one_tissue = []
        for specimen in members:
            current = scores[(scores.tissue == tissue) & (scores.specimen == specimen)]
            delta_row = paired[(paired.tissue == tissue) & (paired.specimen == specimen) &
                               (paired.model == 'EQUAL_MODEL_MEAN')].iloc[0]
            values = {}
            for arena in ARENAS:
                part = current[(current.arena == arena) & (current.support_kind == 'paired_common_strata')].set_index('model').reindex(MODELS)
                values[arena] = {field: mean_full(part[field]) for field in COMPONENTS}
            specimen_summary_rows.append([tissue, specimen,
                number(values[ARENAS[0]]['model_minus_majority'],signed=True,multiplier=100),
                number(values[ARENAS[1]]['model_minus_majority'],signed=True,multiplier=100),
                number(delta_row.matched_minus_unrestricted_pp,signed=True),
                number(values[ARENAS[0]]['rna_minus_majority'],signed=True,multiplier=100),
                number(values[ARENAS[1]]['rna_minus_majority'],signed=True,multiplier=100)])
            support_parts = []
            for support in SUPPORTS:
                rows = []
                part = current[current.support_kind == support].set_index(['model','arena'])
                for model in MODELS:
                    for arena in ARENAS:
                        r = part.loc[(model,arena)]
                        rows.append([model,ARENA_ZH[arena],number(r.model_accuracy,multiplier=100),
                            number(r.majority_accuracy,multiplier=100),number(r.model_minus_majority,signed=True,multiplier=100),
                            number(r.rna_accuracy,multiplier=100),number(r.rna_minus_majority,signed=True,multiplier=100),
                            number(r.normalized_ordering_gain,4,signed=True),r.denominator_status,number(r.molecule_seeds,0)])
                support_parts.append(f'<h4>{esc(SUPPORT_ZH[support])}</h4>' + table(raw_headers,rows,'dense'))
            one_tissue.append(f'<details class="specimen"><summary>{esc(specimen)} · Mean paired change across 11 models: {number(delta_row.matched_minus_unrestricted_pp,signed=True)} percentage points</summary>{"".join(support_parts)}</details>')
        specimen_sections.append(f'<h3>{esc(tissue)} · {len(members)} specimens</h3>' + ''.join(one_tissue))
    specimen_table = table(['Tissue','Specimen','Model minus baseline before matching','Model minus baseline after matching','Paired change',
                             'RNA minus baseline before matching','RNA minus baseline after matching'], specimen_summary_rows)

    rank_sections = []
    for scope in ['TISSUE_BALANCED', *SPECIMENS]:
        r = ranks[ranks.scope == scope].set_index('model').reindex(MODELS)
        rows = [[model, number(v.ordinary_pearson,4),number(v.ordinary_pearson_rank,1),
                 number(v.unrestricted_model_minus_majority,signed=True,multiplier=100),
                 number(v.unrestricted_model_minus_majority_rank,1),
                 number(v.matched_model_minus_majority,signed=True,multiplier=100),
                 number(v.matched_model_minus_majority_rank,1),v.status] for model,v in r.iterrows()]
        t = table(['Model','Pearson','PCC rank','Unmatched gain (pp)','Unmatched rank','Matched gain (pp)','Matched rank','Status'],rows,'dense')
        if scope == 'TISSUE_BALANCED':
            rank_sections.append(t)
        else:
            rank_sections.append(f'<details><summary>{esc(scope)} descriptive within-tissue rankings</summary>{t}</details>')
    tau_rows = [[('Equal weighting of three tissues' if r.scope == 'TISSUE_BALANCED' else r.scope),
                 f'{RANK_ZH.get(r.left,r.left)} / {RANK_ZH.get(r.right,r.right)}',
                 number(r.kendall_tau_b,4),r.status] for r in rank_cmp.itertuples()]
    rank_plot = (f'<img src="{esc(relative(assets / rank_image))}" alt="Descriptive rankings of the same fixed models by Pearson and ordering gain before and after matching">'
                 if rank_image else '<p class="note">Some prespecified ranking components are not estimable, so a ranking plot with a reduced model set is not drawn. All ranks and statuses remain in the table.</p>')

    denominator = scores.drop_duplicates(['tissue','specimen','arena','support_kind'])
    flagged = denominator[denominator.denominator_status != 'ESTIMABLE']
    flag_rows = [[r.tissue,r.specimen,SUPPORT_ZH[r.support_kind],ARENA_ZH[r.arena],
                  number(r.rna_minus_majority,signed=True,multiplier=100),r.denominator_status]
                 for r in flagged.itertuples()]
    flag_table = table(['Tissue','Specimen','Support set','Condition','RNA minus baseline (pp)','Status'],flag_rows) if flag_rows else '<p>The RNA-reference denominator is positive for every specimen, condition and support set, and all are marked estimable in the source tables.</p>'
    base_rows = []
    for tissue,members in SPECIMENS.items():
        for specimen in members:
            for arena in ARENAS:
                group = baseline[(baseline.tissue == tissue)&(baseline.specimen == specimen)&(baseline.arena == arena)]
                fallback = sum(len(str(v).split(',')) for v in group.fallback_genes if pd.notna(v) and str(v).strip())
                base_rows.append([tissue,specimen,ARENA_ZH[arena],len(members)-1,len(group),fallback,
                                  number(group.training_judgeable_questions.min(),0),number(group.training_judgeable_questions.max(),0)])
    baseline_table = table(['Tissue','Held-out specimen','Condition','Baseline training specimens','Split records','Fallback gene-by-split count',
                             'Minimum judgeable training questions','Maximum judgeable training questions'],base_rows,'dense')

    css = '''
:root{--ink:#17303a;--muted:#647078;--accent:#236d59;--line:#dce4e3;--pale:#edf4f1;--bg:#f5f5f1}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.8}main{max-width:1320px;margin:28px auto;padding:0 26px 60px}header{padding:24px 0;border-bottom:1px solid var(--line)}.eyebrow{font-size:12px;letter-spacing:.10em;font-weight:700;color:var(--accent)}h1{font-size:33px;line-height:1.4;margin:12px 0}h2{font-size:23px;margin:0 0 14px}h3{font-size:18px;margin:24px 0 9px}h4{font-size:15px;margin:18px 0 5px}p{margin:10px 0}.lead{font-size:18px;max-width:1100px}.muted,.caption{font-size:14px;color:var(--muted)}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:25px;margin-top:23px}.hero-metrics{display:grid;grid-template-columns:1fr 1.5fr 1fr 1fr;gap:16px;background:var(--pale);padding:22px;margin:21px 0}.hero-metrics span{display:block;font-size:13px;color:var(--muted)}.hero-metrics strong{font-size:31px;display:block;line-height:1.4;margin-top:7px}.hero-metrics .medium{font-size:24px}.table-scroll{overflow:auto;margin:14px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:left;padding:10px 11px;border-bottom:1px solid var(--line)}th{background:var(--pale);font-weight:600}td:first-child{font-weight:600}th{white-space:nowrap}.dense{font-size:12px}.dense td,.dense th{padding:8px}.dense td{white-space:nowrap}img{max-width:100%;height:auto;display:block;margin:12px auto}.note{padding:11px 16px;background:var(--pale);border-left:3px solid var(--accent);font-size:14px}details{padding:10px 0}summary{cursor:pointer;font-weight:600}.specimen{border-top:1px solid var(--line)}a{color:var(--accent);text-underline-offset:3px}.links{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}nav{display:flex;gap:19px;flex-wrap:wrap;margin:17px 0;font-size:14px}.formula{font-family:Consolas,monospace;background:#f2f5f4;padding:10px 14px;font-size:14px}li{margin:7px 0}@media(max-width:800px){main{padding:0 13px 36px}h1{font-size:26px}.card{padding:18px}.hero-metrics{grid-template-columns:1fr 1fr;padding:16px}.hero-metrics strong{font-size:27px}.hero-metrics .medium{font-size:21px}}@media print{body{background:white}main{margin:0}.card{border:0;padding:10px 0}details{display:block}.table-scroll{overflow:visible}}
'''
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HEST two-condition transfer results</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">stRSC · HEST V2 · RESULTS</div><h1>{esc(title)}</h1><p class="lead">{esc(interpretation)}</p><p>{condition_sentence}</p>{primary_card}<p>{esc(consistency_sentence)}</p><p class="muted">Primary comparison: matched minus unmatched model accuracy above the training majority-direction baseline, using jointly estimable gene-by-direction-by-distance strata. The 11 models are equally weighted within each specimen. IDC, PAAD and LUNG each receive one-third weight, with equal specimen weights within tissues.</p><nav><a href="#components">Accuracy components</a><a href="#models">All models</a><a href="#specimens">All specimens</a><a href="#ranks">Rank comparisons</a><a href="#scope">Scope and sources</a></nav></header>
<section class="card" id="components"><h2>Accuracy, the baseline and the RNA reference</h2><p>Model accuracy above the majority-direction baseline is the primary quantity. RNA-reference accuracy above the same baseline is shown separately.</p><div class="formula">Ordering gain = model accuracy − training majority-direction baseline; RNA-reference denominator = RNA-reference accuracy − the same baseline</div><img src="{esc(relative(assets/before_after_image))}" alt="Model accuracy, majority baseline, gain and RNA-reference denominator before and after matching in nine specimens and the tissue-balanced mean"><p class="caption">Thin lines and points show the 11-model mean in each specimen, coloured by tissue; dark diamonds show the equally weighted three-tissue mean. Lines pair the same specimens without location resampling or smoothing. Condition means have no added difference confidence intervals; the primary difference interval is shown separately. Panel units are percentages, percentages, percentage points and percentage points.</p>{component_table}<p class="caption">Condition means use the original SUMMARY values, equal weighting of all 11 models, and equal tissue weights. Paired common strata support the primary test; condition-specific strata describe each condition. These support sets differ, so a difference between condition-specific means is not the paired primary effect. Missing values remain NA; incomplete model sets are not averaged.</p><details><summary>Components in each tissue</summary>{''.join(f'<h3>{t}</h3>'+table(component_headers,component_rows(summary,t)) for t in SPECIMENS)}</details></section>
<section class="card" id="models"><h2>One primary test and eleven secondary model comparisons</h2><p>The primary p value uses all 512 exact sign flips of nine specimen differences. Secondary model comparisons retain the prespecified 11-model Holm family. Intervals come from the original 10,000 within-tissue specimen bootstrap resamples and are not recomputed here.</p><img src="{esc(relative(assets / effect_image))}" alt="Prespecified primary effect and paired changes for all 11 models with their original 95% intervals"><p class="caption">Values, directions and intervals are taken directly from the source tables. No intervals are truncated, models removed or significance stars added. The intervals and exact test use different constructions; use the corresponding test p value rather than interval crossing to assess significance.</p>{secondary_table}<p class="links">{source_link('SUMMARY','Source table: conditions, primary comparison and model comparisons')}</p></section>
<section class="card" id="specimens"><h2>All nine specimens, grouped by tissue</h2>{tissue_table}<p class="caption">Within-tissue means are descriptive. The IDC baseline uses the other three specimens, PAAD the other two, and LUNG the other specimen. Genes, RNA splits and models are not additional independent specimens.</p><h3>Specimen-level components of the primary comparison</h3>{specimen_table}<p class="caption">All values are percentage points after equal weighting of the fixed 11 models within each specimen. Paired changes come from PAIRED_SPECIMEN_CONTRASTS; condition components use paired_common_strata. The sections below retain model accuracy, RNA reference, baseline, ratio and status for every specimen.</p>{''.join(specimen_sections)}<p class="links">{source_link('SPECIMEN_SCORES','All 396 specimen-by-model-by-condition-by-support rows')}{source_link('PAIRED_SPECIMEN_CONTRASTS','Specimen-level paired differences for the primary mean and every model')}</p></section>
<section class="card" id="ranks"><h2>Pearson and ordering rankings: descriptive comparisons</h2><p class="note">{ranking_interpretation}</p><p>Pearson uses the same 50 genes and location support. The two ordering gains use their respective condition-specific evaluable strata. Rankings describe the fixed 11 models; no ranking confidence interval or independent significance claim is added.</p>{rank_plot}{table(['Scope','Comparison','Kendall tau-b','Status'],tau_rows)}{''.join(rank_sections)}<p class="caption">Higher scores rank first and exact ties receive average ranks. If a complete component is not estimable, the corresponding ranking remains unavailable; models are not removed according to performance. Tau-b is taken from RANK_COMPARISONS without recalculation.</p><p class="links">{source_link('RANKS')}{source_link('RANK_COMPARISONS')}{source_link('PCC_SPECIMEN','Specimen-level Pearson on the same support')}</p></section>
<section class="card"><h2>RNA-reference and training-baseline status</h2><p>There are {len(denominator)} specimen-by-condition-by-support entries; {len(flagged)} have a non-positive denominator, a zero denominator or incomplete support in the original tables. Ratios with non-positive denominators retain their original values, but cannot be interpreted as a fraction of recoverable RNA ordering.</p>{flag_table}<details><summary>Majority-direction training and fallback records</summary>{baseline_table}<p class="caption">Fallback counts sum genes requiring the prespecified question-key fallback rule across splits. They are gene-by-split counts, not independent genes or patients. Training-question counts show the minimum and maximum across split records; the selection rule is unchanged.</p></details><p class="links">{source_link('BASELINE_SUPPORT')}</p></section>
<section class="card" id="scope"><h2>Scope of inference</h2><ul><li><b>Transfer of evaluation conditions.</b> The analysis extends the HER2ST comparison to nine HEST specimens and eleven fixed models, comparing unmatched context with RNA-context and depth matching. Pathology-region annotations are absent, so it does not evaluate the third, within-region stage.</li><li><b>Fixed 50-gene panels.</b> Each tissue uses its complete official 50-gene panel. The context representation excludes the entire hash fold containing the target gene. The analysis does not meet or claim the earlier protocol criterion of at least 100 shared genes.</li><li><b>Overlap with other study modules.</b> HEST is outside HER2ST, but the four IDC specimens also belong to the breast Xenium analysis; they are not a wholly independent cell-validation cohort.</li><li><b>Existing model predictions.</b> Some models were trained against depth-normalized targets. This analysis retains their native outputs and compares them with raw-count directions, without reconstructing predictions using held-out RNA depth. Target differences remain part of the interpretation.</li><li><b>Different question pairs.</b> Each unmatched candidate pool is fixed from IDs and coordinates before scoring, with at most one million pairs per specimen. Matching uses source-half context and depth. Distance-stratum counts are matched, but location pairs and within-condition source-difference thresholds differ. The contrast is therefore not a causal estimate of changing matching alone.</li><li><b>RNA references do not add biological replicates.</b> Ten molecule splits and bidirectional judgments construct and repeatedly evaluate questions; the independent units remain nine specimens. LUNG has two and PAAD three specimens, so no within-tissue significance conclusion is made.</li></ul><p class="note">This renderer reads the completed results without changing source scores, the statistical protocol, figures or manuscript.</p><p class="links"><a href="{esc(relative(root/'PROTOCOL.md'))}">Frozen V2 protocol</a><a href="{esc(relative(receipt_path))}">Completion record</a><a href="{esc(relative(root/'REPORT_ZH.md'))}">Original statistical report</a><a href="{esc(relative(assets/'HEST_paired_effects.svg'))}">Paired-interval SVG</a>{f'<a href="{esc(relative(assets/"HEST_descriptive_ranks.svg"))}">Descriptive-rank SVG</a>' if rank_image else ''}</p></section>
</main></body></html>'''
    tmp = output.with_name(output.name + '.tmp')
    tmp.write_text(content,encoding='utf-8')
    os.replace(tmp,output)
    return {'status':'AUTHOR_REVIEW_RENDERED_NOT_ADOPTED','protocol_id':PROTOCOL_ID,
            'source_receipt':str(receipt_path),'output':str(output),'primary_result_class':result_class,
            'source_specimen_rows':len(scores),'secondary_models':len(secondary),
            'new_inference_computed':False,'manuscript_modified':False,
            'effect_plot':str(assets/effect_image),'before_after_plot':str(assets/before_after_image),
            'rank_plot':str(assets/rank_image) if rank_image else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=DEFAULT_ROOT,
                        help='Local completed HEST V2 root containing scores/ and its terminal receipt.')
    parser.add_argument('--output',type=Path,default=None,
                        help='Default: RUN_ROOT/RESULT_REVIEW.html. No result is written before terminal tables exist.')
    args = parser.parse_args()
    print(json.dumps(build(args.run_root,args.output or args.run_root/'RESULT_REVIEW.html'),ensure_ascii=False,indent=2))
