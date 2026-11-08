# External TLS: split-count repeatability and programme agreement

Fixed before generating new split-count scores, 30 September 2026 (Asia/Taipei).

## Question
How repeatable is the measured TLS readout under molecule sampling, and how closely do the existing predictions follow its repeatable variation?

## Inputs retained
- All eight original external sections: KC1, KC2, KC3, LC1, LC2, LC3, LC4, LC5.
- Original prediction-valid tissue spots in their original barcode order; all retained throughout.
- Original 748-gene common panel and 747 corresponding published coefficients, plus intercept. FAIM2 stays in the normalisation denominator.
- Both fixed prediction arms: ordinary and neighbour-difference training. No models, weights, normalisation panels, annotations or prediction arrays are refitted.
- Original `official_tls_transform`: divide by each spot's panel total, divide each gene by its section-wide SD (ddof=0), no centring at this stage and no log transform, apply coefficients and intercept.

## Computation
1. Reconstruct full-count measured TLS scores from the saved raw integer counts and verify agreement with the saved scores. Confirm the original mean section Spearman values from the saved prediction scores. These checks catch barcode/panel/transform mismatches before interpreting new results.
2. Use 20 binomial probability-0.5 molecule splits per section, with NumPy SeedSequence([20260930, section_index]). Define B=N-A. Transform each half separately with the unchanged TLS function, including independent half-specific totals and gene SDs.
3. Keep zero-total rows: the existing function gives zero normalised expression and hence the intercept score. Zero-SD gene columns contribute zero. Record these cases; do not remove spots or sections after splitting.
4. For each split centre P,A,B separately over the same spots. Use mean products for VP, VA, VB, VR=cov(A,B), and C=(cov(P,A)+cov(P,B))/2.
5. Average moments over the 20 splits within each section. RNA repeatability is mean(VR)/sqrt(mean(VA)*mean(VB)); programme q is mean(C)/sqrt(mean(VP)*mean(VR)). q is NA with explicit reason if VP<=0 or mean(VR)<=0. No clipping, absolute-value denominator, or epsilon replacement.
6. Also report mean split-half Spearman, mean prediction-to-half Spearman and Pearson, and full-count prediction-to-RNA Pearson/Spearman. Full-count Spearman is retained as its original separate metric; q is not a correction applied to it.
7. Report all eight sections and equal-section averages, with kidney/lung descriptive summaries. Splits/spots are not independent biological replicates; do not add significance tests or patient-level confidence intervals. If any q is undefined, report coverage and do not silently call an available-section mean the eight-section mean.

## Interpretation and scope
- Repeatability is for the half-depth, independently renormalised TLS scores, not automatically full-depth reliability.
- Programme q concerns the fixed TLS readout, not the separate mean gene-level Spearman of 0.037.
- High repeatability with weak programme agreement supports the distinction between histological TLS recognition and recovery of measured molecular variation. Low repeatability requires a narrower molecular-recovery interpretation.
- No smoothing search, new cohorts, feature extraction or training in this run. The existing q implementation/calibration alignment is audited separately without rerunning prior tests.

## Outputs
SECTION_METRICS.tsv, COHORT_SUMMARY.tsv, SPLIT_MOMENTS.tsv.gz, RNA_HALF_SCORES.npz, INPUT_CHECKS.json, COMPLETE.json, concise Methods/Results updates and an eight-section supplementary table. Original raw counts and predictions remain in their existing directory.
