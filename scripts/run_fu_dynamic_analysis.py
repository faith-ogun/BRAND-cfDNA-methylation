"""FU dynamic-change analysis on the 36-DMR signature.

For the FU prob_NR, the baseline classifier (L1-logistic, C=0.1) is refitted
on the full baseline matrix and then scores the FU matrix. The baseline
prob_NR for the change is the out-of-fold LOOCV prediction from
sensitivity_36dmr.csv.

Tests:
  1a. Wilcoxon rank-sum (Mann-Whitney U) on delta_prob_NR, R vs NR
  1b. Cox PH, PFS_time ~ delta_prob_NR
  2.  Per-region Wilcoxon rank-sum on delta_beta, R vs NR, BH-corrected over 36 regions
  3.  Aggregated direction-of-change count, R vs NR
Tests 1a and 1b are repeated in the patients whose FU sample was taken at week 6.

Outputs:
  results/sensitivity_analysis/fu_dynamic_results.csv          per-patient deltas + scores
  results/sensitivity_analysis/fu_dynamic_region_tests.csv     test 2 per-region results
  Console summary of all tests.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
SENS       = REPO / 'results' / 'sensitivity_analysis'

BASELINE_MAT   = SENS / 'dmr36_matrix.csv'
FU_MAT         = SENS / 'dmr36_matrix_fu.csv'
LOOCV_BASELINE = SENS / 'sensitivity_36dmr.csv'  # has out-of-fold prob_nonresponder
LOOCV_FINAL    = SENS / 'loocv_results_final.csv'

OUT_PERPAT  = SENS / 'fu_dynamic_results.csv'
OUT_REGION  = SENS / 'fu_dynamic_region_tests.csv'

C_LOG = 0.1   # locked baseline classifier hyperparameter (run_sensitivity.py)


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = p[order] * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return np.clip(out, 0, 1)


def cliffs_delta(a, b):
    """Effect size for Mann-Whitney; |delta| in [0, 1]."""
    a = np.asarray(a); b = np.asarray(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return float('nan')
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return (gt - lt) / (m * n)


def main():
    # Load matrices and outcome
    base = pd.read_csv(BASELINE_MAT, index_col='patient_id')
    fu   = pd.read_csv(FU_MAT, index_col='patient_id')
    pids = list(base.index)
    assert set(fu.index) == set(pids), 'FU patients differ from baseline set'
    fu = fu.loc[pids]   # align FU to baseline patient order
    region_cols = list(base.columns)
    assert list(fu.columns) == region_cols, 'region columns differ'

    out = pd.read_csv(LOOCV_BASELINE).set_index('patient_id').loc[pids]
    actual = out['actual'].astype(int).values            # 1 = non-responder
    prob_NR_base = out['prob_nonresponder'].values

    final = pd.read_csv(LOOCV_FINAL).set_index('patient_id').loc[pids]
    pfs_months = final['PFS Months'].astype(float).values
    pfs_event  = final['PFS'].astype(float).astype(int).values

    print(f'Loaded {len(pids)} patients')
    print(f'  responders: {(actual == 0).sum()}, non-responders: {(actual == 1).sum()}')
    print(f'  PFS events: {pfs_event.sum()}, censored: {(pfs_event == 0).sum()}')

    # --- Lock classifier on the full baseline matrix, score FU ---
    Xb = base.values.astype(float)
    Xfu = fu.values.astype(float)
    sc = StandardScaler()
    Xb_s = sc.fit_transform(Xb)
    Xfu_s = sc.transform(Xfu)

    clf = LogisticRegression(penalty='l1', solver='liblinear', C=C_LOG, max_iter=5000)
    clf.fit(Xb_s, actual)
    prob_NR_fu = clf.predict_proba(Xfu_s)[:, 1]
    nz = int((clf.coef_ != 0).sum())
    print(f'\nLocked baseline classifier: L1-logistic C={C_LOG}, '
          f'{nz}/{len(region_cols)} non-zero region coefs')

    delta_prob_NR = prob_NR_fu - prob_NR_base

    # ======================================================================
    # TEST 1a: Wilcoxon (Mann-Whitney U) on delta_prob_NR, R vs NR
    # ======================================================================
    d_resp   = delta_prob_NR[actual == 0]
    d_nonres = delta_prob_NR[actual == 1]
    u, p1a = mannwhitneyu(d_nonres, d_resp, alternative='two-sided')
    cd = cliffs_delta(d_nonres, d_resp)

    print('\n' + '=' * 70)
    print('TEST 1a: Mann-Whitney on delta_prob_NR (NR vs R)')
    print('=' * 70)
    print(f'  median delta, R:  {np.median(d_resp):+.4f}  (n={len(d_resp)})')
    print(f'  median delta, NR: {np.median(d_nonres):+.4f}  (n={len(d_nonres)})')
    print(f'  Mann-Whitney U:   {u:.1f}')
    print(f'  two-sided p:      {p1a:.4f}')
    print(f"  Cliff's delta:    {cd:+.3f}")

    # ======================================================================
    # TEST 1b: Cox PH, PFS_time ~ delta_prob_NR
    # ======================================================================
    cox_df = pd.DataFrame({
        'T': pfs_months,
        'E': pfs_event,
        'delta_prob_NR': delta_prob_NR,
    })
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='T', event_col='E')
    summ = cph.summary.loc['delta_prob_NR']
    hr = summ['exp(coef)']
    coef = summ['coef']
    se = summ['se(coef)']
    hr_per_0p1 = float(np.exp(coef * 0.1))
    ci_lo_0p1  = float(np.exp((coef - 1.96 * se) * 0.1))
    ci_hi_0p1  = float(np.exp((coef + 1.96 * se) * 0.1))
    p1b = summ['p']

    print('\n' + '=' * 70)
    print('TEST 1b: Cox PH, PFS ~ delta_prob_NR')
    print('=' * 70)
    print(f'  HR per 1.0 unit:  {hr:.3f}  (raw)')
    print(f'  HR per 0.1 unit:  {hr_per_0p1:.3f}  (95% CI {ci_lo_0p1:.3f} to {ci_hi_0p1:.3f})')
    print(f'  p-value:          {p1b:.4f}')

    # ======================================================================
    # TEST 2: Per-region Wilcoxon rank-sum (Mann-Whitney U) on delta_beta, BH-corrected
    # ======================================================================
    delta_beta = fu.values - base.values   # (42, 36)
    rows = []
    for i, region in enumerate(region_cols):
        db_r  = delta_beta[actual == 0, i]
        db_nr = delta_beta[actual == 1, i]
        u_r, p_r = mannwhitneyu(db_nr, db_r, alternative='two-sided')
        rows.append({
            'region': region,
            'median_delta_R':  float(np.median(db_r)),
            'median_delta_NR': float(np.median(db_nr)),
            'U': float(u_r),
            'p_raw': float(p_r),
        })
    region_df = pd.DataFrame(rows)
    region_df['q_BH'] = bh_adjust(region_df['p_raw'].values)
    region_df = region_df.sort_values('p_raw').reset_index(drop=True)
    region_df.to_csv(OUT_REGION, index=False)

    sig_q = (region_df['q_BH'] < 0.05).sum()
    print('\n' + '=' * 70)
    print('TEST 2: Per-region Wilcoxon on delta_beta, BH-corrected (36 tests)')
    print('=' * 70)
    print(f'  regions with q < 0.05: {sig_q}/36')
    print(f'  min raw p: {region_df["p_raw"].min():.4f}')
    print(f'  min BH q:  {region_df["q_BH"].min():.4f}')
    print('  Top 5 regions by raw p:')
    print(region_df.head(5)[['region', 'median_delta_R', 'median_delta_NR', 'p_raw', 'q_BH']].to_string(index=False))

    # ======================================================================
    # TEST 3: Aggregated direction-of-change
    # ======================================================================
    # Sign of each coef: positive coef => higher methylation pushes prob_NR up.
    # For NR-pushing regions (coef > 0), moving toward responder = delta_beta < 0.
    # For R-pushing regions (coef < 0), moving toward responder = delta_beta > 0.
    coefs = clf.coef_.flatten()
    # "responder-direction" score per patient = count of regions where:
    #   coef > 0 AND delta < 0, OR coef < 0 AND delta > 0.
    # Regions with coef == 0 contribute 0 (no signal).
    resp_dir = np.where(coefs > 0, -delta_beta, np.where(coefs < 0, delta_beta, 0))
    moved_toward_R = (resp_dir > 0).astype(int).sum(axis=1)

    mt_r  = moved_toward_R[actual == 0]
    mt_nr = moved_toward_R[actual == 1]
    u3, p3 = mannwhitneyu(mt_r, mt_nr, alternative='two-sided')
    cd3 = cliffs_delta(mt_r, mt_nr)

    print('\n' + '=' * 70)
    print('TEST 3: Aggregated direction-of-change count (R vs NR)')
    print('=' * 70)
    print(f'  Non-zero coefs used: {int((coefs != 0).sum())}/{len(coefs)}')
    print(f'  median count toward-R, R:  {np.median(mt_r):.1f}  (n={len(mt_r)})')
    print(f'  median count toward-R, NR: {np.median(mt_nr):.1f}  (n={len(mt_nr)})')
    print(f'  Mann-Whitney U:   {u3:.1f}')
    print(f'  two-sided p:      {p3:.4f}')
    print(f"  Cliff's delta:    {cd3:+.3f}")

    # ======================================================================
    # Per-patient output
    # ======================================================================
    per_pat = pd.DataFrame({
        'patient_id': pids,
        'actual': actual,
        'prob_NR_baseline': prob_NR_base,
        'prob_NR_fu': prob_NR_fu,
        'delta_prob_NR': delta_prob_NR,
        'count_regions_moved_toward_R': moved_toward_R,
        'PFS_Months': pfs_months,
        'pfs_event': pfs_event,
    })
    per_pat.to_csv(OUT_PERPAT, index=False)
    print(f'\nWrote: {OUT_PERPAT}')
    print(f'Wrote: {OUT_REGION}')

    # ======================================================================
    # Robustness check: wk6-only subset for tests 1a and 1b
    # ======================================================================
    WK3_PATIENTS = {'SJH30', 'SJH39', 'SJH53', 'BH5', 'BH6', 'BH29', 'BH42'}
    mask_wk6 = np.array([p not in WK3_PATIENTS for p in pids])
    n_wk6 = mask_wk6.sum()

    d_r6  = delta_prob_NR[mask_wk6 & (actual == 0)]
    d_nr6 = delta_prob_NR[mask_wk6 & (actual == 1)]
    u_w, p_w = mannwhitneyu(d_nr6, d_r6, alternative='two-sided')

    cox6 = pd.DataFrame({
        'T': pfs_months[mask_wk6],
        'E': pfs_event[mask_wk6],
        'delta_prob_NR': delta_prob_NR[mask_wk6],
    })
    cph6 = CoxPHFitter()
    cph6.fit(cox6, duration_col='T', event_col='E')
    s6 = cph6.summary.loc['delta_prob_NR']
    hr6_0p1 = float(np.exp(s6['coef'] * 0.1))
    se6 = s6['se(coef)']
    ci_lo6 = float(np.exp((s6['coef'] - 1.96 * se6) * 0.1))
    ci_hi6 = float(np.exp((s6['coef'] + 1.96 * se6) * 0.1))

    print('\n' + '=' * 70)
    print(f'ROBUSTNESS CHECK (wk6-only subset, n={n_wk6})')
    print('=' * 70)
    print(f'  Test 1a wk6: median R = {np.median(d_r6):+.4f}, NR = {np.median(d_nr6):+.4f}, '
          f'U = {u_w:.1f}, p = {p_w:.4f}')
    print(f'  Test 1b wk6: HR per 0.1 unit = {hr6_0p1:.3f} '
          f'(95% CI {ci_lo6:.3f} to {ci_hi6:.3f}), p = {s6["p"]:.4f}')

    # ======================================================================
    # Summary block
    # ======================================================================
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f"""
Test 1a (Mann-Whitney on delta_prob_NR, NR vs R):
  median delta R = {np.median(d_resp):+.4f}, NR = {np.median(d_nonres):+.4f}
  U = {u:.1f}, p = {p1a:.4f}, Cliff's delta = {cd:+.3f}

Test 1b (Cox PH, PFS ~ delta_prob_NR):
  HR per 0.1 unit = {hr_per_0p1:.3f} (95% CI {ci_lo_0p1:.3f} to {ci_hi_0p1:.3f}), p = {p1b:.4f}

Test 2 (Per-region Wilcoxon on delta_beta, BH q over 36 tests):
  regions with q < 0.05: {sig_q}/36
  min raw p = {region_df["p_raw"].min():.4f}, min q = {region_df["q_BH"].min():.4f}

Test 3 (Aggregated direction-of-change count, R vs NR):
  median R = {np.median(mt_r):.1f}, NR = {np.median(mt_nr):.1f}
  U = {u3:.1f}, p = {p3:.4f}, Cliff's delta = {cd3:+.3f}
""")


if __name__ == '__main__':
    main()
