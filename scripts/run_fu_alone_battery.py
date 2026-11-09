"""FU-alone battery: same stat panel as baseline, but on FU samples.

Mirrors run_sensitivity.py + run_continuous_validation.py for FU. Two parts:

  Part A. Binary classifier on FU:
    - In each LOOCV fold, the L1-logistic (C=0.1) is refitted on the other 41
      baseline samples and labels and predicts prob_NR for the held-out
      patient's FU sample.
    - Compute accuracy, AUC, sensitivity, specificity, PPV, NPV, Fisher exact,
      Cox HR binary, Cox HR per 10pp of prob_NR, Harrell's C on prob_NR.

  Part B. Continuous Cox on FU:
    - LOOCV penalised Cox on the FU 36-DMR matrix (mirrors
      run_continuous_validation.py exactly, FU swapped for baseline).
    - Harrell's C + 2000-iter bootstrap 95% CI.

Outcome time/event: PFS measured from treatment start, the same clock as the
baseline analysis (the FU sample was drawn at week 3 or 6 after treatment start).

Inputs:
  results/sensitivity_analysis/dmr36_matrix.csv         baseline 42x36 (training data for each fold)
  results/sensitivity_analysis/dmr36_matrix_fu.csv      FU 42x36
  results/sensitivity_analysis/sensitivity_36dmr.csv    actual labels + PFS from baseline run
  results/sensitivity_analysis/loocv_results_final.csv  PFS time + event

Outputs:
  results/sensitivity_analysis/fu_alone_perpatient.csv
  results/sensitivity_analysis/fu_alone_battery.csv     one-row summary
  Console summary in the same format as run_sensitivity.py.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
SENS       = REPO / 'results' / 'sensitivity_analysis'

BASELINE_MAT = SENS / 'dmr36_matrix.csv'
FU_MAT       = SENS / 'dmr36_matrix_fu.csv'
LABELS_CSV   = SENS / 'sensitivity_36dmr.csv'
LOOCV_FINAL  = SENS / 'loocv_results_final.csv'

OUT_PERPAT  = SENS / 'fu_alone_perpatient.csv'
OUT_BATTERY = SENS / 'fu_alone_battery.csv'

C_LOG = 0.1                # locked binary classifier
COX_PENALIZER = 0.5        # mirrors run_continuous_validation.py


def main():
    base = pd.read_csv(BASELINE_MAT, index_col='patient_id')
    fu   = pd.read_csv(FU_MAT, index_col='patient_id')
    pids = list(base.index)
    fu = fu.loc[pids]

    labels = pd.read_csv(LABELS_CSV).set_index('patient_id').loc[pids]
    y = labels['actual'].astype(int).values   # 1 = non-responder

    final = pd.read_csv(LOOCV_FINAL).set_index('patient_id').loc[pids]
    T = final['PFS Months'].astype(float).values
    E = final['PFS'].astype(float).astype(int).values

    print(f'{len(pids)} patients, {y.sum()} non-responders, {(y == 0).sum()} responders')
    print(f'  PFS events: {E.sum()}, censored: {(E == 0).sum()}')

    # =====================================================================
    # PART A. Binary classifier on FU, LOOCV.
    # Per fold: hold out patient i; refit L1-logistic on the OTHER 41 baseline
    # samples + labels; predict patient i's FU sample. Mirrors the baseline
    # LOOCV so the numbers are directly comparable with baseline.
    # =====================================================================
    Xb = base.values.astype(float)
    Xfu = fu.values.astype(float)
    n = len(pids)
    prob_fu = np.full(n, np.nan)
    nz_per_fold = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        sc_i = StandardScaler()
        Xtr_s  = sc_i.fit_transform(Xb[tr])
        Xte_s  = sc_i.transform(Xfu[i:i+1])
        clf_i = LogisticRegression(penalty='l1', solver='liblinear',
                                   C=C_LOG, max_iter=5000)
        clf_i.fit(Xtr_s, y[tr])
        prob_fu[i] = clf_i.predict_proba(Xte_s)[0, 1]
        nz_per_fold.append(int((clf_i.coef_ != 0).sum()))
    pred_fu = (prob_fu >= 0.5).astype(int)
    print(f'\nLOOCV binary classifier (refit per fold on baseline-of-41), '
          f'mean non-zero coefs across folds: {np.mean(nz_per_fold):.1f}/36')

    # binary metrics
    acc = accuracy_score(y, pred_fu)
    auc = roc_auc_score(y, prob_fu)
    cm  = confusion_matrix(y, pred_fu, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    ppv  = tp / (tp + fp) if (tp + fp) else float('nan')
    npv  = tn / (tn + fn) if (tn + fn) else float('nan')
    _, fisher_p = fisher_exact(cm)

    # Cox HR binary (predicted NR vs not)
    cox_df = pd.DataFrame({'T': T, 'E': E, 'pred': pred_fu})
    cph_b = CoxPHFitter().fit(cox_df, duration_col='T', event_col='E')
    sb = cph_b.summary.loc['pred']
    coef, se = sb['coef'], sb['se(coef)']
    hr_bin = float(np.exp(coef))
    hr_bin_lo = float(np.exp(coef - 1.96 * se))
    hr_bin_hi = float(np.exp(coef + 1.96 * se))
    p_bin = sb['p']

    # Cox HR per 10pp of prob_NR
    cox_df2 = pd.DataFrame({'T': T, 'E': E, 'prob': prob_fu})
    cph_c = CoxPHFitter().fit(cox_df2, duration_col='T', event_col='E')
    sc_ = cph_c.summary.loc['prob']
    coef_c, se_c = sc_['coef'], sc_['se(coef)']
    hr_10 = float(np.exp(coef_c * 0.1))
    hr_10_lo = float(np.exp((coef_c - 1.96 * se_c) * 0.1))
    hr_10_hi = float(np.exp((coef_c + 1.96 * se_c) * 0.1))
    p_10 = sc_['p']

    # Harrell's C on prob_NR (binary-classifier output as a risk score)
    cidx_bin = concordance_index(T, -prob_fu, E)
    # bootstrap CI for cidx_bin
    rng = np.random.default_rng(42)
    boot = []
    idx = np.arange(len(pids))
    for _ in range(2000):
        s = rng.choice(idx, size=len(idx), replace=True)
        if E[s].sum() < 2:
            continue
        try:
            boot.append(concordance_index(T[s], -prob_fu[s], E[s]))
        except Exception:
            continue
    ci_lo_bin, ci_hi_bin = np.percentile(boot, [2.5, 97.5])

    print('\n' + '=' * 70)
    print('PART A. Binary classifier on FU (LOOCV, refit on baseline-of-41 per fold)')
    print('=' * 70)
    print(f'  Accuracy:     {acc:.3f}')
    print(f'  AUC:          {auc:.3f}')
    print(f'  Sensitivity:  {sens:.3f}  ({tp}/{tp+fn})')
    print(f'  Specificity:  {spec:.3f}  ({tn}/{tn+fp})')
    print(f'  PPV:          {ppv:.3f}  ({tp}/{tp+fp})')
    print(f'  NPV:          {npv:.3f}  ({tn}/{tn+fn})')
    print(f'  Fisher p:     {fisher_p:.4f}')
    print(f'  Cox HR binary: {hr_bin:.3f}  (95% CI {hr_bin_lo:.3f} to {hr_bin_hi:.3f}), p = {p_bin:.4f}')
    print(f'  Cox HR per 10pp: {hr_10:.3f} (95% CI {hr_10_lo:.3f} to {hr_10_hi:.3f}), p = {p_10:.4f}')
    print(f"  Harrell's C on prob_NR: {cidx_bin:.3f}  (95% CI {ci_lo_bin:.3f} to {ci_hi_bin:.3f})")

    # =====================================================================
    # PART B. Continuous Cox on FU (LOOCV penalised Cox on FU 36-DMR matrix)
    # =====================================================================
    print('\nRunning LOOCV penalised Cox on FU matrix...')
    Xv = fu.values.astype(float)
    n, p = Xv.shape
    risk = np.full(n, np.nan)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr = Xv[tr].copy()
        Xte = Xv[i:i + 1].copy()
        med = np.nanmedian(Xtr, axis=0)
        for c in range(p):
            Xtr[np.isnan(Xtr[:, c]), c] = med[c]
            Xte[np.isnan(Xte[:, c]), c] = med[c]
        sc2 = StandardScaler()
        Xtr_s = sc2.fit_transform(Xtr)
        Xte_s = sc2.transform(Xte)
        df_tr = pd.DataFrame(Xtr_s, columns=[f'f{c}' for c in range(p)])
        df_tr['T'] = T[tr]; df_tr['E'] = E[tr]
        cph = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=0.0)
        try:
            cph.fit(df_tr, duration_col='T', event_col='E')
            df_te = pd.DataFrame(Xte_s, columns=[f'f{c}' for c in range(p)])
            risk[i] = float(cph.predict_partial_hazard(df_te).values[0])
        except Exception as e:
            print(f'  fold {i:>2} failed: {e}')

    ok = ~np.isnan(risk)
    cidx_cont = concordance_index(T[ok], -risk[ok], E[ok])
    boot2 = []
    idx2 = np.where(ok)[0]
    for _ in range(2000):
        s = rng.choice(idx2, size=len(idx2), replace=True)
        if E[s].sum() < 2:
            continue
        try:
            boot2.append(concordance_index(T[s], -risk[s], E[s]))
        except Exception:
            continue
    ci_lo_c, ci_hi_c = np.percentile(boot2, [2.5, 97.5])

    print('\n' + '=' * 70)
    print('PART B. Continuous Cox on FU (LOOCV, penalised)')
    print('=' * 70)
    print(f"  Harrell's C: {cidx_cont:.3f}  (95% CI {ci_lo_c:.3f} to {ci_hi_c:.3f}, 2000 boot)")

    # =====================================================================
    # Save outputs
    # =====================================================================
    pd.DataFrame({
        'patient_id': pids,
        'actual': y,
        'predicted': pred_fu,
        'prob_nonresponder_fu': prob_fu,
        'cox_risk_score_fu': risk,
        'PFS Months': T,
        'pfs_event': E,
    }).to_csv(OUT_PERPAT, index=False)

    pd.DataFrame([{
        'cohort': 'FU-alone',
        'n': len(pids),
        'accuracy': acc, 'auc': auc,
        'sensitivity': sens, 'specificity': spec,
        'ppv': ppv, 'npv': npv,
        'fisher_p': fisher_p,
        'hr_binary': hr_bin, 'hr_binary_lo': hr_bin_lo, 'hr_binary_hi': hr_bin_hi, 'hr_binary_p': p_bin,
        'hr_per_10pp': hr_10, 'hr_per_10pp_lo': hr_10_lo, 'hr_per_10pp_hi': hr_10_hi, 'hr_per_10pp_p': p_10,
        'cindex_binary_prob': cidx_bin, 'cindex_binary_prob_lo': ci_lo_bin, 'cindex_binary_prob_hi': ci_hi_bin,
        'cindex_continuous_cox': cidx_cont, 'cindex_continuous_cox_lo': ci_lo_c, 'cindex_continuous_cox_hi': ci_hi_c,
    }]).to_csv(OUT_BATTERY, index=False)

    print(f'\nWrote: {OUT_PERPAT}')
    print(f'Wrote: {OUT_BATTERY}')

    # =====================================================================
    # Side-by-side summary block
    # =====================================================================
    print('\n' + '=' * 70)
    print('FU vs BASELINE side-by-side')
    print('=' * 70)
    print(f"""
                     Baseline (LOOCV)    FU-alone (LOOCV)
Accuracy             0.643               {acc:.3f}
AUC                  0.691               {auc:.3f}
Sensitivity          0.368               {sens:.3f}
Specificity          0.870               {spec:.3f}
PPV                  0.636               {ppv:.3f}
NPV                  0.667               {npv:.3f}
Fisher exact p       0.182               {fisher_p:.4f}
Cox HR binary        1.29 (0.58-2.88)    {hr_bin:.2f} ({hr_bin_lo:.2f}-{hr_bin_hi:.2f})
Cox HR per 10pp      1.50 (0.78-2.90)    {hr_10:.2f} ({hr_10_lo:.2f}-{hr_10_hi:.2f})
C-index (binary)     0.617               {cidx_bin:.3f}
C-index (cont Cox)   0.700 (0.594-0.791) {cidx_cont:.3f} ({ci_lo_c:.3f}-{ci_hi_c:.3f})
""")


if __name__ == '__main__':
    main()
