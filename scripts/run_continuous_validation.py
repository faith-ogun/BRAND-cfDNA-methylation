#!/usr/bin/env python3
"""
Continuous-outcome validation of the 36-DMR panel.

The binary outcome (durable clinical benefit, PFS of 6 months or longer)
places patients with similar PFS on either side of the cut-off, for example
5.8 and 6.2 months. This script evaluates the panel against PFS as a
time-to-event outcome: LOOCV with a penalised Cox model on the 36 stable DMRs,
scored by Harrell's concordance index, the survival-analysis counterpart of
the AUC (0.5 is chance).

Metilene discovery needs two groups, so it uses the binary split; this step
uses time to event directly and handles censoring.

Runs locally. No HPC, no Metilene.

Inputs:
  results/sensitivity_analysis/loocv_results_final.csv  PFS months, PFS event
  results/sensitivity_analysis/dmr36_matrix.csv  42 x 36 methylation matrix

Output:
  results/sensitivity_analysis/continuous_validation.csv  per-patient risk scores
  prints concordance index + 95% bootstrap CI
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings('ignore')

SCRIPT_DIR   = Path(__file__).resolve().parent
REPO         = SCRIPT_DIR.parent
SENS         = REPO / 'results' / 'sensitivity_analysis'
LOOCV_FINAL  = SENS / 'loocv_results_final.csv'
DMR36_MATRIX = SENS / 'dmr36_matrix.csv'

# p ~ n (36 features, 41 training patients), so strong penalisation is needed.
# Fixed a priori; not tuned on the outcome.
PENALIZER = 0.5


def main():
    # --- outcome: PFS time + event indicator ---
    # The 'PFS' column IS the event indicator: 1 = progressed (event),
    # 0 = censored (progression-free at last follow-up).
    loocv = pd.read_csv(LOOCV_FINAL).set_index('patient_id')
    pfs_months = loocv['PFS Months'].astype(float)
    pfs_event = loocv['PFS'].astype(float).astype(int)

    # --- 36-DMR methylation matrix ---
    X = pd.read_csv(DMR36_MATRIX, index_col='patient_id')
    patient_order = list(X.index)
    assert len(patient_order) == 42, f'expected 42 patients, got {len(patient_order)}'

    T = pfs_months.loc[patient_order].values
    E = pfs_event.loc[patient_order].values.astype(int)
    Xv = X.values.astype(float)
    n, p = Xv.shape
    print(f'{n} patients, {p} DMR features')
    print(f'  events (progression): {int(E.sum())}   censored: {int((E == 0).sum())}')
    print(f'  Cox penalizer (ridge): {PENALIZER}')

    # --- LOOCV: penalised Cox, predict held-out risk score ---
    risk = np.full(n, np.nan)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr = Xv[tr].copy()
        Xte = Xv[i:i + 1].copy()

        med = np.nanmedian(Xtr, axis=0)
        for c in range(p):
            Xtr[np.isnan(Xtr[:, c]), c] = med[c]
            Xte[np.isnan(Xte[:, c]), c] = med[c]

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        df_tr = pd.DataFrame(Xtr_s, columns=[f'f{c}' for c in range(p)])
        df_tr['T'] = T[tr]
        df_tr['E'] = E[tr]

        cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=0.0)
        try:
            cph.fit(df_tr, duration_col='T', event_col='E')
            df_te = pd.DataFrame(Xte_s, columns=[f'f{c}' for c in range(p)])
            risk[i] = float(cph.predict_partial_hazard(df_te).values[0])
        except Exception as e:
            print(f'  fold {i:>2} ({patient_order[i]}) failed: {e}')

    ok = ~np.isnan(risk)
    print(f'\n{int(ok.sum())}/{n} folds produced a risk score')

    # --- concordance index ---
    # Higher partial hazard means shorter survival, so negate for concordance_index
    # (which expects higher score = longer survival).
    cidx = concordance_index(T[ok], -risk[ok], E[ok])

    rng = np.random.default_rng(42)
    idx = np.where(ok)[0]
    boot = []
    for _ in range(2000):
        s = rng.choice(idx, size=len(idx), replace=True)
        if E[s].sum() < 2:
            continue
        try:
            boot.append(concordance_index(T[s], -risk[s], E[s]))
        except Exception:
            continue
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print('\n' + '=' * 60)
    print('CONTINUOUS VALIDATION: 36-DMR panel, LOOCV penalised Cox')
    print('=' * 60)
    print(f'  Concordance index: {cidx:.3f}')
    print(f'  95% bootstrap CI:  {lo:.3f} to {hi:.3f}  ({len(boot)} resamples)')
    print(f'  Reference:         0.5 = chance')
    print('=' * 60)

    out = pd.DataFrame({
        'patient_id': patient_order,
        'PFS_Months': T,
        'pfs_event': E,
        'cox_risk_score': risk,
    })
    out.to_csv(SENS / 'continuous_validation.csv', index=False)
    print(f'\nSaved: {SENS / "continuous_validation.csv"}')


if __name__ == '__main__':
    main()
