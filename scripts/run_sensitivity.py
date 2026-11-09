#!/usr/bin/env python3
"""
BRAND 36-DMR panel and comparison models.

Runs three LOOCVs on the same 42 patients:

  1. 36-DMR-only        - the stable DMRs (found in all 42 folds of run_loocv.py)
  2. Clinical-only      - PD-L1, ECOG, smoking, age, gender, histology
  3. Combined           - 36-DMR + clinical

Inputs (all in this directory):
  results/dmr_stability.csv         # from main run
  results/loocv_results_final.csv   # patient order, response label, PFS
  metilene_full_matrix_filtered.bed # CpG x patient matrix
  loocv42_clinical.csv              # extracted from TWISTmethylomepatients.xlsx

Outputs:
  results/sensitivity_36dmr.csv
  results/sensitivity_clinical.csv
  results/sensitivity_combined.csv
  Prints accuracy, AUC, Cox HR (binary + continuous) per model.

Runtime: ~3-5 min (most of it streaming the 4 GB matrix once).
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

SCRIPT_DIR      = Path(__file__).resolve().parent
RESULTS         = SCRIPT_DIR / 'results'
STABILITY_CSV   = RESULTS / 'dmr_stability.csv'
LOOCV_FINAL_CSV = RESULTS / 'loocv_results_final.csv'
MATRIX          = SCRIPT_DIR / 'metilene_full_matrix_filtered.bed'
CLINICAL_CSV    = SCRIPT_DIR / 'loocv42_clinical.csv'


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_region(rid):
    chrom, span = rid.split(':')
    s, e = span.split('-')
    return chrom, int(s), int(e)


def load_36_regions():
    stab = pd.read_csv(STABILITY_CSV)
    rows = stab[stab['fold_count'] == 42]
    assert len(rows) == 36, f'expected 36 always-recurring DMRs, got {len(rows)}'
    return [parse_region(r) for r in rows['dmr']]


def read_matrix_header():
    with open(MATRIX) as f:
        hdr = f.readline().strip().split('\t')
    raw_cols = hdr[2:]                                  # resp_BHxx / nonresp_SJHyy
    pids = [c.split('_', 1)[1] for c in raw_cols]       # strip prefix
    return hdr, raw_cols, pids


def extract_dmr_matrix(regions, raw_cols, hdr_pids):
    """Stream the 4 GB bed once, keep only CpGs falling in any of the 36
    regions, then aggregate to mean methylation per region per patient.
    Returns (n_regions, n_pids_in_bed) np.ndarray."""
    print(f'\nStreaming {MATRIX.name} (filtering to {len(regions)} regions)...')
    by_chr = {}
    for c, s, e in regions:
        by_chr.setdefault(c, []).append((s, e))

    kept_chunks = []
    n_scanned = 0
    for chunk in pd.read_csv(MATRIX, sep='\t', chunksize=200_000, low_memory=False):
        n_scanned += len(chunk)
        sub = chunk[chunk['chr'].isin(by_chr.keys())]
        if len(sub) == 0:
            continue
        mask = np.zeros(len(sub), dtype=bool)
        for c, spans in by_chr.items():
            in_c = (sub['chr'].values == c)
            for s, e in spans:
                mask |= in_c & (sub['pos'].values >= s) & (sub['pos'].values < e)
        if mask.any():
            kept_chunks.append(sub[mask])
        if n_scanned % 2_000_000 == 0:
            kept_so_far = sum(len(k) for k in kept_chunks)
            print(f'  scanned {n_scanned:>10,} rows  |  kept {kept_so_far:,}')

    df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()
    print(f'  total scanned: {n_scanned:,}, kept {len(df):,} CpGs across {len(regions)} regions')

    mat = np.full((len(regions), len(hdr_pids)), np.nan)
    for i, (c, s, e) in enumerate(regions):
        m = (df['chr'] == c) & (df['pos'] >= s) & (df['pos'] < e)
        if m.sum() > 0:
            mat[i, :] = df.loc[m, raw_cols].mean(axis=0).values
    return mat


def build_clinical_features(clinical, patient_order):
    """Returns (42, n_features) array + column names, aligned to patient_order.
    Imputation deferred to fold (no leakage)."""
    c = clinical.set_index('patient_id').loc[patient_order].copy()

    feats = pd.DataFrame(index=c.index)
    feats['age']        = pd.to_numeric(c['Age at diagnosis'], errors='coerce')
    feats['ecog']       = pd.to_numeric(c['ECOG at diagnosis'], errors='coerce')
    feats['pdl1_range'] = pd.to_numeric(
        c['PDL1 Range (1=<1; 2 = 1-49; 3=>50)'], errors='coerce'
    )

    smoke_map = {
        'never': 0, 'never smoker': 0, 'non-smoker': 0,
        'former': 1, 'ex-smoker': 1, 'ex': 1, 'former smoker': 1,
        'current': 2, 'current smoker': 2, 'smoker': 2,
    }
    feats['smoking']  = c['Smoking status'].astype(str).str.strip().str.lower().map(smoke_map)
    feats['gender_f'] = c['Gender'].astype(str).str.upper().str.startswith('F').astype(float)

    h = c['Histopathological subtype'].astype(str).str.lower()
    feats['hist_adeno']    = h.str.contains('adeno').astype(float)
    feats['hist_squamous'] = h.str.contains('squam').astype(float)

    return feats.values, list(feats.columns)


def loocv_predict(X, y, C=0.1):
    """L1 logistic LOOCV with fold-internal median impute + standard scale.
    Main-run protocol is C=0.1; weaker C (e.g. 1.0) leaves more signal for
    small-feature models where C=0.1 collapses to a constant predictor."""
    n = len(y)
    preds, probs = np.empty(n, dtype=int), np.empty(n)
    for i in range(n):
        X_tr = np.delete(X, i, axis=0).astype(float).copy()
        X_te = X[i:i+1].astype(float).copy()
        y_tr = np.delete(y, i)

        col_med = np.nanmedian(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = col_med[j]
            X_te[np.isnan(X_te[:, j]), j] = col_med[j]
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X_tr)
        Xte_s = sc.transform(X_te)

        clf = LogisticRegression(penalty='l1', solver='saga', C=C,
                                 max_iter=5000, random_state=42)
        clf.fit(Xtr_s, y_tr)
        preds[i] = int(clf.predict(Xte_s)[0])
        probs[i] = float(clf.predict_proba(Xte_s)[0, 1])
    return preds, probs


def bootstrap_auc_ci(y, prob, n_iter=2000, seed=42):
    """Percentile-bootstrap 95% CI for AUC over the 42 predictions.
    Returns (auc_point, lo, hi). Resamples patients with replacement;
    iterations with only one class are skipped."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yb, pb))
        except Exception:
            continue
    aucs = np.array(aucs)
    try:
        point = roc_auc_score(y, prob)
    except Exception:
        point = float('nan')
    if len(aucs) == 0:
        return point, float('nan'), float('nan')
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, lo, hi


def cox_block(predicted, prob, pfs_months, pfs_event):
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print('  (lifelines not installed - skipping Cox)')
        return

    df = pd.DataFrame({
        'PFS Months': pfs_months,
        'pfs_event': pfs_event,
        'predicted_NR': predicted,
        'prob_NR': prob,
    }).dropna()

    for name, col in [('binary', 'predicted_NR'), ('continuous', 'prob_NR')]:
        try:
            cph = CoxPHFitter()
            cph.fit(df[['PFS Months', 'pfs_event', col]],
                    duration_col='PFS Months', event_col='pfs_event')
            hr = float(np.exp(cph.params_[col]))
            ci = np.exp(cph.confidence_intervals_.values[0])
            p = float(cph.summary['p'][col])
            print(f'  Cox HR ({name:>10}): {hr:>7.2f}  '
                  f'(95% CI {ci[0]:.2f}-{ci[1]:.2f}), p={p:.3f}')
        except Exception as e:
            print(f'  Cox HR ({name:>10}): failed - {e}')


def report(name, y, pred, prob, pfs_months, pfs_event, save_path, patient_ids):
    print('\n' + '=' * 70)
    print(f'MODEL: {name}')
    print('=' * 70)
    acc = accuracy_score(y, pred)
    auc, lo, hi = bootstrap_auc_ci(y, prob)
    print(f'  Accuracy: {int((pred == y).sum())}/{len(y)} = {acc:.3f}')
    print(f'  AUC:      {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f}, 2000 bootstraps)')
    cm = confusion_matrix(y, pred)
    print('  Confusion matrix              Predicted R    Predicted NR')
    print(f'    Actual R:                    {cm[0,0]:>5}          {cm[0,1]:>5}')
    print(f'    Actual NR:                   {cm[1,0]:>5}          {cm[1,1]:>5}')
    cox_block(pred, prob, pfs_months, pfs_event)

    pd.DataFrame({
        'patient_id': patient_ids,
        'actual': y,
        'predicted': pred,
        'prob_nonresponder': prob,
        'PFS Months': pfs_months,
        'pfs_event': pfs_event,
    }).to_csv(save_path, index=False)
    print(f'  Saved: {save_path}')


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Patient order, response labels, PFS - take from the main run for parity
    loocv = pd.read_csv(LOOCV_FINAL_CSV)
    patient_order = loocv['patient_id'].tolist()
    actual = loocv['actual'].values.astype(int)
    pfs_months = loocv['PFS Months'].values.astype(float)
    # The 'PFS' column from the source xlsx is a numeric 1/0 event indicator
    # (1 = progressed, 0 = censored).
    pfs_event = loocv['PFS'].astype(float).astype(int).values
    assert len(patient_order) == 42, f'expected 42 patients, got {len(patient_order)}'
    print(f'Loaded {len(patient_order)} patients from main LOOCV results')
    print(f'  responders:    {(actual == 0).sum()}')
    print(f'  nonresponders: {(actual == 1).sum()}')

    # Clinical features
    clinical = pd.read_csv(CLINICAL_CSV)
    missing = set(patient_order) - set(clinical['patient_id'])
    assert not missing, f'patients missing from clinical CSV: {missing}'
    X_clin, clin_cols = build_clinical_features(clinical, patient_order)
    print(f'\nClinical features ({len(clin_cols)}): {clin_cols}')
    print(f'  NaN fraction: {np.isnan(X_clin.astype(float)).mean():.3f}')

    # 36-DMR methylation features
    regions = load_36_regions()
    print(f'\nLoaded {len(regions)} always-recurring DMRs from dmr_stability.csv')
    _, raw_cols, hdr_pids = read_matrix_header()
    print(f'Bed file has {len(hdr_pids)} patient columns')
    missing_bed = set(patient_order) - set(hdr_pids)
    assert not missing_bed, f'patients missing from bed header: {missing_bed}'

    mat_full = extract_dmr_matrix(regions, raw_cols, hdr_pids)
    pid_idx = [hdr_pids.index(p) for p in patient_order]
    X_dmr = mat_full[:, pid_idx].T   # (42, 36)
    print(f'Methylation feature matrix: {X_dmr.shape}')
    print(f'  NaN fraction: {np.isnan(X_dmr).mean():.3f}')

    # Dump the 42x36 matrix so the continuous / per-patient follow-up analyses
    # can run locally without re-streaming the 4 GB bed file.
    region_names = [f'{c}:{s}-{e}' for c, s, e in regions]
    pd.DataFrame(X_dmr, index=patient_order, columns=region_names).to_csv(
        RESULTS / 'dmr36_matrix.csv', index_label='patient_id')
    print(f'  Saved: {RESULTS / "dmr36_matrix.csv"}')

    # --- Models. C=0.1 for methylation and combined, as in run_loocv.py;
    # C=1.0 for clinical, because with only 7 features L1 at C=0.1 sets every
    # coefficient to zero and gives a constant predictor. ---
    pred, prob = loocv_predict(X_dmr, actual, C=0.1)
    report('36-DMR-only (L1, C=0.1)',
           actual, pred, prob, pfs_months, pfs_event,
           RESULTS / 'sensitivity_36dmr.csv', patient_order)

    pred, prob = loocv_predict(X_clin, actual, C=1.0)
    report('Clinical-only (age, ECOG, PDL1, smoking, gender, histology; L1, C=1.0)',
           actual, pred, prob, pfs_months, pfs_event,
           RESULTS / 'sensitivity_clinical.csv', patient_order)

    X_comb = np.concatenate([X_dmr, X_clin], axis=1)
    pred, prob = loocv_predict(X_comb, actual, C=0.1)
    report('Combined: 36-DMR + clinical (L1, C=0.1)',
           actual, pred, prob, pfs_months, pfs_event,
           RESULTS / 'sensitivity_combined.csv', patient_order)

    print('\n' + '=' * 70)
    print('Done. Three sensitivity CSVs written to results/.')
    print('=' * 70)


if __name__ == '__main__':
    main()
