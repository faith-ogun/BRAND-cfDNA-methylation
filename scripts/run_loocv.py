#!/usr/bin/env python3
"""
BRAND DMR Classifier - Full LOOCV Pipeline (standalone for SLURM / HPC)

Runs everything end-to-end:
  1. Clean bedgraphs (skips if already done)
  2. Merge into CpG x Patient matrix
  3. Write filtered metilene input (>=90% coverage, skips if exists)
  4. LOOCV: 42 folds of metilene + logistic regression
  5. Results summary + classification report
  6. DMR stability analysis across folds
  7. Cox proportional hazards

Usage:
    python run_loocv.py                     # run all 42 folds
    python run_loocv.py --start 0 --end 14  # run folds 0-13 only
    python run_loocv.py --skip-to-results   # skip LOOCV, just run analysis on existing results

SLURM (one fold per array task; fold 0 first, as it writes the filtered matrix):
    sbatch submit_loocv_array.sh                   # fold 0
    sbatch --array=1-41%5 submit_loocv_array.sh    # remaining folds
"""

import os
import sys
import gzip
import shutil
import subprocess
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR HPC ENVIRONMENT
# =============================================================================

# Everything lives in the same directory as this script.
# Layout:
#   scripts/
#     run_loocv.py                          <-- this script
#     submit_loocv_array.sh                 <-- SLURM array job, one fold per task
#     BRAND_merged_analysis_dataset.csv     <-- clinical metadata
#     sample_mapping.csv                    <-- patient-to-sample mapping
#     metilene_v0.2-9/                      <-- metilene source + binary
#     cleaned_bedgraphs/                    <-- 42 cleaned bedgraph files
#     metilene_full_matrix_filtered.bed     <-- pre-filtered CpG matrix (optional)
#     results/                              <-- output goes here (created automatically)

SCRIPT_DIR = Path(__file__).resolve().parent

BEDGRAPH_DIR   = SCRIPT_DIR / 'cleaned_bedgraphs'  # not used if cleaned files exist
CLINICAL_CSV   = SCRIPT_DIR / 'BRAND_merged_analysis_dataset.csv'
SAMPLE_MAP_CSV = SCRIPT_DIR / 'sample_mapping.csv'

OUTPUT_DIR  = SCRIPT_DIR / 'results'
CLEANED_DIR = SCRIPT_DIR / 'cleaned_bedgraphs'
LOOCV_DIR   = OUTPUT_DIR / 'loocv_folds'

METILENE_BIN = str(SCRIPT_DIR / 'metilene_v0.2-9' / 'metilene')

PFS_THRESHOLD    = 6
CANONICAL_CHROMS = {f'chr{i}' for i in range(1, 23)}
EXCLUDED_PATIENTS = ['SJH15']

MIN_COVERAGE_FRAC = 0.90

RESULTS_CSV = OUTPUT_DIR / 'loocv_results_incremental.csv'
FULL_MATRIX = SCRIPT_DIR / 'metilene_full_matrix_filtered.bed'


# =============================================================================
# FUNCTIONS
# =============================================================================

def clean_bedgraph(input_path, output_path):
    open_fn = gzip.open if str(input_path).endswith('.gz') else open
    mode = 'rt' if str(input_path).endswith('.gz') else 'r'
    kept = 0
    skipped = 0
    with open_fn(input_path, mode) as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith('track') or line.startswith('#'):
                skipped += 1
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                skipped += 1
                continue
            if parts[0] not in CANONICAL_CHROMS:
                skipped += 1
                continue
            fout.write(f'{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\n')
            kept += 1
    return kept, skipped


def load_bedgraph(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None,
                     names=['chr', 'pos', 'end', 'meth'],
                     dtype={'chr': str, 'pos': int, 'end': int, 'meth': float})
    return df[['chr', 'pos', 'meth']]


def prewrite_full_metilene_matrix(merged_df, patient_order, response_labels,
                                  output_path, min_cov_frac=MIN_COVERAGE_FRAC,
                                  chunk_size=100_000):
    resp_pids    = [p for p, l in zip(patient_order, response_labels) if l == 0]
    nonresp_pids = [p for p, l in zip(patient_order, response_labels) if l == 1]
    ordered = resp_pids + nonresp_pids
    columns = ['chr', 'pos'] + ordered

    header = ['chr', 'pos']
    header += [f'resp_{p}' for p in resp_pids]
    header += [f'nonresp_{p}' for p in nonresp_pids]

    min_patients = int(np.ceil(min_cov_frac * len(ordered)))
    total_kept = 0
    total_dropped = 0

    with open(output_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for start in range(0, len(merged_df), chunk_size):
            chunk = merged_df[columns].iloc[start:start + chunk_size].copy()
            coverage = chunk[ordered].notna().sum(axis=1)
            chunk = chunk[coverage >= min_patients]
            total_kept += len(chunk)
            total_dropped += chunk_size - len(chunk)
            if len(chunk) > 0:
                chunk['pos'] = chunk['pos'].astype(int)
                chunk.to_csv(f, sep='\t', header=False, index=False,
                             na_rep='NA', float_format='%.4f')
            del chunk

    print(f'  Kept {total_kept:,} CpGs, dropped {total_dropped:,} '
          f'(>={min_cov_frac*100:.0f}% coverage filter)')
    return header


def write_fold_input_fast(full_matrix_path, fold_output_path, header, held_out_pid):
    drop_col = None
    for i, name in enumerate(header):
        if name.endswith(f'_{held_out_pid}'):
            drop_col = i + 1
            break
    if drop_col is None:
        raise ValueError(f'{held_out_pid} not found in header')
    n_cols = len(header)
    keep = [str(i) for i in range(1, n_cols + 1) if i != drop_col]
    cmd = f"cut -f{','.join(keep)} '{full_matrix_path}' > '{fold_output_path}'"
    subprocess.run(cmd, shell=True, check=True)


def run_metilene(input_path, group1_prefix='resp', group2_prefix='nonresp'):
    cmd = [
        METILENE_BIN,
        '-a', group1_prefix,
        '-b', group2_prefix,
        '-m', '5',
        '-d', '0.1',
        '-f', '1',
        '-t', '8',
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    if result.returncode != 0:
        print(f'    Metilene error: {result.stderr[:200]}')
        return pd.DataFrame()
    rows = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 10:
            qval = float(parts[3])
            if qval < 0.05:
                rows.append({
                    'chr': parts[0], 'start': int(parts[1]), 'end': int(parts[2]),
                    'qval': qval, 'mean_meth_diff': float(parts[4]),
                    'num_cpgs': int(parts[5]), 'p_mwu': float(parts[6]),
                    'p_2dks': float(parts[7]), 'mean_g1': float(parts[8]),
                    'mean_g2': float(parts[9])
                })
        elif len(parts) >= 8:
            qval = float(parts[3])
            if qval < 0.05:
                rows.append({
                    'chr': parts[0], 'start': int(parts[1]), 'end': int(parts[2]),
                    'qval': qval, 'mean_meth_diff': float(parts[4]),
                    'num_cpgs': int(parts[5]),
                    'mean_g1': float(parts[6]), 'mean_g2': float(parts[7])
                })
    return pd.DataFrame(rows)


def extract_dmr_values(merged_df, dmr_df, patient_columns):
    matrix = np.full((len(dmr_df), len(patient_columns)), np.nan)
    for i, (_, dmr) in enumerate(dmr_df.iterrows()):
        mask = (
            (merged_df['chr'] == dmr['chr']) &
            (merged_df['pos'] >= dmr['start']) &
            (merged_df['pos'] < dmr['end'])
        )
        region = merged_df.loc[mask, patient_columns]
        if len(region) > 0:
            matrix[i, :] = region.mean(axis=0).values
    return matrix


# =============================================================================
# DOWNSTREAM ANALYSIS (runs after all 42 folds complete)
# =============================================================================

def run_results_summary(results_df):
    """Print per-patient results table and classification metrics."""
    print('\n' + '=' * 90)
    print('PER-PATIENT RESULTS')
    print('=' * 90)
    print(f'{"Patient":>8}  {"Actual":>6}  {"Predicted":>9}  {"P(NR)":>7}  {"PFS(mo)":>8}  {"DMRs":>5}  {"":>3}')
    print('-' * 90)

    correct = 0
    for _, r in results_df.iterrows():
        match = '+' if r['actual'] == r['predicted'] else '-'
        if r['actual'] == r['predicted']:
            correct += 1
        a = 'NR' if r['actual'] == 1 else 'R'
        p = 'NR' if r['predicted'] == 1 else 'R'
        print(f"{r['patient_id']:>8}  {a:>6}  {p:>9}  {r['prob_nonresponder']:>7.3f}  "
              f"{r['PFS Months']:>8.1f}  {r['n_dmrs']:>5}  {match:>3}")

    print('=' * 90)
    acc = correct / len(results_df)
    print(f'\nAccuracy: {correct}/{len(results_df)} = {acc:.3f}')

    try:
        auc = roc_auc_score(results_df['actual'], results_df['prob_nonresponder'])
        print(f'AUC: {auc:.3f}')
    except Exception:
        print('AUC: could not compute')

    cm = confusion_matrix(results_df['actual'], results_df['predicted'])
    print(f'\nConfusion matrix:')
    print(f'                  Predicted R    Predicted NR')
    print(f'  Actual R:        {cm[0,0]:>6}         {cm[0,1]:>6}')
    print(f'  Actual NR:       {cm[1,0]:>6}         {cm[1,1]:>6}')

    print(f'\n{classification_report(results_df["actual"], results_df["predicted"], target_names=["Responder", "Non-responder"])}')


def run_dmr_stability():
    """Analyse DMR consistency across LOOCV folds."""
    print('\n' + '=' * 60)
    print('DMR STABILITY ACROSS FOLDS')
    print('=' * 60)

    fold_dmr_counts = []
    all_fold_dmrs = []

    for fold_dir in sorted(LOOCV_DIR.glob('fold_*')):
        dmr_file = fold_dir / 'dmrs.csv'
        if dmr_file.exists():
            fd = pd.read_csv(dmr_file)
            fold_dmr_counts.append(len(fd))
            for _, d in fd.iterrows():
                all_fold_dmrs.append(f"{d['chr']}:{d['start']}-{d['end']}")

    if not fold_dmr_counts:
        print('No fold DMR files found.')
        return

    print(f'\nDMRs per fold:')
    print(f'  Mean:   {np.mean(fold_dmr_counts):.0f}')
    print(f'  Median: {np.median(fold_dmr_counts):.0f}')
    print(f'  Min:    {np.min(fold_dmr_counts)}')
    print(f'  Max:    {np.max(fold_dmr_counts)}')

    dmr_freq = Counter(all_fold_dmrs)
    n_folds = len(fold_dmr_counts)

    n_all  = sum(1 for v in dmr_freq.values() if v == n_folds)
    n_most = sum(1 for v in dmr_freq.values() if v >= n_folds * 0.9)
    n_half = sum(1 for v in dmr_freq.values() if v >= n_folds * 0.5)

    print(f'\nDMR recurrence:')
    print(f'  Total unique DMRs across all folds: {len(dmr_freq)}')
    print(f'  In ALL {n_folds} folds:             {n_all}')
    print(f'  In >=90% of folds:                  {n_most}')
    print(f'  In >=50% of folds:                  {n_half}')

    # Save stability summary
    stability_df = pd.DataFrame([
        {'dmr': k, 'fold_count': v, 'fold_fraction': v / n_folds}
        for k, v in dmr_freq.most_common()
    ])
    stability_df.to_csv(OUTPUT_DIR / 'dmr_stability.csv', index=False)
    print(f'\nSaved: {OUTPUT_DIR / "dmr_stability.csv"}')


def run_cox_regression(results_df):
    """Cox proportional hazards - predicted response vs PFS."""
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print('\nSkipping Cox regression - lifelines not installed.')
        print('  Install with: pip install lifelines')
        return

    print('\n' + '=' * 60)
    print('COX PROPORTIONAL HAZARDS')
    print('=' * 60)

    cox_df = results_df[['PFS Months', 'prob_nonresponder']].copy()
    cox_df['predicted_NR'] = results_df['predicted'].values

    # PFS event indicator. The 'PFS' column from the source xlsx is a numeric
    # 1/0 indicator (1 = progressed, 0 = censored).
    cox_df['pfs_event'] = pd.to_numeric(results_df['PFS'], errors='coerce').astype('Int64')
    cox_df = cox_df.dropna()
    cox_df['pfs_event'] = cox_df['pfs_event'].astype(int)

    print('\n--- Binary predictor (predicted NR=1 vs R=0) ---\n')
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df[['PFS Months', 'pfs_event', 'predicted_NR']],
                duration_col='PFS Months', event_col='pfs_event')
        cph.print_summary()
        hr = np.exp(cph.params_['predicted_NR'])
        ci = np.exp(cph.confidence_intervals_.values[0])
        p = cph.summary['p']['predicted_NR']
        print(f'\nHR = {hr:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f}), p = {p:.4f}')
    except Exception as e:
        print(f'Model failed: {e}')

    print('\n--- Continuous predictor (P(non-responder) score) ---\n')
    cph2 = CoxPHFitter()
    try:
        cph2.fit(cox_df[['PFS Months', 'pfs_event', 'prob_nonresponder']],
                 duration_col='PFS Months', event_col='pfs_event')
        cph2.print_summary()
    except Exception as e:
        print(f'Model failed: {e}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BRAND DMR Classifier LOOCV')
    parser.add_argument('--start', type=int, default=0,
                        help='First fold index (inclusive, default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='Last fold index (exclusive, default: all)')
    parser.add_argument('--skip-to-results', action='store_true',
                        help='Skip LOOCV, just run downstream analysis on existing results')
    args = parser.parse_args()

    for d in [OUTPUT_DIR, CLEANED_DIR, LOOCV_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- Verify metilene ---
    if not args.skip_to_results:
        if not (os.path.isfile(METILENE_BIN) and os.access(METILENE_BIN, os.X_OK)):
            print(f'ERROR: metilene not found at {METILENE_BIN}')
            print('  Compile with: cd metilene_v0.2-9 && make')
            sys.exit(1)

    print(f'Project:   {SCRIPT_DIR}')
    print(f'Output:    {OUTPUT_DIR}')
    if not args.skip_to_results:
        print(f'Metilene:  {METILENE_BIN}')

    # --- Load clinical data ---
    print('\nLoading clinical data...')
    clinical = pd.read_csv(CLINICAL_CSV)
    clinical = clinical[~clinical['Patient_ID'].isin(EXCLUDED_PATIENTS)].reset_index(drop=True)
    clinical['response'] = clinical['PFS Months'].apply(
        lambda x: 'responder' if x >= PFS_THRESHOLD else 'nonresponder'
    )
    clinical['response_binary'] = (clinical['response'] == 'nonresponder').astype(int)

    sample_map = pd.read_csv(SAMPLE_MAP_CSV)
    baseline_map = sample_map[
        (sample_map['timepoint'] == 'baseline') &
        (~sample_map['patient_id'].isin(EXCLUDED_PATIENTS))
    ].copy()
    baseline_map = baseline_map.merge(
        clinical[['Patient_ID', 'response', 'response_binary', 'PFS Months',
                  'OS Months', 'PFS', 'OS']],
        left_on='patient_id', right_on='Patient_ID', how='left'
    ).drop(columns='Patient_ID')

    n_r  = (baseline_map['response'] == 'responder').sum()
    n_nr = (baseline_map['response'] == 'nonresponder').sum()
    print(f'Patients: {len(baseline_map)} ({n_r}R / {n_nr}NR)')

    # --- Skip-to-results shortcut ---
    if args.skip_to_results:
        if not RESULTS_CSV.exists():
            print(f'\nERROR: {RESULTS_CSV} not found. Run LOOCV first.')
            sys.exit(1)
        results_df = pd.read_csv(RESULTS_CSV)
        results_df = results_df.merge(
            baseline_map[['patient_id', 'PFS Months', 'OS Months', 'PFS', 'OS']],
            on='patient_id', how='left'
        )
        run_results_summary(results_df)
        run_dmr_stability()
        run_cox_regression(results_df)
        results_df.to_csv(OUTPUT_DIR / 'loocv_results_final.csv', index=False)
        print(f'\nFinal results: {OUTPUT_DIR / "loocv_results_final.csv"}')
        return

    # --- Clean bedgraphs ---
    print('\nCleaning bedgraph files...')
    for _, row in baseline_map.iterrows():
        pid = row['patient_id']
        snum = str(row['sample_num']).zfill(3)
        out_path = CLEANED_DIR / f'{pid}_baseline.bedgraph'
        if out_path.exists():
            continue
        found = None
        for pattern in [f'PN0634_{snum}*.bedGraph.gz', f'PN0634_{snum}*.bedGraph',
                        f'PN0634_{snum}*.bismark.cov.gz']:
            matches = glob(str(BEDGRAPH_DIR / pattern))
            if matches:
                found = matches[0]
                break
        if found is None:
            print(f'  WARNING: {pid} (PN0634_{snum}) raw file not found')
            continue
        kept, skipped = clean_bedgraph(found, out_path)
        print(f'  {pid}: {kept:,} CpGs kept')

    patients_with_data = []
    for _, row in baseline_map.iterrows():
        pid = row['patient_id']
        if (CLEANED_DIR / f'{pid}_baseline.bedgraph').exists():
            patients_with_data.append(pid)
    print(f'{len(patients_with_data)} patients with cleaned bedgraphs')

    # --- Merge ---
    print('\nMerging CpG matrix...')
    all_chunks = []
    for chrom_num in range(1, 23):
        chrom = f'chr{chrom_num}'
        chrom_data = {}
        for pid in patients_with_data:
            bg = load_bedgraph(CLEANED_DIR / f'{pid}_baseline.bedgraph')
            bg_chr = bg[bg['chr'] == chrom].set_index('pos')['meth']
            chrom_data[pid] = bg_chr
        merged_chr = pd.DataFrame(chrom_data)
        merged_chr.index.name = 'pos'
        merged_chr['chr'] = chrom
        all_chunks.append(merged_chr)
        cov = merged_chr.drop(columns='chr').notna().mean().mean() * 100
        print(f'  {chrom}: {len(merged_chr):>8,} CpGs, {cov:.1f}% coverage')

    merged = pd.concat(all_chunks).reset_index()
    print(f'\nFull matrix: {len(merged):,} CpGs x {len(patients_with_data)} patients')
    print(f'Memory: {merged.memory_usage(deep=True).sum() / 1e9:.2f} GB')

    # --- Prepare LOOCV ---
    patient_order = patients_with_data.copy()
    response_labels = baseline_map.set_index('patient_id').loc[patient_order, 'response_binary'].values

    if args.end is None:
        args.end = len(patient_order)

    # --- Load completed results (legacy CSV + per-fold CSVs) ---
    loocv_results = []
    done_pids = set()
    if RESULTS_CSV.exists():
        done_df = pd.read_csv(RESULTS_CSV)
        loocv_results = done_df.to_dict('records')
        done_pids = set(done_df['patient_id'])
    for fold_csv in sorted(OUTPUT_DIR.glob('fold_*.csv')):
        fdf = pd.read_csv(fold_csv)
        for rec in fdf.to_dict('records'):
            if rec['patient_id'] not in done_pids:
                loocv_results.append(rec)
                done_pids.add(rec['patient_id'])
    if done_pids:
        print(f'\nResuming: {len(done_pids)} folds already completed.')

    # --- Prewrite filtered matrix ---
    if FULL_MATRIX.exists():
        print(f'\nFiltered matrix exists: {FULL_MATRIX}')
        with open(FULL_MATRIX) as f:
            full_header = f.readline().strip().split('\t')
    else:
        print(f'\nWriting filtered matrix (>={MIN_COVERAGE_FRAC*100:.0f}% coverage)...')
        write_start = time.time()
        full_header = prewrite_full_metilene_matrix(
            merged, patient_order, response_labels, FULL_MATRIX
        )
        print(f'Done in {(time.time()-write_start)/60:.1f} min')

    # --- LOOCV loop ---
    print(f'\nStarting LOOCV: folds {args.start}-{args.end-1}')
    print('=' * 70)

    loocv_start = time.time()

    for fold_idx in range(args.start, args.end):
        held_out_pid = patient_order[fold_idx]

        if held_out_pid in done_pids:
            print(f'Fold {fold_idx+1:>2}/{len(patient_order)}: {held_out_pid} - skipping (done)')
            continue

        fold_start = time.time()
        held_out_label = response_labels[fold_idx]
        actual_str = 'NR' if held_out_label == 1 else 'R'
        print(f'Fold {fold_idx+1:>2}/{len(patient_order)}: holding out {held_out_pid} (actual: {actual_str})')

        train_pids = [p for p in patient_order if p != held_out_pid]
        train_labels = np.array([response_labels[patient_order.index(p)] for p in train_pids])
        resp_pids    = [p for p, l in zip(train_pids, train_labels) if l == 0]
        nonresp_pids = [p for p, l in zip(train_pids, train_labels) if l == 1]
        ordered_train = resp_pids + nonresp_pids

        fold_dir = LOOCV_DIR / f'fold_{fold_idx:02d}_{held_out_pid}'
        os.makedirs(fold_dir, exist_ok=True)
        input_file = fold_dir / 'metilene_input.bed'

        print(f'  Dropping {held_out_pid} column...', end=' ', flush=True)
        write_fold_input_fast(FULL_MATRIX, input_file, full_header, held_out_pid)
        print('done')

        print(f'  Running Metilene...', end=' ', flush=True)
        fold_dmrs = run_metilene(input_file)
        input_file.unlink(missing_ok=True)
        print(f'{len(fold_dmrs)} DMRs (q<0.05)')

        if len(fold_dmrs) == 0:
            print(f'  No DMRs - assigning P(NR)=0.5')
            loocv_results.append({
                'patient_id': held_out_pid, 'actual': held_out_label,
                'prob_nonresponder': 0.5, 'predicted': 0, 'n_dmrs': 0
            })
            pd.DataFrame([loocv_results[-1]]).to_csv(
                OUTPUT_DIR / f'fold_{fold_idx:02d}_{held_out_pid}.csv', index=False
            )
            continue

        fold_dmrs.to_csv(fold_dir / 'dmrs.csv', index=False)

        all_pids_this_fold = ordered_train + [held_out_pid]
        dmr_values = extract_dmr_values(merged, fold_dmrs, all_pids_this_fold)

        X_train = dmr_values[:, :len(ordered_train)].T
        X_test  = dmr_values[:, -1:].T
        y_train = np.array([0]*len(resp_pids) + [1]*len(nonresp_pids))

        col_medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[np.isnan(X_train[:, j]), j] = col_medians[j]
            X_test[np.isnan(X_test[:, j]), j]   = col_medians[j]
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=5000, random_state=42)
        clf.fit(X_train_s, y_train)

        pred = clf.predict(X_test_s)[0]
        prob = clf.predict_proba(X_test_s)[0, 1]

        pred_str = 'NR' if pred == 1 else 'R'
        match = 'OK' if pred == held_out_label else 'MISS'
        print(f'  Prediction: {pred_str} (P(NR)={prob:.3f}) [{match}]')

        loocv_results.append({
            'patient_id': held_out_pid, 'actual': held_out_label,
            'prob_nonresponder': prob, 'predicted': pred, 'n_dmrs': len(fold_dmrs)
        })
        pd.DataFrame([loocv_results[-1]]).to_csv(
            OUTPUT_DIR / f'fold_{fold_idx:02d}_{held_out_pid}.csv', index=False
        )

        fold_elapsed = time.time() - fold_start
        remaining = fold_elapsed * (args.end - fold_idx - 1)
        print(f'  ({fold_elapsed:.0f}s, ~{remaining/60:.0f} min remaining)\n')

    total_elapsed = time.time() - loocv_start
    print('=' * 70)
    print(f'LOOCV done - {total_elapsed/60:.1f} min total')

    # --- Downstream analysis ---
    results_df = pd.DataFrame(loocv_results)
    results_df = results_df.merge(
        baseline_map[['patient_id', 'PFS Months', 'OS Months', 'PFS', 'OS']],
        on='patient_id', how='left'
    )

    if len(results_df) == len(patient_order):
        run_results_summary(results_df)
        run_dmr_stability()
        run_cox_regression(results_df)
        results_df.to_csv(OUTPUT_DIR / 'loocv_results_final.csv', index=False)
        print(f'\nFinal results: {OUTPUT_DIR / "loocv_results_final.csv"}')
    else:
        print(f'\nCompleted {len(results_df)}/{len(patient_order)} folds')
        print(f'Incremental results: {RESULTS_CSV}')
        print('Re-run to complete remaining folds (completed folds will be skipped)')


if __name__ == '__main__':
    main()
