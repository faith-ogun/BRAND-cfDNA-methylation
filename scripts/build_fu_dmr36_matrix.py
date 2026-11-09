"""Build the 42 x 36 follow-up methylation matrix for the locked panel.

Apples-to-apples with baseline. The baseline pipeline applied a >=90% cross-
patient coverage filter when building metilene_full_matrix_filtered.bed, so only
a defined subset of CpGs ended up in dmr36_matrix.csv. To make FU comparable we
lock that exact CpG set (extracted from the existing Metilene bed) and average
the same CpGs in each FU sample's bedGraph.

Pipeline:
  1. Stream metilene_full_matrix_filtered.bed, keep only rows whose (chr, pos)
     falls inside one of the 36 locked regions. This gives the locked CpG set
     and reproduces the baseline matrix exactly as a sanity check.
  2. For each FU sample bedGraph, filter to the locked CpG set, group by region,
     compute the mean methylation per region (NaN-aware).
  3. Write dmr36_matrix_fu.csv.

Inputs:
  scripts/metilene_full_matrix_filtered.bed        locked CpG x baseline-patient matrix
  database_files/bedgraph_files/*.bedGraph.gz     per-sample CpG bedgraphs
  database_files/metadata/sample_mapping.csv      patient + timepoint per file
  results/sensitivity_analysis/dmr_stability.csv  source of the 36 locked regions
  results/sensitivity_analysis/loocv_results_final.csv   42 analysis patients
  results/sensitivity_analysis/dmr36_matrix.csv          baseline ref for sanity

Outputs:
  results/sensitivity_analysis/dmr36_matrix_fu.csv               42 x 36, FU
  results/sensitivity_analysis/dmr36_matrix_baseline_rebuilt.csv 42 x 36, baseline rebuild
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
SENS       = REPO / 'results' / 'sensitivity_analysis'
DB         = REPO / 'database_files'
BG_DIR     = DB / 'bedgraph_files'

STABILITY_CSV = SENS / 'dmr_stability.csv'
LOOCV_FINAL   = SENS / 'loocv_results_final.csv'
BASELINE_REF  = SENS / 'dmr36_matrix.csv'
MAPPING_CSV   = DB / 'metadata' / 'sample_mapping.csv'
METILENE_BED  = SCRIPT_DIR / 'metilene_full_matrix_filtered.bed'

OUT_FU       = SENS / 'dmr36_matrix_fu.csv'
OUT_BASELINE = SENS / 'dmr36_matrix_baseline_rebuilt.csv'

TOLERANCE_PP = 0.5   # max acceptable abs diff vs reference, percentage points


def load_regions():
    stab = pd.read_csv(STABILITY_CSV)
    rows = stab[stab['fold_count'] == 42]
    assert len(rows) == 36, f'expected 36 locked DMRs, got {len(rows)}'
    regions = []
    for r in rows['dmr']:
        chrom, span = r.split(':')
        s, e = span.split('-')
        regions.append((chrom, int(s), int(e)))
    return regions


def load_sample_table():
    """42 baseline + 42 follow-up rows, with bedGraph file paths resolved."""
    analysis_pids = set(pd.read_csv(LOOCV_FINAL)['patient_id'].astype(str))
    assert len(analysis_pids) == 42

    mp = pd.read_csv(MAPPING_CSV)
    mp = mp[mp['patient_id'].isin(analysis_pids)].copy()
    mp = mp[mp['timepoint'].isin(['baseline', 'follow-up'])].copy()

    # filepath in mapping is relative to bismark_coverage; swap for bedGraph
    def to_bg(p):
        name = Path(p).name.replace('.bismark.cov.gz', '.bedGraph.gz')
        return BG_DIR / name
    mp['bg_path'] = mp['filepath'].apply(to_bg)

    missing = [p for p in mp['bg_path'] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} bedGraph files missing, first: {missing[0]}')

    expected = 84
    assert len(mp) == expected, f'expected {expected} rows, got {len(mp)}'
    return mp[['patient_id', 'timepoint', 'bg_path']].reset_index(drop=True)


def extract_locked_cpgs_per_region(regions):
    """Stream metilene_full_matrix_filtered.bed and return:
      locked_cpgs: dict {(chr, pos): (region_idx, region_str)}
      baseline_means: DataFrame indexed by baseline patient_id, one col per region
    The baseline_means reproduces dmr36_matrix.csv exactly when aggregation
    matches.
    """
    by_chr = {}
    for i, (c, s, e) in enumerate(regions):
        by_chr.setdefault(c, []).append((i, s, e))

    region_names = [f'{c}:{s}-{e}' for c, s, e in regions]

    print(f'Streaming {METILENE_BED.name} for the 36 locked regions...')
    locked = {}                            # (chr, pos) -> region_idx
    per_region_rows = [[] for _ in regions]  # collect rows for baseline mean

    header = None
    baseline_pids = None

    n_scanned = 0
    for chunk in pd.read_csv(METILENE_BED, sep='\t', chunksize=200_000, low_memory=False):
        if header is None:
            header = list(chunk.columns)
            # cols 2: are resp_<pid> / nonresp_<pid>
            raw_cols = header[2:]
            baseline_pids = [c.split('_', 1)[1] for c in raw_cols]
        n_scanned += len(chunk)
        spans_for_chr = chunk['chr'].map(by_chr)
        sub = chunk[spans_for_chr.notna()]
        if len(sub) == 0:
            continue
        for c, c_sub in sub.groupby('chr'):
            spans = by_chr[c]
            for ridx, s, e in spans:
                m = (c_sub['pos'].values >= s) & (c_sub['pos'].values < e)
                if m.any():
                    matching = c_sub[m]
                    for _, row in matching.iterrows():
                        locked[(c, int(row['pos']))] = ridx
                    per_region_rows[ridx].append(matching)

    raw_cols = header[2:]
    n_locked = len(locked)
    print(f'  scanned {n_scanned:,} rows; locked {n_locked} CpGs in the 36 regions')

    # baseline reproduction: per region, mean across CpGs per patient (NaN-aware)
    baseline_mat = np.full((len(regions), len(baseline_pids)), np.nan)
    for ridx in range(len(regions)):
        if not per_region_rows[ridx]:
            continue
        block = pd.concat(per_region_rows[ridx], ignore_index=True)
        vals = block[raw_cols].apply(pd.to_numeric, errors='coerce')
        baseline_mat[ridx, :] = vals.mean(axis=0, skipna=True).values

    baseline_df = pd.DataFrame(baseline_mat.T, index=baseline_pids, columns=region_names)
    baseline_df.index.name = 'patient_id'
    return locked, baseline_df


def aggregate_one_sample(bg_path: Path, locked: dict, regions) -> dict:
    """For one bedGraph, average methylation across the locked CpGs per region."""
    sums = {i: 0.0 for i in range(len(regions))}
    counts = {i: 0 for i in range(len(regions))}
    open_fn = gzip.open if str(bg_path).endswith('.gz') else open
    with open_fn(bg_path, 'rt') as fh:
        for line in fh:
            if line.startswith('track') or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            chrom = parts[0]
            try:
                pos = int(parts[1])
            except ValueError:
                continue
            ridx = locked.get((chrom, pos))
            if ridx is None:
                continue
            try:
                val = float(parts[3])
            except ValueError:
                continue
            sums[ridx] += val
            counts[ridx] += 1
    return {i: (sums[i] / counts[i] if counts[i] > 0 else float('nan'))
            for i in range(len(regions))}


def build_matrix_for(samples: pd.DataFrame, locked, regions) -> pd.DataFrame:
    region_names = [f'{c}:{s}-{e}' for c, s, e in regions]
    rows = []
    for i, row in enumerate(samples.itertuples(index=False), start=1):
        agg = aggregate_one_sample(row.bg_path, locked, regions)
        rows.append([agg[r] for r in range(len(regions))])
        if i % 10 == 0 or i == len(samples):
            print(f'  processed {i}/{len(samples)} samples')
    mat = pd.DataFrame(rows, index=samples['patient_id'].values, columns=region_names)
    mat.index.name = 'patient_id'
    return mat


def compare_to_reference(rebuilt: pd.DataFrame, reference_path: Path, label: str):
    ref = pd.read_csv(reference_path, index_col='patient_id')
    common = [p for p in ref.index if p in rebuilt.index]
    ref = ref.loc[common]
    rb  = rebuilt.loc[common, ref.columns]
    diff = (rb - ref).abs()
    max_abs = np.nanmax(diff.values) if diff.size else 0.0
    worst_region = diff.max(axis=0).idxmax()
    worst_patient = diff[worst_region].idxmax()
    print(f'\n[{label}] Sanity check vs {reference_path.name}:')
    print(f'  max abs diff (pp): {max_abs:.4f}')
    print(f'  worst cell: patient={worst_patient}, region={worst_region}, '
          f'ref={ref.loc[worst_patient, worst_region]:.4f}, '
          f'rebuilt={rb.loc[worst_patient, worst_region]:.4f}')
    if max_abs > TOLERANCE_PP:
        bad = diff.stack().sort_values(ascending=False).head(10)
        print('  TOP-10 divergences:'); print(bad.to_string())
        raise AssertionError(
            f'{label} rebuild diverges by > {TOLERANCE_PP} pp (max {max_abs:.4f})')
    print(f'  WITHIN TOLERANCE ({TOLERANCE_PP} pp).')


def main():
    regions = load_regions()
    print(f'Loaded {len(regions)} locked DMRs')

    samples = load_sample_table()
    print(f'Sample table: {len(samples)} rows '
          f'({(samples.timepoint == "baseline").sum()} baseline, '
          f'{(samples.timepoint == "follow-up").sum()} follow-up)')

    # Step 1: extract locked CpG set + reproduce baseline matrix from the Metilene bed
    locked, baseline_from_bed = extract_locked_cpgs_per_region(regions)

    # Sanity check 1: bed-derived baseline matches the existing reference exactly
    compare_to_reference(baseline_from_bed, BASELINE_REF, 'baseline-from-bed')

    # Step 2: rebuild baseline from bedGraphs using the locked CpG set
    # (proves the locked-CpG bedGraph aggregation also matches the reference)
    baseline_samples = samples[samples['timepoint'] == 'baseline'].reset_index(drop=True)
    print('\nRebuilding baseline matrix from bedGraphs using locked CpGs...')
    baseline_from_bg = build_matrix_for(baseline_samples, locked, regions)
    compare_to_reference(baseline_from_bg, BASELINE_REF, 'baseline-from-bedGraph')

    # Step 3: FU matrix, same locked CpGs
    fu_samples = samples[samples['timepoint'] == 'follow-up'].reset_index(drop=True)
    print('\nBuilding follow-up matrix from bedGraphs using locked CpGs...')
    fu_mat = build_matrix_for(fu_samples, locked, regions)

    baseline_from_bg.to_csv(OUT_BASELINE)
    fu_mat.to_csv(OUT_FU)
    print(f'\nWrote: {OUT_BASELINE}')
    print(f'Wrote: {OUT_FU}')

    print('\nFU matrix summary:')
    print(f'  shape: {fu_mat.shape}')
    print(f'  NaN fraction: {fu_mat.isna().values.mean():.4f}')
    print(f'  value range: {np.nanmin(fu_mat.values):.2f} to {np.nanmax(fu_mat.values):.2f}')


if __name__ == '__main__':
    main()
