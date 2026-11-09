"""Annotate the 36 DMRs with nearest gene using the Ensembl REST API (GRCh38).

For each region (chr:start-end):
  1. Query overlapping protein-coding / lincRNA / antisense genes via
     /overlap/region/human/{chr}:{start}-{end}?feature=gene
  2. If overlapping gene(s) found, take the one whose body covers the most of the
     DMR (or the first protein_coding if tied).
  3. If none, expand the window by +/- 5kb, then 10kb, then 25kb, then 100kb,
     and record the nearest gene plus its TSS distance.
  4. Record gene symbol, Ensembl ID, biotype, overlap mode (overlap / nearest_Xkb),
     and TSS distance.

Output:
  - Reads:  figures/supplementary/supplement_table_36dmr.csv
  - Writes: same file in-place, with `nearest_gene`, `gene_ensembl_id`,
            `gene_biotype`, `annotation_mode`, `tss_distance_bp` columns filled.
  - Also prints a table to stdout for quick review.

Notes:
  Ensembl REST defaults to GRCh38, matches Bismark coverage convention used here.
  Rate-limited to ~10 req/sec (Ensembl allows 15).
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
CSV = REPO / "figures" / "supplementary" / "supplement_table_36dmr.csv"

BASE = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json"}
WANTED_BIOTYPES = {"protein_coding", "lincRNA", "lncRNA", "antisense", "miRNA"}
EXPAND_STEPS_KB = [0, 5, 10, 25, 100, 500]


def fetch_genes(chrom: str, start: int, end: int, retries: int = 5):
    """Return list of overlapping genes from Ensembl REST.

    Timeouts, rate limiting (429) and server errors are retried with a
    growing pause. If Ensembl still has not answered, this raises rather
    than returning an empty list, so a failed request is never read as a
    region with no genes.
    """
    region = f"{chrom.replace('chr', '')}:{start}-{end}"
    url = f"{BASE}/overlap/region/human/{region}?feature=gene"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
        except requests.exceptions.RequestException:
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json() or []
    raise RuntimeError(f"Ensembl REST gave no answer for {region} after {retries} attempts")


def choose_gene(genes, dmr_start, dmr_end):
    """Pick the most relevant gene from a list of overlapping ones."""
    if not genes:
        return None
    # Prefer wanted biotypes, then protein_coding, then by overlap span
    def score(g):
        biotype_rank = 0 if g.get("biotype") == "protein_coding" else 1
        in_wanted = 0 if g.get("biotype") in WANTED_BIOTYPES else 1
        # overlap span (negative so larger overlap sorts first)
        ov = max(0, min(int(g["end"]), dmr_end) - max(int(g["start"]), dmr_start))
        return (in_wanted, biotype_rank, -ov)
    return sorted(genes, key=score)[0]


def tss_distance(gene, dmr_start, dmr_end):
    """Distance from the DMR centre to the gene: to its TSS when the DMR
    overlaps the gene, otherwise to the nearest end of the gene.

    Uses strand-aware TSS where possible (start for +ve, end for -ve).
    Falls back to gene midpoint if strand absent.
    """
    gs, ge = int(gene["start"]), int(gene["end"])
    strand = gene.get("strand", 0)
    if strand == 1:
        tss = gs
    elif strand == -1:
        tss = ge
    else:
        tss = (gs + ge) // 2
    centre = (dmr_start + dmr_end) // 2
    if gs <= dmr_end and ge >= dmr_start:
        # overlap, distance to TSS specifically (signed magnitude)
        return abs(centre - tss)
    if centre < gs:
        return gs - centre
    return centre - ge


def annotate_region(chrom: str, start: int, end: int):
    """Find the best gene annotation, expanding window if needed."""
    for step_kb in EXPAND_STEPS_KB:
        pad = step_kb * 1000
        genes = fetch_genes(chrom, max(1, start - pad), end + pad)
        time.sleep(0.1)  # be polite to Ensembl
        # Filter to wanted biotypes when at step 0 to be selective;
        # widen at later steps.
        if step_kb == 0:
            relevant = [g for g in genes if g.get("biotype") in WANTED_BIOTYPES] or genes
        else:
            relevant = [g for g in genes if g.get("biotype") in WANTED_BIOTYPES] or genes
        gene = choose_gene(relevant, start, end)
        if gene is None:
            continue
        mode = "overlap" if step_kb == 0 else f"nearest_within_{step_kb}kb"
        dist = tss_distance(gene, start, end)
        return {
            "nearest_gene": gene.get("external_name") or gene.get("gene_id") or "",
            "gene_ensembl_id": gene.get("gene_id", ""),
            "gene_biotype": gene.get("biotype", ""),
            "annotation_mode": mode,
            "tss_distance_bp": int(dist),
        }
    return {
        "nearest_gene": "",
        "gene_ensembl_id": "",
        "gene_biotype": "",
        "annotation_mode": "no_gene_within_500kb",
        "tss_distance_bp": -1,
    }


def main():
    if not CSV.exists():
        print(f"ERROR: {CSV} not found. Run build_supplementary_figures.py first.")
        sys.exit(1)
    df = pd.read_csv(CSV)
    print(f"Annotating {len(df)} regions against Ensembl GRCh38...")

    results = []
    for i, row in df.iterrows():
        chrom = row["chromosome"]
        start = int(row["start"])
        end = int(row["end"])
        print(f"  [{i+1:>2}/{len(df)}] {chrom}:{start}-{end}", end=" -> ", flush=True)
        ann = annotate_region(chrom, start, end)
        print(f"{ann['nearest_gene'] or '(none)':<14} "
              f"{ann['gene_biotype']:<18} "
              f"{ann['annotation_mode']:<22} "
              f"TSS Δ {ann['tss_distance_bp']:>7} bp")
        results.append(ann)

    ann_df = pd.DataFrame(results)
    for col in ann_df.columns:
        df[col] = ann_df[col].values

    # Preserve the original column order, replacing the empty nearest_gene
    # column with the populated one and inserting the new columns next to it.
    cols = list(df.columns)
    # remove duplicates if any
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    df = df[cols]
    df.to_csv(CSV, index=False)

    print(f"\nWrote: {CSV}")
    print("\nSummary:")
    print(df.groupby("annotation_mode").size().to_string())
    n_named = int((df["nearest_gene"] != "").sum())
    print(f"\nRegions with a named gene: {n_named}/{len(df)}")


if __name__ == "__main__":
    main()
