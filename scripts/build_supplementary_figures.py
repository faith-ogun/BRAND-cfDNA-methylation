"""Supplementary figures and table for the methylation paper.

Builds three deliverables at 300 dpi (paper-ready) into figures/supplementary/:

1. supplement_km_36dmr.png
   Kaplan-Meier curves split at the median 36-DMR continuous risk score, with
   log-rank p, C-index annotation, and at-risk numbers. Visual companion to the
   C-index 0.700 result.

2. supplement_forest_36dmr.png
   Per-region forest plot of the 36 DMRs ordered by p-value of the FU delta-beta
   Wilcoxon test. Shows median delta-beta in responders and non-responders, BH q,
   and indicates that no single region passes the multiple-testing threshold (0/36
   at q < 0.05), supporting that the panel acts multivariately.

3. supplement_table_36dmr.csv
   Tabular companion: 36 rows, one per region, with chromosome, start, end,
   baseline mean methylation in R and NR, baseline mean delta (NR minus R), FU
   delta medians, Wilcoxon raw and BH-adjusted p. The gene-annotation columns
   are left blank here and filled by annotate_36dmrs.py.

Run from repo root:
    python3 scripts/build_supplementary_figures.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
SENS = REPO / "results" / "sensitivity_analysis"
OUT = REPO / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
C_R = "#2c7bb6"       # responder / low risk / DCB
C_NR = "#c0392b"      # non-responder / high risk / no DCB


# ---------------------------------------------------------------------------
# Figure 1: KM by median 36-DMR continuous risk score
# ---------------------------------------------------------------------------
def km_figure():
    df = pd.read_csv(SENS / "continuous_validation.csv")
    df = df.dropna(subset=["PFS_Months", "pfs_event", "cox_risk_score"]).copy()
    df["pfs_event"] = df["pfs_event"].astype(int)

    median_p = df["cox_risk_score"].median()
    df["risk_group"] = np.where(df["cox_risk_score"] >= median_p, "high", "low")

    cidx = concordance_index(df["PFS_Months"], -df["cox_risk_score"], df["pfs_event"])
    high = df[df["risk_group"] == "high"]
    low = df[df["risk_group"] == "low"]
    lr = logrank_test(high["PFS_Months"], low["PFS_Months"],
                      high["pfs_event"], low["pfs_event"])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    kmf_h = KaplanMeierFitter()
    kmf_h.fit(high["PFS_Months"], high["pfs_event"],
              label=f"High predicted risk (n={len(high)})")
    kmf_h.plot_survival_function(ax=ax, ci_show=True, color=C_NR)

    kmf_l = KaplanMeierFitter()
    kmf_l.fit(low["PFS_Months"], low["pfs_event"],
              label=f"Low predicted risk (n={len(low)})")
    kmf_l.plot_survival_function(ax=ax, ci_show=True, color=C_R)

    ax.set_xlabel("PFS (months)", fontsize=11)
    ax.set_ylabel("Progression-free survival", fontsize=11)
    ax.set_title("36-DMR continuous risk score, KM by median split (n=42)",
                 fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    annot = (
        f"Harrell's C-index = {cidx:.2f}\n"
        f"95% CI 0.59 to 0.79 (2000-iter bootstrap)\n"
        f"Log-rank p = {lr.p_value:.3f}\n"
        f"Median split at risk score = {median_p:.2f}\n"
        f"High-risk events/n = {int(high['pfs_event'].sum())}/{len(high)}\n"
        f"Low-risk events/n  = {int(low['pfs_event'].sum())}/{len(low)}"
    )
    legend = ax.legend(loc="upper right", fontsize=10, frameon=True)
    plt.tight_layout()

    # Position annotation directly below the legend, right-edge aligned.
    fig.canvas.draw()
    leg_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(leg_bbox.x1, leg_bbox.y0 - 0.015, annot, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="grey", alpha=0.92))
    out = OUT / "supplement_km_36dmr.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}, C-index = {cidx:.3f}, log-rank p = {lr.p_value:.3f}")


# ---------------------------------------------------------------------------
# Figure 2: Per-region forest plot of FU delta-beta Wilcoxon
# ---------------------------------------------------------------------------
def forest_figure():
    df = pd.read_csv(SENS / "fu_dynamic_region_tests.csv")
    df = df.sort_values("p_raw").reset_index(drop=True)
    # plotted top-to-bottom by ascending p (most significant at top)

    n = len(df)
    fig, ax = plt.subplots(figsize=(9.5, max(7.5, 0.28 * n + 2)))
    y = np.arange(n)[::-1]                              # top = lowest p

    # Plot median delta-beta in R and NR as paired markers
    ax.scatter(df["median_delta_R"], y, color=C_R, s=42, zorder=3,
               label="DCB median Δβ (FU minus baseline)")
    ax.scatter(df["median_delta_NR"], y, color=C_NR, s=42, zorder=3,
               label="no DCB median Δβ (FU minus baseline)")

    # Connect the paired points
    for yi, dR, dNR in zip(y, df["median_delta_R"], df["median_delta_NR"]):
        ax.plot([dR, dNR], [yi, yi], color="grey", lw=0.6, alpha=0.55, zorder=1)

    ax.axvline(0, color="black", lw=0.8, ls="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["region"], fontsize=7.5, family="monospace")
    ax.set_xlabel("Median change in methylation % (FU minus baseline)", fontsize=11)
    ax.set_title("Per-region change in methylation by response group (36 DMRs)",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    legend = ax.legend(loc="lower right", fontsize=9, frameon=True)

    # Annotate BH q on the right
    ax_q = ax.twinx()
    ax_q.set_ylim(ax.get_ylim())
    ax_q.set_yticks(y)
    q_labels = [f"q = {q:.2f}" for q in df["q_BH"]]
    ax_q.set_yticklabels(q_labels, fontsize=7.5)
    ax_q.set_ylabel("Benjamini-Hochberg adjusted p across 36 tests", fontsize=10)
    ax_q.tick_params(axis="y", length=0)

    plt.tight_layout()

    # Position summary banner directly on top of the legend, matched in width.
    # Done after tight_layout so the legend's final bbox is settled.
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    leg_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    n_sig = int((df["q_BH"] < 0.05).sum())
    summary = (
        f"Mann-Whitney U, FU minus baseline\n"
        f"BH correction across 36 regions\n"
        f"{n_sig}/{n} regions reach q < 0.05\n"
        f"min raw p = {df['p_raw'].min():.4f}, "
        f"min q = {df['q_BH'].min():.3f}"
    )
    # Draw a box the same width as the legend, sitting directly on top of it.
    box_height = 0.048   # axes-fraction; tuned to fit four lines at fontsize 7 snugly
    box = FancyBboxPatch(
        (leg_bbox.x0, leg_bbox.y1),
        leg_bbox.width, box_height,
        boxstyle="round,pad=0.002",
        transform=ax.transAxes,
        facecolor="white", edgecolor="grey", alpha=0.92, linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(box)
    # Text centred horizontally inside the box.
    ax.text(leg_bbox.x0 + leg_bbox.width / 2,
            leg_bbox.y1 + box_height / 2,
            summary,
            transform=ax.transAxes,
            ha="center", va="center", fontsize=7, family="monospace",
            zorder=6)
    out = OUT / "supplement_forest_36dmr.png"
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}, {n} regions, {n_sig} significant at q<0.05")


# ---------------------------------------------------------------------------
# Table: 36 DMRs with coordinates, baseline R / NR means, FU Wilcoxon stats
# ---------------------------------------------------------------------------
def supplementary_table():
    X = pd.read_csv(SENS / "dmr36_matrix.csv", index_col="patient_id")
    labels = pd.read_csv(SENS / "loocv_results_final.csv")[
        ["patient_id", "actual"]
    ].set_index("patient_id")
    X = X.join(labels)
    R = X[X["actual"] == 0].drop(columns="actual")
    NR = X[X["actual"] == 1].drop(columns="actual")

    region_stats = pd.DataFrame({
        "region": R.columns,
        "baseline_mean_DCB_pct": R.mean().values,
        "baseline_mean_noDCB_pct": NR.mean().values,
    })
    region_stats["baseline_delta_noDCB_minus_DCB_pct"] = (
        region_stats["baseline_mean_noDCB_pct"]
        - region_stats["baseline_mean_DCB_pct"]
    )
    region_stats["baseline_direction_in_noDCB"] = np.where(
        region_stats["baseline_delta_noDCB_minus_DCB_pct"] > 0,
        "hypermethylated", "hypomethylated",
    )

    # Parse chr:start-end into columns
    coord = region_stats["region"].str.extract(
        r"^(?P<chromosome>chr[^:]+):(?P<start>\d+)-(?P<end>\d+)$"
    )
    coord["start"] = coord["start"].astype(int)
    coord["end"] = coord["end"].astype(int)
    coord["width_bp"] = coord["end"] - coord["start"]

    # FU dynamic Wilcoxon
    fu_wilcox = pd.read_csv(SENS / "fu_dynamic_region_tests.csv").rename(
        columns={
            "median_delta_R": "fu_median_delta_DCB",
            "median_delta_NR": "fu_median_delta_noDCB",
            "p_raw": "fu_wilcoxon_p_raw",
            "q_BH": "fu_wilcoxon_q_BH",
        }
    )[
        ["region", "fu_median_delta_DCB", "fu_median_delta_noDCB",
         "fu_wilcoxon_p_raw", "fu_wilcoxon_q_BH"]
    ]

    full = (
        pd.concat([coord, region_stats], axis=1)
        .merge(fu_wilcox, on="region", how="left")
    )
    full.insert(0, "nearest_gene", "")        # filled by annotate_36dmrs.py
    full = full[[
        "region", "chromosome", "start", "end", "width_bp",
        "nearest_gene",
        "baseline_mean_DCB_pct", "baseline_mean_noDCB_pct",
        "baseline_delta_noDCB_minus_DCB_pct", "baseline_direction_in_noDCB",
        "fu_median_delta_DCB", "fu_median_delta_noDCB",
        "fu_wilcoxon_p_raw", "fu_wilcoxon_q_BH",
    ]]

    # Order by absolute baseline delta (biggest separator first)
    full = (
        full.assign(_abs=full["baseline_delta_noDCB_minus_DCB_pct"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )

    # Round for readability
    for col in [
        "baseline_mean_DCB_pct", "baseline_mean_noDCB_pct",
        "baseline_delta_noDCB_minus_DCB_pct",
        "fu_median_delta_DCB", "fu_median_delta_noDCB",
    ]:
        full[col] = full[col].round(2)
    for col in ["fu_wilcoxon_p_raw", "fu_wilcoxon_q_BH"]:
        full[col] = full[col].round(4)

    out = OUT / "supplement_table_36dmr.csv"
    full.to_csv(out, index=False)
    n_hyper = int((full["baseline_direction_in_noDCB"] == "hypermethylated").sum())
    print(f"  wrote {out.name}, {len(full)} rows, {n_hyper}/36 hypermethylated in no DCB at baseline")


def main():
    print("Building supplementary figures and table...")
    km_figure()
    forest_figure()
    supplementary_table()
    print(f"All outputs in: {OUT}")


if __name__ == "__main__":
    main()
