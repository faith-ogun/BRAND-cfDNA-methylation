"""Cox proportional-hazards assumption tests for the two methylation models.

Supplementary Table 2 lists these tests for the survival models in the
manuscript; Model B below is its 36-DMR row.

Two models are tested:

  Model A: baseline 36-DMR continuous risk score (the C-index 0.700 model).
           The model is a ridge-penalised Cox fitted inside LOOCV.
           For the assumption test we use the out-of-fold risk score as a
           single covariate, which is the standard way to test PH for a
           penalised multi-feature model (testing all 36 coefficients
           separately with 42 patients and 31 events would be badly
           underpowered).

  Model B: dynamic change model (the HR 0.30 result). Single covariate:
           per-patient change in 36-DMR non-responder probability between
           baseline and follow-up, in the responder direction.

Both use lifelines' proportional_hazard_test, which is the Schoenfeld
residual test (the Python equivalent of R's cox.zph). Supplementary Table 2
reports the 'km' time transform, the default in R's cox.zph; the 'rank' and
'identity' transforms are printed for comparison.

Output: prints the tests. Step 8 of the README saves this output to
results/sensitivity_analysis/ph_assumption_methylation.txt.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
SENS = REPO / "results" / "sensitivity_analysis"


def test_model(df, covariate, label):
    """Fit a Cox model on one covariate and run the Schoenfeld residual test."""
    cph = CoxPHFitter()
    cph.fit(df[["T", "E", covariate]], duration_col="T", event_col="E")

    results = {}
    for transform in ("rank", "km", "identity"):
        zph = proportional_hazard_test(cph, df[["T", "E", covariate]],
                                       time_transform=transform)
        chi2 = float(zph.summary["test_statistic"].iloc[0])
        p = float(zph.summary["p"].iloc[0])
        results[transform] = (chi2, p)

    s = cph.summary.loc[covariate]
    coef, se, hr, pval = (float(s["coef"]), float(s["se(coef)"]),
                          float(s["exp(coef)"]), float(s["p"]))

    print(f"\n{'=' * 72}")
    print(f"{label}")
    print(f"{'=' * 72}")
    print(f"  n = {len(df)}, events = {int(df['E'].sum())}, "
          f"censored = {int((df['E'] == 0).sum())}")
    print(f"  Cox coefficient: {coef:.4f} (se {se:.4f}), "
          f"HR per 1 unit = {hr:.3f}, HR per 0.1 unit = {np.exp(0.1 * coef):.3f}, "
          f"model p = {pval:.4f}")
    print(f"\n  Schoenfeld residual test (proportional hazards assumption):")
    for transform, (chi2, p) in results.items():
        verdict = "assumption HOLDS" if p >= 0.05 else "ASSUMPTION VIOLATED"
        marker = "  <-- km, as in R's cox.zph" if transform == "km" else ""
        print(f"    time transform '{transform:8s}': "
              f"chi2 = {chi2:6.3f}, p = {p:.3f}   {verdict}{marker}")

    return results["km"]


def main():
    # ---------------- Model A: baseline 36-DMR continuous risk score --------
    cv = pd.read_csv(SENS / "continuous_validation.csv")
    a = cv.dropna(subset=["PFS_Months", "pfs_event", "cox_risk_score"]).copy()
    a = a.rename(columns={"PFS_Months": "T", "pfs_event": "E"})
    a["E"] = a["E"].astype(int)
    # log the partial hazard so the covariate is the linear predictor
    a["risk_score"] = np.log(a["cox_risk_score"])
    chi2_a, p_a = test_model(a, "risk_score",
                             "MODEL A: PFS & baseline 36-DMR continuous risk score")

    # ---------------- Model B: dynamic change model -------------------------
    dyn = pd.read_csv(SENS / "fu_dynamic_results.csv")
    b = dyn.dropna(subset=["PFS_Months", "pfs_event", "delta_prob_NR"]).copy()
    b = b.rename(columns={"PFS_Months": "T", "pfs_event": "E"})
    b["E"] = b["E"].astype(int)
    # responder direction, matching the published HR 0.30
    b["delta_prob_R"] = -b["delta_prob_NR"]
    chi2_b, p_b = test_model(b, "delta_prob_R",
                             "MODEL B: PFS & dynamic change in 36-DMR score")

    # ---------------- Summary, km transform (Supplementary Table 2) ---------
    print(f"\n{'=' * 72}")
    print("SUMMARY, km time transform (Supplementary Table 2 reports Model B)")
    print(f"{'=' * 72}")
    print(f"{'Cox Proportional Hazards Assumption':<52}{'Chi2':>8}{'p':>8}")
    print(f"{'PFS & baseline 36-DMR risk score':<52}{chi2_a:>8.2f}{p_a:>8.2f}")
    print(f"{'PFS & dynamic change in 36-DMR score':<52}{chi2_b:>8.2f}{p_b:>8.2f}")


if __name__ == "__main__":
    main()
