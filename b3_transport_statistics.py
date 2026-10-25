#!/usr/bin/env python3
"""B3: statistics for the internal partition and the external cohort.


For every model's risk scores in an evaluation cohort:
  * Harrell's C with 2,000-replicate patient bootstrap 95% CI (seed 42)
  * comparable / concordant / discordant / tied-risk pair counts
  * Uno's C (tau = 24 months) and cumulative/dynamic AUC at 12 and 24 months,
    with censoring weights estimated in the EVALUATION cohort
  * hazard ratio per SD of the risk score (Cox), and - for models whose score is
    a Cox linear predictor (--lp-models) - the calibration slope
Paired comparisons against --reference-model on identical bootstrap resamples:
  * Delta C with 95% and 90% CIs
  * TOST equivalence at +/- --margin (equivalent if the 90% CI lies inside)
  * minimum detectable Delta C at 80% power, (1.96 + 0.84) x bootstrap SE
With --surv-probs (predicted survival probabilities): IBS over 3-24 months with
evaluation-cohort censoring weights, the development Kaplan-Meier reference IBS,
and calibration-in-the-large at 12 and 24 months (mean predicted vs observed KM).

Example
  python b3_transport_statistics.py --risks cgga_risks.csv --survival cgga_surv.csv \
      --train-survival tcga_dev_surv.csv --reference-model M3 --margin 0.05 \
      --lp-models M1_cox --out results/b3_cgga
  risks CSV: sample, then one column per model (higher = higher risk).
  surv-probs CSV (optional): sample, model, time, surv_prob.
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

import core


def cox_coef(x, y):
    try:
        return float(CoxPHSurvivalAnalysis(alpha=1e-6).fit(x[:, None], y).coef_[0])
    except Exception:
        return np.nan


def km_at(time, event, ts):
    tt, s = kaplan_meier_estimator(event.astype(bool), time)
    return np.array([s[tt <= t][-1] if np.any(tt <= t) else 1.0 for t in ts])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risks", required=True, nargs="+")
    ap.add_argument("--survival", required=True)
    ap.add_argument("--train-survival", required=True, help="development cohort (KM reference)")
    ap.add_argument("--reference-model")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--lp-models", nargs="*", default=[])
    ap.add_argument("--surv-probs")
    ap.add_argument("--tau", type=float, default=24.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    risks = None
    for p in a.risks:
        r = pd.read_csv(p)
        r = r.drop(columns=[c for c in ("cohort",) if c in r.columns])
        r["sample"] = r["sample"].astype(str)
        risks = r if risks is None else risks.merge(r, on="sample", how="outer")
    risks = risks.set_index("sample")
    surv = core.read_survival(a.survival)
    ids = [s for s in risks.index if s in surv.index]
    risks, surv = risks.loc[ids], surv.loc[ids]
    t, e = surv.time.to_numpy(float), surv.event.to_numpy(int)
    y = Surv.from_arrays(e.astype(bool), t)
    tr = core.read_survival(a.train_survival)
    tau = min(a.tau, t[e == 1].max() * 0.999)
    models = list(risks.columns)
    rng = np.random.default_rng(42)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(a.n_boot)]

    def boot_c(r):
        out = np.full(a.n_boot, np.nan)
        for b, idx in enumerate(boots):
            if e[idx].sum() == 0:
                continue
            try:
                out[b] = core.cindex(r[idx], t[idx], e[idx])
            except Exception:
                pass
        return out

    rows, bc = [], {}
    for m in models:
        r = risks[m].to_numpy(float)
        bc[m] = boot_c(r)
        row = dict(model=m, n=len(ids), events=int(e.sum()), harrell_c=core.cindex(r, t, e),
                   ci_low=np.nanpercentile(bc[m], 2.5), ci_high=np.nanpercentile(bc[m], 97.5),
                   **core.pair_counts(r, t, e))
        try:
            row["uno_c_tau"] = concordance_index_ipcw(y, y, r, tau=tau)[0]
        except Exception:
            row["uno_c_tau"] = np.nan
        for h in (12, 24):
            try:
                row[f"auc_{h}m"] = cumulative_dynamic_auc(y, y, r, [h])[0][0]
            except Exception:
                row[f"auc_{h}m"] = np.nan
        z = (r - r.mean()) / (r.std() if r.std() > 0 else 1)
        row["hr_per_sd"] = float(np.exp(cox_coef(z, y)))
        if m in a.lp_models:
            row["calibration_slope"] = cox_coef(r, y)
        rows.append(row)
    res = pd.DataFrame(rows)
    res["tau_used"] = tau
    res.to_csv(os.path.join(a.out, "b3_discrimination.csv"), index=False)

    if a.reference_model:
        prow = []
        for m in models:
            if m == a.reference_model:
                continue
            d = bc[m] - bc[a.reference_model]
            obs = res.set_index("model").loc[m, "harrell_c"] - res.set_index("model").loc[a.reference_model, "harrell_c"]
            se = np.nanstd(d, ddof=1)
            lo90, hi90 = np.nanpercentile(d, [5, 95])
            prow.append(dict(comparison=f"{m} - {a.reference_model}", delta_c=obs,
                             ci95_low=np.nanpercentile(d, 2.5), ci95_high=np.nanpercentile(d, 97.5),
                             ci90_low=lo90, ci90_high=hi90, bootstrap_se=se,
                             equivalent_within_margin=bool(lo90 > -a.margin and hi90 < a.margin),
                             margin=a.margin, min_detectable_delta_80pct=2.8 * se))
        pd.DataFrame(prow).to_csv(os.path.join(a.out, "b3_paired_equivalence.csv"), index=False)
        print(pd.DataFrame(prow).round(3).to_string(index=False))

    if a.surv_probs:
        sp = pd.read_csv(a.surv_probs)
        sp["sample"] = sp["sample"].astype(str)
        grid = np.arange(3, 24.01, 1.0)
        grid = grid[(grid > t.min()) & (grid < t.max())]
        ref = km_at(tr.time.to_numpy(float), tr.event.to_numpy(int), grid)
        crow = [dict(model="KM_development_reference",
                     ibs=integrated_brier_score(y, y, np.tile(ref, (len(ids), 1)), grid))]
        obs = km_at(t, e, [12, 24])
        for m, d in sp.groupby("model"):
            P = d.pivot_table(index="sample", columns="time", values="surv_prob").reindex(ids)
            P = P.T.reindex(sorted(set(P.columns) | set(grid))).interpolate(limit_direction="both").T
            Pm = P[grid].to_numpy(float)
            row = dict(model=m, ibs=integrated_brier_score(y, y, Pm, grid))
            for h, o in zip((12, 24), obs):
                col = P.columns[np.argmin(np.abs(P.columns.to_numpy(float) - h))]
                row[f"mean_pred_surv_{h}m"] = P[col].mean()
                row[f"observed_km_{h}m"] = o
            crow.append(row)
        pd.DataFrame(crow).to_csv(os.path.join(a.out, "b3_calibration_ibs.csv"), index=False)
        print(pd.DataFrame(crow).round(3).to_string(index=False))

    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
