#!/usr/bin/env python3
"""B2: MGMT positive-control check in each cohort.


For a clinical table (sample, time, event, mgmt, and optionally tmz / radiotherapy):
  * value counts of the raw MGMT field, so its coding can be checked against the
    CGGA data dictionary
  * median OS and log-rank test, methylated vs unmethylated, in all patients and
    in the temozolomide-treated (and TMZ + RT) subsets
  * Kaplan-Meier figure for the Supplement
Use --methylated-values to declare which raw labels mean "methylated".

Example
  python b2_mgmt_check.py --clinical cgga_clinical.csv --cohort CGGA \
      --methylated-values methylated 1 --tmz-column chemo_status --tmz-values 1 yes \
      --rt-column radio_status --rt-values 1 yes --out results/b2
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


def km_median(t, e):
    tt, s = kaplan_meier_estimator(e.astype(bool), t)
    below = tt[s <= 0.5]
    return float(below[0]) if len(below) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--cohort", default="cohort")
    ap.add_argument("--mgmt-column", default="mgmt")
    ap.add_argument("--methylated-values", nargs="+", required=True)
    ap.add_argument("--unmethylated-values", nargs="*", help="default: everything else that is not missing")
    ap.add_argument("--tmz-column")
    ap.add_argument("--tmz-values", nargs="*", default=["1", "yes", "Yes", "TRUE", "True"])
    ap.add_argument("--rt-column")
    ap.add_argument("--rt-values", nargs="*", default=["1", "yes", "Yes", "TRUE", "True"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pd.read_csv(a.clinical)
    d.columns = [c.lower() for c in d.columns]
    for want, opts in (("time", ("time", "os_months", "os_time", "months")),
                       ("event", ("event", "os_event", "status", "os_status"))):
        if want not in d.columns:
            d[want] = pd.to_numeric(d[next(c for c in opts if c in d.columns)], errors="coerce")
    d = d.dropna(subset=["time", "event"])
    raw = d[a.mgmt_column.lower()].astype(str).str.strip()
    print(f"[{a.cohort}] raw MGMT values:\n{raw.value_counts(dropna=False).to_string()}\n")
    meth = raw.isin(a.methylated_values)
    if a.unmethylated_values:
        unm = raw.isin(a.unmethylated_values)
    else:
        unm = ~meth & ~raw.isin(["nan", "NA", "", "None", "NaN", "unknown", "Unknown"])
    d["group"] = np.where(meth, "methylated", np.where(unm, "unmethylated", None))

    subsets = {"all": d}
    if a.tmz_column:
        tmz = d[a.tmz_column.lower()].astype(str).isin(a.tmz_values)
        subsets["TMZ"] = d[tmz]
        if a.rt_column:
            rt = d[a.rt_column.lower()].astype(str).isin(a.rt_values)
            subsets["TMZ+RT"] = d[tmz & rt]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(subsets), figsize=(4.5 * len(subsets), 3.8), squeeze=False)
    rows = []
    for ax, (name, s) in zip(axes[0], subsets.items()):
        s = s[s.group.notna()]
        row = dict(cohort=a.cohort, subset=name, n=len(s))
        for g in ("methylated", "unmethylated"):
            sg = s[s.group == g]
            row[f"n_{g}"] = len(sg)
            row[f"median_os_{g}"] = km_median(sg.time.to_numpy(float), sg.event.to_numpy(int)) if len(sg) else np.nan
            if len(sg):
                tt, sv = kaplan_meier_estimator(sg.event.astype(bool), sg.time.astype(float))
                ax.step(tt, sv, where="post", label=f"{g} (n={len(sg)})")
        if s.group.nunique() == 2:
            y = Surv.from_arrays(s.event.astype(bool), s.time.astype(float))
            chi2, p = compare_survival(y, s.group.to_numpy())
            row["logrank_p"] = p
        rows.append(row)
        ax.set_title(f"{a.cohort}: {name} (log-rank p = {row.get('logrank_p', np.nan):.3g})", fontsize=9)
        ax.set_xlabel("Months")
        ax.set_ylabel("Overall survival")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"b2_mgmt_km_{a.cohort}.png"), dpi=300)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, f"b2_mgmt_{a.cohort}.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
