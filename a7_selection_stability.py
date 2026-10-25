#!/usr/bin/env python3
"""A7: stability of univariate feature selection across resamples.


For each representation (expression, eNESH, R1, R2, R2b, ...) and K in {10, 25, 50}:
  * the top-K univariate set (S_g = max(C, 1 - C)) in every training set of
    (a) the 3 x 5 outer CV of the development cohort and
    (b) the 100 event-stratified 80:20 partitions of the full cohort;
  * mean pairwise Jaccard index and Kuncheva consistency index (0 = chance,
    1 = identical sets), with the Jaccard expected by chance;
  * per-feature selection frequencies (top 20 reported).
With --selected it also scores the feature sets actually chosen by
nested_cv_survival.py (variable K, so only Jaccard is reported).

Example
  python a7_selection_stability.py --rep M2=expr.csv --rep M3=enesh.csv \
      --rep R1=r1.csv --rep R2=r2.csv --rep R2b=r2b.csv --survival surv.csv \
      --dev dev_ids.txt --all all144_ids.txt --out results/a7
"""
from __future__ import annotations

import argparse
import itertools
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import core

KS = (10, 25, 50)


def kuncheva(a, b, k, p):
    r = len(a & b)
    return (r * p - k * k) / (k * (p - k))


def stability(sets, k, p):
    jac, kun = [], []
    for a, b in itertools.combinations(sets, 2):
        jac.append(len(a & b) / len(a | b))
        if k is not None:
            kun.append(kuncheva(a, b, k, p))
    out = dict(n_sets=len(sets), mean_jaccard=np.mean(jac), sd_jaccard=np.std(jac))
    if k is not None:
        out.update(mean_kuncheva=np.mean(kun), chance_jaccard=k / (2 * p - k))
    return out


def top_sets(X, t, e, splits, k_max):
    sets = []
    for tr in splits:
        Xt = X[tr]
        var = Xt.var(0) > 0
        c = core.cindex_columns(Xt[:, var], t[tr], e[tr])
        s = np.full(X.shape[1], -1.0)
        s[var] = np.maximum(c, 1 - c)
        sets.append(np.argsort(-s, kind="stable")[:k_max])
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", action="append", required=True, help="NAME=features x samples matrix")
    ap.add_argument("--survival", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--all", help="all-cohort IDs for the 100 partitions (default: matrix columns)")
    ap.add_argument("--n-partitions", type=int, default=100)
    ap.add_argument("--holdout-fraction", type=float, default=0.20)
    ap.add_argument("--selected", action="append", default=[],
                    help="NAME=<name>_selected_features.csv from nested_cv_survival.py")
    ap.add_argument("--symbols", help="CSV: ensembl_id, symbol")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    surv = core.read_survival(a.survival)
    sym = None
    if a.symbols:
        s = pd.read_csv(a.symbols).iloc[:, :2]
        sym = dict(zip(s.iloc[:, 0].astype(str).str.split(".").str[0], s.iloc[:, 1]))

    rows, freqs = [], []
    for spec in a.rep:
        name, path = spec.split("=", 1)
        F = core.read_matrix(path)
        for design in ("outer_cv", "partitions"):
            if design == "outer_cv":
                ids = [x for x in core.read_list(a.dev) if x in F.columns and x in surv.index]
                X = F[ids].T.to_numpy(float)
                t = surv.loc[ids, "time"].to_numpy(float)
                e = surv.loc[ids, "event"].to_numpy(int)
                splits = [tr for rep in (1, 2, 3)   # same folds as the Stage D notebook
                          for tr, _ in StratifiedKFold(5, shuffle=True, random_state=42 + 100 * rep).split(X, e)]
            else:
                ids = core.read_list(a.all) if a.all else list(F.columns)
                ids = [x for x in ids if x in F.columns and x in surv.index]
                X = F[ids].T.to_numpy(float)
                t = surv.loc[ids, "time"].to_numpy(float)
                e = surv.loc[ids, "event"].to_numpy(int)
                splits = []   # seeds 0..99, exactly as in the paper's partition analysis (final.ipynb)
                for sd in range(a.n_partitions):
                    tr, _ = train_test_split(np.arange(len(ids)), test_size=a.holdout_fraction,
                                             stratify=e, random_state=sd)
                    splits.append(tr)
            p = X.shape[1]
            ranked = top_sets(X, t, e, splits, max(KS))
            for k in KS:
                sets = [set(r[:k]) for r in ranked]
                rows.append(dict(representation=name, design=design, K=k, p=p, **stability(sets, k, p)))
                cnt = pd.Series(np.concatenate([list(s) for s in sets])).value_counts() / len(sets)
                top = cnt.head(20)
                for idx, fr in top.items():
                    g = F.index[idx]
                    freqs.append(dict(representation=name, design=design, K=k, feature=g,
                                      symbol=(sym or {}).get(g, ""), selection_frequency=fr))
            print(f"[{name}] {design} done", flush=True)
    for spec in a.selected:
        name, path = spec.split("=", 1)
        sel = pd.read_csv(path)
        sets = [set(str(x).split(";")) for x in sel.features]
        rows.append(dict(representation=name, design="nested_cv_selected", K=None, **stability(sets, None, None)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "a7_stability.csv"), index=False)
    pd.DataFrame(freqs).to_csv(os.path.join(a.out, "a7_top_selection_frequencies.csv"), index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
