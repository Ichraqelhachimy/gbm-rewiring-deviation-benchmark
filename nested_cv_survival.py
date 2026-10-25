#!/usr/bin/env python3
"""Re-implementation of the paper's Stage D pipeline for any molecular feature matrix.

Used to re-run survival discrimination for the new representations requested
in the revision (A2 binary vs non-binary eNESH, A3/A4 eNESH under alternative
references, densities and methods, A1 matched-expression features), under
EXACTLY the same leakage-controlled design as the paper:

  * outer 3 x 5 stratified CV (by event), seeds 42 + 100*repeat as in the Stage D notebook
  * inner 3-fold CV selecting learner, hyperparameters and K in {10, 25, 50}
  * univariate ranking S_g = max(C_g, 1 - C_g) recomputed inside every training set
  * RSF  : 300 trees, min_samples_leaf {5, 10}, max_features {sqrt, 0.3}
  * CoxNet: (l1_ratio, alpha_min_ratio) in {(0.5,0.01), (0.9,0.01), (0.5,0.1)};
            predictions use the smallest penalty on the path, as in the paper
            (--coxnet-alpha-grid tunes it instead, as a sensitivity analysis)
  * GBSA : 100 iterations, (lr 0.05, depth 2) or (lr 0.1, depth 3), subsample 0.8

Outputs: per-fold C-indices, the selected configuration and the selected feature
set of every outer fold (input for A7), and - with --holdout / --external - the
final development-fitted model applied to frozen features (risk scores for B3).

Example
  python nested_cv_survival.py --features enesh_nonbinary.csv --survival surv.csv \
      --dev dev_ids.txt --holdout internal_ids.txt --external cgga_features.csv \
      --external-survival cgga_surv.csv --name M3_nonbinary --out results/cv
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

import core

warnings.filterwarnings("ignore")
K_GRID = (10, 25, 50)


def learner_grid():
    g = []
    for leaf in (5, 10):
        for mf in ("sqrt", 0.3):
            g.append(("RSF", dict(n_estimators=300, min_samples_leaf=leaf, max_features=mf)))
    for l1, amr in ((0.5, 0.01), (0.9, 0.01), (0.5, 0.1)):
        g.append(("CoxNet", dict(l1_ratio=l1, alpha_min_ratio=amr)))
    g.append(("GBSA", dict(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8)))
    g.append(("GBSA", dict(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8)))
    return g


def rank_features(X, t, e):
    c = core.cindex_columns(X, t, e)
    s = np.maximum(c, 1 - c)
    s[~np.isfinite(s)] = -1
    return np.argsort(-s, kind="stable")


class Model:
    """One learner + its preprocessing, fitted on a feature subset."""

    def __init__(self, kind, params, seeds=(42,), alpha=None):
        self.kind, self.params, self.seeds, self.alpha = kind, params, seeds, alpha

    def fit(self, X, y):
        if self.kind == "RSF":
            self.models = [RandomSurvivalForest(random_state=s, n_jobs=-1, **self.params).fit(X, y)
                           for s in self.seeds]
        elif self.kind == "GBSA":
            self.models = [GradientBoostingSurvivalAnalysis(random_state=42, **self.params).fit(X, y)]
        else:
            self.scaler = StandardScaler().fit(X)
            kw = dict(self.params)
            if self.alpha is not None:
                kw.pop("alpha_min_ratio", None)
                kw["alphas"] = [self.alpha]
            self.models = [CoxnetSurvivalAnalysis(max_iter=100000, **kw).fit(self.scaler.transform(X), y)]
        return self

    def predict(self, X):
        if self.kind == "CoxNet":
            return self.models[0].predict(self.scaler.transform(X))
        return np.mean([m.predict(X) for m in self.models], 0)


def coxnet_alpha_grid(X, y, params, n=10):
    try:
        m = CoxnetSurvivalAnalysis(n_alphas=50, max_iter=100000, **params).fit(StandardScaler().fit_transform(X), y)
        a = m.alphas_
        return list(np.unique(a[np.linspace(0, len(a) - 1, n).astype(int)]))
    except Exception:
        return [0.1]


ALPHA_GRID = False   # False = exactly as the paper: predict with the smallest penalty on the path


def candidates(X, y):
    out = []
    for kind, p in learner_grid():
        if kind == "CoxNet":
            for a in (coxnet_alpha_grid(X, y, p) if ALPHA_GRID else [None]):
                out.append((kind, p, a))
        else:
            out.append((kind, p, None))
    return out


def safe_cindex(risk, t, e):
    try:
        return core.cindex(risk, t, e)
    except Exception:
        return np.nan


def inner_select(X, t, e, seed=42):
    """Select (learner, params, alpha, K) by 3-fold inner CV on the training set."""
    y = Surv.from_arrays(e.astype(bool), t)
    cands = candidates(X, y)
    scores = np.zeros((len(cands), len(K_GRID)))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, e):
        order = rank_features(X[tr], t[tr], e[tr])
        ytr = Surv.from_arrays(e[tr].astype(bool), t[tr])
        for ki, K in enumerate(K_GRID):
            cols = order[:K]
            for ci, (kind, p, a) in enumerate(cands):
                try:
                    r = Model(kind, p, alpha=a).fit(X[tr][:, cols], ytr).predict(X[va][:, cols])
                    scores[ci, ki] += safe_cindex(r, t[va], e[va]) / 3
                except Exception:
                    scores[ci, ki] += np.nan
    ci, ki = np.unravel_index(np.nanargmax(scores), scores.shape)
    kind, p, a = cands[ci]
    return dict(kind=kind, params=p, alpha=a, K=K_GRID[ki], inner_c=float(scores[ci, ki]))


def fit_final(X, t, e, cfg, final=False):
    order = rank_features(X, t, e)
    cols = order[:cfg["K"]]
    seeds = (42, 43, 44, 45, 46) if (final and cfg["kind"] == "RSF") else (42,)
    m = Model(cfg["kind"], cfg["params"], seeds=seeds, alpha=cfg["alpha"]).fit(
        X[:, cols], Surv.from_arrays(e.astype(bool), t))
    return m, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="features x samples matrix (TCGA, all 144)")
    ap.add_argument("--survival", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--holdout", help="internal partition IDs")
    ap.add_argument("--external", help="external features x samples matrix (frozen definitions)")
    ap.add_argument("--external-survival")
    ap.add_argument("--name", default="model")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--coxnet-alpha-grid", action="store_true",
                    help="tune the CoxNet penalty on a 10-value path grid (default: smallest penalty, as in the paper)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global ALPHA_GRID
    ALPHA_GRID = a.coxnet_alpha_grid

    F = core.read_matrix(a.features)
    surv = core.read_survival(a.survival)
    dev = [s for s in core.read_list(a.dev) if s in F.columns and s in surv.index]
    Xd = F[dev].T
    Xd = Xd.loc[:, Xd.var() > 0]          # development-only zero-variance filter
    feats = Xd.columns
    X, t, e = Xd.to_numpy(float), surv.loc[dev, "time"].to_numpy(float), surv.loc[dev, "event"].to_numpy(int)

    rows, selected = [], []
    # fold generator identical to stageD_survival_oof_stacking.ipynb:
    # outer StratifiedKFold(5, shuffle, seed 42 + 100*repeat), inner seed 42 + 100*repeat + fold
    splits = []
    for rep in range(1, a.repeats + 1):
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=42 + 100 * rep)
        for k, (tr, te) in enumerate(skf.split(X, e), start=1):
            splits.append((tr, te, 42 + 100 * rep + k))
    for f, (tr, te, inner_seed) in enumerate(splits):
        cfg = inner_select(X[tr], t[tr], e[tr], seed=inner_seed)
        m, cols = fit_final(X[tr], t[tr], e[tr], cfg)
        c = safe_cindex(m.predict(X[te][:, cols]), t[te], e[te])
        rows.append(dict(fold=f, c_index=c, learner=cfg["kind"], K=cfg["K"],
                         params=json.dumps(cfg["params"]), alpha=cfg["alpha"]))
        selected.append(dict(fold=f, features=";".join(feats[cols])))
        print(f"[{a.name}] outer fold {f + 1:2d}: C = {c:.3f} ({cfg['kind']}, K={cfg['K']})", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(a.out, f"{a.name}_outer_folds.csv"), index=False)
    pd.DataFrame(selected).to_csv(os.path.join(a.out, f"{a.name}_selected_features.csv"), index=False)

    summary = dict(name=a.name, n_dev=len(dev), events_dev=int(e.sum()), n_features=len(feats),
                   cv_mean=folds.c_index.mean(), cv_sd=folds.c_index.std(ddof=1),
                   cv_median=folds.c_index.median(), cv_min=folds.c_index.min(), cv_max=folds.c_index.max())

    # final model on the full development cohort
    cfg = inner_select(X, t, e, seed=42 + 1000)
    m, cols = fit_final(X, t, e, cfg, final=True)
    final_feats = list(feats[cols])
    summary.update(final_learner=cfg["kind"], final_params=json.dumps(cfg["params"]),
                   final_alpha=cfg["alpha"], final_K=cfg["K"])
    pd.Series(final_feats, name="feature").to_csv(os.path.join(a.out, f"{a.name}_final_features.csv"), index=False)
    risks = []

    def apply(Fm, ids, label):
        avail = [g for g in final_feats if g in Fm.index]
        missing = [g for g in final_feats if g not in Fm.index]
        if missing:   # availability-adapted refit, no re-ranking or re-tuning (as in the paper)
            idx = [list(feats).index(g) for g in avail]
            mm = Model(cfg["kind"], cfg["params"], seeds=m.seeds, alpha=cfg["alpha"]).fit(
                X[:, idx], Surv.from_arrays(e.astype(bool), t))
        else:
            mm = m
        r = mm.predict(Fm.loc[avail, ids].T.to_numpy(float))
        risks.append(pd.DataFrame({"sample": ids, "cohort": label, a.name: r}))
        return r, missing

    if a.holdout:
        ho = [s for s in core.read_list(a.holdout) if s in F.columns and s in surv.index]
        r, miss = apply(F, ho, "internal")
        summary.update(internal_c=safe_cindex(r, surv.loc[ho, "time"].to_numpy(float),
                                               surv.loc[ho, "event"].to_numpy(int)),
                       internal_missing=";".join(miss))
    if a.external:
        FE = core.read_matrix(a.external)
        se = core.read_survival(a.external_survival)
        ids = [s for s in FE.columns if s in se.index]
        r, miss = apply(FE, ids, "external")
        summary.update(external_c=safe_cindex(r, se.loc[ids, "time"].to_numpy(float),
                                               se.loc[ids, "event"].to_numpy(int)),
                       external_missing=";".join(miss))
    if risks:
        pd.concat(risks).to_csv(os.path.join(a.out, f"{a.name}_risk_scores.csv"), index=False)
    pd.Series(summary).to_csv(os.path.join(a.out, f"{a.name}_summary.csv"), header=False)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
