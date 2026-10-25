#!/usr/bin/env python3
"""A5: the |z| regime of the data and the validity of the first-order expansion.


Reports
  1. Exactness check: Eq. 4 versus brute-force recomputation of Pearson r on
     N + 1 samples for a random subset of patients and edges (max |difference|).
  2. Distribution of |z| for anchors and for all network genes (quantiles,
     fraction > 2, > 5, > sqrt(N)).
  3. Error of the first-order (influence-function) approximation, Eq. S3,
     relative to the exact Eq. 4, binned by max(|z_i|, |z_j|); the |z| at which
     the median absolute error first exceeds 0.1 and 0.5.
  4. Saturation: for large |z|, sPCC -> sign(z_i z_j)(N + 1) - N rho. Fraction of
     patient-edges within 5% of that limit.
Produces a two-panel figure for the Supplement.

Example
  python a5_z_regime.py --gtex gtex.csv --tumour tcga_cal.csv --edges edges.csv \
      --anchors anchors.txt --out results/a5
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import core


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtex", required=True)
    ap.add_argument("--tumour", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--check-patients", type=int, default=5)
    ap.add_argument("--check-edges", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gtex, tum = core.read_matrix(a.gtex), core.read_matrix(a.tumour)
    net = core.Network(core.read_edges(a.edges), gtex.index.intersection(tum.index), core.read_list(a.anchors))
    G = gtex.loc[net.genes].to_numpy(float)
    T = tum.loc[net.genes].to_numpy(float)
    N = G.shape[1]
    mu, sd = core.ref_stats(G)
    rho = core.edge_corr(G, net.ia, net.ib)
    Z = core.zscores(T, mu, sd)
    out = {}

    # 1. exactness check against brute force
    rng = np.random.default_rng(42)
    ps = rng.choice(T.shape[1], size=min(a.check_patients, T.shape[1]), replace=False)
    es = rng.choice(len(net.ia), size=min(a.check_edges, len(net.ia)), replace=False)
    maxdiff = 0.0
    for p in ps:
        for e in es:
            i, j = net.ia[e], net.ib[e]
            xi = np.append(G[i], T[i, p])
            xj = np.append(G[j], T[j, p])
            brute = (N + 1) * np.corrcoef(xi, xj)[0, 1] - N * rho[e]
            closed = core.spcc_exact(Z[i, p], Z[j, p], rho[e], N)
            if np.isfinite(brute) and np.isfinite(closed):
                maxdiff = max(maxdiff, abs(brute - closed))
    out["closed_form_max_abs_difference"] = maxdiff

    # 2. |z| distribution
    absz_all = np.abs(Z[np.isfinite(Z)])
    absz_anc = np.abs(Z[net.aidx])
    absz_anc = absz_anc[np.isfinite(absz_anc)]
    for lab, v in (("anchors", absz_anc), ("all_genes", absz_all)):
        for q in (0.25, 0.5, 0.75, 0.9, 0.99):
            out[f"abs_z_{lab}_q{int(q * 100)}"] = float(np.quantile(v, q))
        out[f"frac_abs_z_gt2_{lab}"] = float((v > 2).mean())
        out[f"frac_abs_z_gt5_{lab}"] = float((v > 5).mean())
        out[f"frac_abs_z_gt_sqrtN_{lab}"] = float((v > np.sqrt(N)).mean())
    out["mean_per_gene_abs_z_gt2_anchors"] = float((np.nanmean(np.abs(Z[net.aidx]), 1) > 2).mean())

    # 3. first-order error
    zi, zj = Z[net.ia], Z[net.ib]
    exact = core.spcc_exact(zi, zj, rho[:, None], N)
    first = core.spcc_first_order(zi, zj, rho[:, None])
    err = np.abs(exact - first)
    zmax = np.maximum(np.abs(zi), np.abs(zj))
    ok = np.isfinite(err) & np.isfinite(zmax)
    bins = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, np.inf])
    cat = np.digitize(zmax[ok], bins) - 1
    tab = pd.DataFrame({"bin": cat, "err": err[ok]}).groupby("bin").err.agg(
        ["count", "median", lambda s: s.quantile(.9)])
    tab.columns = ["n_patient_edges", "median_abs_error", "q90_abs_error"]
    tab.index = [f"[{bins[b]}, {bins[b + 1]})" for b in tab.index]
    tab.to_csv(os.path.join(a.out, "a5_first_order_error_by_zmax.csv"))
    for thr in (0.1, 0.5):
        hit = tab[tab.median_abs_error > thr]
        out[f"zmax_where_median_error_exceeds_{thr}"] = hit.index[0] if len(hit) else "never"
        out[f"frac_patient_edges_error_gt_{thr}"] = float((err[ok] > thr).mean())

    # 4. saturation
    limit = np.sign(zi * zj) * (N + 1) - N * rho[:, None]
    near = np.abs(exact - limit) <= 0.05 * np.abs(limit)
    out["frac_patient_edges_within5pct_of_saturation"] = float(near[ok].mean())

    pd.Series(out).to_csv(os.path.join(a.out, "a5_summary.csv"), header=False)
    print(pd.Series(out).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.clip(absz_anc, 0, 30), bins=120, color="#4a6fa5")
    ax[0].axvline(2, ls="--", color="k", lw=0.8)
    ax[0].set_xlabel("|z| of anchor genes (patient x gene, clipped at 30)")
    ax[0].set_ylabel("Count")
    mids = np.arange(len(tab))
    ax[1].plot(mids, tab.median_abs_error, "o-", label="median")
    ax[1].plot(mids, tab.q90_abs_error, "s--", label="90th percentile")
    ax[1].set_yscale("log")
    ax[1].set_xticks(mids, tab.index, rotation=60, fontsize=7)
    ax[1].axhline(0.1, color="grey", lw=0.6)
    ax[1].set_xlabel("max(|z_i|, |z_j|)")
    ax[1].set_ylabel("|exact Eq. 4 − first-order Eq. S3|")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "a5_z_regime.png"), dpi=300)


if __name__ == "__main__":
    main()
