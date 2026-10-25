#!/usr/bin/env python3
"""A1 + A2: fair, information-matched decomposition of eNESH variance.


For every modelling gene, eNESH across development patients is regressed on
two parallel predictor ladders that use the SAME information:

  Expression ladder                         Deviation ladder
  E1  own expression x                      D1  R1 = |z|
  E3  x, neighbour mean xN, global xG       D3  R1, R2, R2b  (+ G = global mean |z| -> D4)
  E7  E3 + x*xN + x^2 + xN^2 + xG^2         D7  D4 + R1^2 + R2^2 + G^2
  COMB  E7 + D7 (does deviation coding add anything over matched expression?)

Both in-sample adjusted R^2 and 5-fold cross-validated R^2 are reported, overall
and split by binary vs non-binary features, reference-isolated vs connected
anchors, and one-sided vs two-sided genes (genes whose tumours all lie on one
side of the GTEx mean, where |z| is an affine function of expression).

Example
  python a1_a2_variance_decomposition.py --gtex gtex.csv --tumour tcga_cal.csv \
      --edges edges.csv --anchors anchors.txt --dev dev_ids.txt \
      --model-genes model_genes.txt --out results/a1
  (add --enesh enesh.csv to use the saved eNESH instead of recomputing it)
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import core

LADDER = ["E1", "E3", "E7", "D1", "D3", "D4", "D7", "COMB"]


def design_matrices(x, xN, xG, R1, R2, G):
    R2b = R1 * R2
    E3 = np.column_stack([x, xN, xG])
    E7 = np.column_stack([E3, x * xN, x ** 2, xN ** 2, xG ** 2])
    D3 = np.column_stack([R1, R2, R2b])
    D4 = np.column_stack([D3, G])
    D7 = np.column_stack([D4, R1 ** 2, R2 ** 2, G ** 2])
    return {"E1": x[:, None], "E3": E3, "E7": E7, "D1": R1[:, None], "D3": D3,
            "D4": D4, "D7": D7, "COMB": np.column_stack([E7, D7])}


def decompose(enesh_dev: pd.DataFrame, expr_dev: np.ndarray, Z_dev: np.ndarray,
              net: core.Network, ref_keep: np.ndarray, genes: list[str]) -> pd.DataFrame:
    """enesh_dev: anchors x dev patients; expr_dev, Z_dev: net.genes x dev patients."""
    xN_all = net.neighbour_mean(expr_dev)
    R2_all = net.neighbour_mean(np.abs(np.nan_to_num(Z_dev)))
    xG = expr_dev.mean(0)
    G = np.nanmean(np.abs(Z_dev), 0)
    ref_deg = np.asarray(net.inc[:, ref_keep].sum(1)).ravel()
    rows = []
    for g in genes:
        if g not in net.gidx or g not in enesh_dev.index:
            continue
        k = net.gidx[g]
        y = enesh_dev.loc[g].to_numpy(float)
        x, z = expr_dev[k], Z_dev[k]
        R1, R2 = np.abs(z), R2_all[k]
        rec = {"gene": g, "n_unique_enesh": len(np.unique(y)),
               "binary": len(np.unique(y)) == 2,
               "reference_isolated": ref_deg[k] == 0,
               "majority_sign_fraction": max((z > 0).mean(), (z < 0).mean()),
               "mean_abs_z": np.nanmean(np.abs(z))}
        rec["one_sided"] = rec["majority_sign_fraction"] == 1.0
        if np.var(y) == 0:
            rows.append(rec)
            continue
        for name, v in {"x": x, "R1": R1, "R2": R2, "R2b": R1 * R2}.items():
            rec[f"abs_r_{name}"] = abs(np.corrcoef(y, v)[0, 1]) if np.var(v) > 0 else np.nan
        for m, X in design_matrices(x, xN_all[k], xG, R1, R2, G).items():
            rec[f"adjR2_{m}"], rec[f"cvR2_{m}"] = core.ols_r2(y, X)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("gene")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    groups = {"all": df, "non_binary": df[~df.binary], "binary": df[df.binary],
              "connected": df[~df.reference_isolated], "isolated": df[df.reference_isolated],
              "one_sided": df[df.one_sided], "two_sided": df[~df.one_sided]}
    cols = [c for c in df.columns if c.startswith(("adjR2_", "cvR2_", "abs_r_"))]
    out = []
    for name, d in groups.items():
        row = {"subset": name, "n_genes": len(d)}
        for c in cols:
            row[f"median_{c}"] = d[c].median()
        out.append(row)
    s = pd.DataFrame(out).set_index("subset")
    # gain of deviation coding over information-matched expression
    s["median_cvR2_gain_COMB_minus_E7"] = [
        (g["cvR2_COMB"] - g["cvR2_E7"]).median() for g in groups.values()]
    s["median_cvR2_gain_D7_minus_E7"] = [
        (g["cvR2_D7"] - g["cvR2_E7"]).median() for g in groups.values()]
    return s


def plot(df: pd.DataFrame, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, d) in zip(axes, [("Non-binary features", df[~df.binary]),
                                     ("Binary (mostly reference-isolated) features", df[df.binary])]):
        data = [d[f"cvR2_{m}"].dropna() for m in LADDER]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(LADDER) + 1), LADDER)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{title} (n = {len(d):,})", fontsize=10)
        ax.set_xlabel("Predictor set (E = expression, D = deviation)")
    axes[0].set_ylabel("Cross-validated R² of eNESH")
    fig.tight_layout()
    fig.savefig(path, dpi=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtex", required=True)
    ap.add_argument("--tumour", required=True, help="calibrated TCGA matrix (all 144 patients)")
    ap.add_argument("--edges", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--dev", required=True, help="development sample IDs (115)")
    ap.add_argument("--model-genes", help="the 3,669 modelling genes (default: all anchors)")
    ap.add_argument("--enesh", help="optional saved eNESH matrix (anchors x patients)")
    ap.add_argument("--density", type=float, default=0.20)
    ap.add_argument("--dmax", type=float, default=None, help="use 166 to match the paper exactly")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gtex, tum = core.read_matrix(a.gtex), core.read_matrix(a.tumour)
    net = core.Network(core.read_edges(a.edges), gtex.index.intersection(tum.index), core.read_list(a.anchors))
    G = gtex.loc[net.genes].to_numpy(float)
    T = tum.loc[net.genes].to_numpy(float)
    feat = core.build_network_features(G, T, net, density=a.density, dmax=a.dmax)
    if a.enesh:
        en = core.read_matrix(a.enesh)
    else:
        en = pd.DataFrame(feat["enesh"], index=net.anchors, columns=tum.columns)
        en.to_csv(os.path.join(a.out, "enesh_recomputed.csv"))
    dev = [s for s in core.read_list(a.dev) if s in tum.columns]
    cols = [tum.columns.get_loc(s) for s in dev]
    genes = core.read_list(a.model_genes) if a.model_genes else net.anchors

    df = decompose(en[dev], T[:, cols], feat["Z"][:, cols], net, feat["ref_keep"], genes)
    df.to_csv(os.path.join(a.out, "a1_per_gene.csv"))
    s = summarise(df)
    s.to_csv(os.path.join(a.out, "a1_summary.csv"))
    plot(df, os.path.join(a.out, "a1_cvR2_ladder.png"))
    pd.set_option("display.width", 200)
    print(s[[c for c in s.columns if "cvR2" in c]].round(3).T)


if __name__ == "__main__":
    main()
