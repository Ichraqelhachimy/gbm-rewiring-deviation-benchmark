#!/usr/bin/env python3
"""Build the matched-gene feature matrices used by the other scripts.

Writes, for the TCGA cohort and (optionally) a harmonized external cohort, all
restricted to the same modelling genes:
  expr.csv            calibrated expression (M2)
  r1.csv              |z|                     (R1)
  r2.csv              mean neighbour |z|      (R2)
  r2b.csv             R1 x R2                 (R2b)
  enesh.csv           eNESH                   (M3)
  enesh_nonbinary.csv eNESH, features with > 2 distinct development values (A2)
  enesh_binary.csv    eNESH, features with exactly 2 distinct values       (A2)
  matched_expr_*.csv  neighbour-mean expression, for an expression model with
                      the same information as R2 (A1 survival analogue)
The external cohort uses the frozen GTEx reference statistics, the frozen
reference network and the TCGA dmax; nothing is re-selected.

Example
  python make_feature_matrices.py --gtex gtex.csv --tumour tcga_cal.csv \
      --edges edges.csv --anchors anchors.txt --dev dev_ids.txt \
      --model-genes model_genes.txt --external cgga_harmonized.csv \
      --dmax 166 --out features
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
    ap.add_argument("--dev", required=True)
    ap.add_argument("--model-genes")
    ap.add_argument("--external")
    ap.add_argument("--density", type=float, default=0.20)
    ap.add_argument("--dmax", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gtex, tum = core.read_matrix(a.gtex), core.read_matrix(a.tumour)
    net = core.Network(core.read_edges(a.edges), gtex.index.intersection(tum.index), core.read_list(a.anchors))
    G = gtex.loc[net.genes].to_numpy(float)
    genes = core.read_list(a.model_genes) if a.model_genes else net.anchors
    genes = [g for g in genes if g in net.gidx]
    rows = [net.gidx[g] for g in genes]
    dev = [s for s in core.read_list(a.dev) if s in tum.columns]

    feat = core.build_network_features(G, tum.loc[net.genes].to_numpy(float), net,
                                       density=a.density, dmax=a.dmax)
    en_all = pd.DataFrame(feat["enesh"], index=net.anchors, columns=tum.columns)
    nuniq = en_all.loc[genes, dev].nunique(axis=1)
    binary = nuniq[nuniq == 2].index
    nonbin = nuniq[nuniq > 2].index

    def write(prefix, X, Z, en, cols):
        absz = np.abs(np.nan_to_num(Z))
        R2 = net.neighbour_mean(absz)
        xN = net.neighbour_mean(X)
        mats = {"expr": X[rows], "r1": absz[rows], "r2": R2[rows], "r2b": absz[rows] * R2[rows],
                "matched_expr_neighbour": xN[rows]}
        for k, M in mats.items():
            pd.DataFrame(M, index=genes, columns=cols).to_csv(os.path.join(a.out, f"{prefix}{k}.csv"))
        en.loc[genes].to_csv(os.path.join(a.out, f"{prefix}enesh.csv"))
        en.loc[nonbin].to_csv(os.path.join(a.out, f"{prefix}enesh_nonbinary.csv"))
        en.loc[binary].to_csv(os.path.join(a.out, f"{prefix}enesh_binary.csv"))

    write("tcga_", tum.loc[net.genes].to_numpy(float), feat["Z"], en_all, tum.columns)
    print(f"TCGA: {len(genes)} genes ({len(nonbin)} non-binary, {len(binary)} binary eNESH); dmax = {feat['dmax']}")

    if a.external:
        ext = core.read_matrix(a.external)
        avail = [g for g in net.genes if g in ext.index]
        mu, sd = core.ref_stats(G)
        X = np.full((len(net.genes), ext.shape[1]), np.nan)
        idx = [net.gidx[g] for g in avail]
        X[idx] = ext.loc[avail].to_numpy(float)
        Z = core.zscores(X, mu, sd)
        W = core.patient_edges(Z, net, feat["rho"], G.shape[1])
        W = np.where(np.isfinite(W), W, 0.0)          # edges with a missing gene cannot be retained
        valid_edge = np.isfinite(Z[net.ia]).all(1) & np.isfinite(Z[net.ib]).all(1)
        ref_keep = feat["ref_keep"] & valid_edge       # frozen network projected onto available genes
        pat_keep = core.top_k_mask(W, int(ref_keep.sum()))
        e, _ = core.enesh(net, ref_keep, np.sign(feat["rho"]), pat_keep, np.sign(W), feat["dmax"])
        en_ext = pd.DataFrame(e, index=net.anchors, columns=ext.columns)
        missing = [g for g in genes if g not in ext.index]
        keep_genes = [g for g in genes if g in ext.index]
        Xs = np.nan_to_num(X)
        write("ext_", Xs, Z, en_ext, ext.columns)
        # drop unavailable genes from external matrices (availability-adapted, no replacement)
        for f in os.listdir(a.out):
            if f.startswith("ext_"):
                p = os.path.join(a.out, f)
                df = pd.read_csv(p, index_col=0)
                df.drop(index=[g for g in missing if g in df.index]).to_csv(p)
        print(f"External: {len(keep_genes)} of {len(genes)} genes available; "
              f"{int(ref_keep.sum())} reference edges after projection; missing: {';'.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
