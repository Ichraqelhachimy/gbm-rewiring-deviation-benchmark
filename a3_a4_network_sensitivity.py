#!/usr/bin/env python3
"""A3 + A4: does "networks largely encode deviation" depend on the reference,
the density and the single-sample method?


For every combination of
  reference : the GTEx reference, GTEx subsets (sample lists), random GTEx halves,
              any other normal-brain matrix, and TUMOUR_LOO (standard leave-one-out
              LIONESS inside the tumour cohort, i.e. a reference close to the samples)
  density   : e.g. 0.10 0.15 0.20 fdr
  method    : spcc (add-one-in LIONESS::PCC, the paper) and ssn (SSN z-score edges)
it reports
  * reference distance       : median over patients of mean |z|
  * overlap of each patient's retained edges with the top-|z_i z_j| edges (chance = density)
  * overlap with the reference network
  * eNESH variance decomposition (A1 ladder): median CV R^2 for E7 (matched expression)
    and D7 (deviation), all and non-binary features
  * the same decomposition for an alternative rewiring score, the unsigned Jaccard
    distance between an anchor's reference and patient neighbourhoods
and optionally saves each eNESH matrix so nested_cv_survival.py can be run on it.

Example
  python a3_a4_network_sensitivity.py --tumour tcga_cal.csv --edges edges.csv \
      --anchors anchors.txt --dev dev_ids.txt --model-genes model_genes.txt \
      --reference GTEX=gtex.csv --reference GTEX_BA9=gtex.csv@ba9_ids.txt \
      --reference GTEX_CORTEX=gtex.csv@cortex_ids.txt --gtex-halves 5 \
      --reference TUMOUR_LOO --densities 0.10 0.15 0.20 fdr --methods spcc ssn \
      --save-enesh --out results/a3
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import core
from a1_a2_variance_decomposition import decompose


def jaccard_distance(net, ref_keep, pat_keep):
    A = net.anchor_inc
    inter = A @ (ref_keep[:, None] & pat_keep).astype(np.float32)
    union = A @ (ref_keep[:, None] | pat_keep).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = 1 - inter / union
    d[union == 0] = 0.0
    return np.asarray(d)


def parse_reference(spec, tumour_genes):
    if spec == "TUMOUR_LOO":
        return "TUMOUR_LOO", None
    name, rest = spec.split("=", 1)
    path, _, ids = rest.partition("@")
    M = core.read_matrix(path)
    if ids:
        keep = [s for s in core.read_list(ids) if s in M.columns]
        M = M[keep]
    return name, M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tumour", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--model-genes")
    ap.add_argument("--reference", action="append", required=True,
                    help="NAME=matrix.csv[@sample_ids.txt] or TUMOUR_LOO; repeatable")
    ap.add_argument("--gtex-halves", type=int, default=0,
                    help="also use N random halves of the FIRST reference (seed 42)")
    ap.add_argument("--densities", nargs="+", default=["0.20"])
    ap.add_argument("--methods", nargs="+", default=["spcc"], choices=["spcc", "ssn"])
    ap.add_argument("--save-enesh", action="store_true")
    ap.add_argument("--decompose-genes", type=int, default=0,
                    help="run the variance decomposition on a fixed random subset of this many "
                         "modelling genes (seed 42); 0 = all. 800 keeps medians within about 0.01")
    ap.add_argument("--skip-jaccard", action="store_true", help="skip the Jaccard-distance score")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tum = core.read_matrix(a.tumour)
    refs = [parse_reference(s, tum.index) for s in a.reference]
    if a.gtex_halves:
        base_name, base = next((n, M) for n, M in refs if M is not None)
        rng = np.random.default_rng(42)
        for h in range(a.gtex_halves):
            cols = rng.choice(base.columns, size=base.shape[1] // 2, replace=False)
            refs.append((f"{base_name}_half{h + 1}", base[cols]))

    genes_common = tum.index
    for _, M in refs:
        if M is not None:
            genes_common = genes_common.intersection(M.index)
    net = core.Network(core.read_edges(a.edges), genes_common, core.read_list(a.anchors))
    T = tum.loc[net.genes].to_numpy(float)
    dev = [s for s in core.read_list(a.dev) if s in tum.columns]
    cols = [tum.columns.get_loc(s) for s in dev]
    genes = core.read_list(a.model_genes) if a.model_genes else net.anchors
    genes = [g for g in genes if g in net.gidx]
    if a.decompose_genes and a.decompose_genes < len(genes):
        genes = list(np.random.default_rng(42).choice(genes, a.decompose_genes, replace=False))
        print(f"variance decomposition on a fixed random subset of {len(genes)} genes")
    print(f"{len(net.genes):,} genes, {len(net.ia):,} candidate edges, {len(net.anchors):,} anchors")

    rows, per_patient = [], []
    for name, M in refs:
        R = None if M is None else M.loc[net.genes].to_numpy(float)
        for dens in a.densities:
            rule = "fdr" if dens == "fdr" else "density"
            d = 0.20 if dens == "fdr" else float(dens)
            for meth in a.methods:
                tag = f"{name}|{dens}|{meth}"
                feat = core.build_network_features(R, T, net, density=d, method=meth,
                                                   ref_rule=rule, loo=(M is None))
                ov = core.overlap_diagnostics(feat, net)
                ov.insert(0, "sample", tum.columns)
                ov.insert(0, "config", tag)
                per_patient.append(ov)
                en = pd.DataFrame(feat["enesh"], index=net.anchors, columns=tum.columns)
                jd = pd.DataFrame(jaccard_distance(net, feat["ref_keep"], feat["pat_keep"]),
                                  index=net.anchors, columns=tum.columns)
                Zd = feat["Z"][:, cols]
                dec_e = decompose(en[dev], T[:, cols], Zd, net, feat["ref_keep"], genes)
                dec_j = None if a.skip_jaccard else decompose(jd[dev], T[:, cols], Zd, net, feat["ref_keep"], genes)
                nb = ~dec_e.binary
                row = dict(reference=name, density=dens, method=meth,
                           n_reference=(T.shape[1] - 1 if M is None else R.shape[1]),
                           retained_edges=feat["k"], dmax=feat["dmax"],
                           reference_distance_median_abs_z=ov.mean_abs_z.median(),
                           overlap_top_zizj_median=ov.overlap_top_zizj.median(),
                           overlap_top_zizj_IQR=f"{ov.overlap_top_zizj.quantile(.25):.3f}-{ov.overlap_top_zizj.quantile(.75):.3f}",
                           overlap_reference_median=ov.overlap_reference.median(),
                           chance=feat["k"] / len(net.ia),
                           anchors_isolated=int(dec_e.reference_isolated.sum()),
                           binary_features=int(dec_e.binary.sum()),
                           enesh_cvR2_E7_all=dec_e.cvR2_E7.median(),
                           enesh_cvR2_D7_all=dec_e.cvR2_D7.median(),
                           enesh_cvR2_COMB_all=dec_e.cvR2_COMB.median(),
                           enesh_cvR2_E7_nonbinary=dec_e.cvR2_E7[nb].median(),
                           enesh_cvR2_D7_nonbinary=dec_e.cvR2_D7[nb].median(),
                           jaccard_cvR2_E7_all=np.nan if dec_j is None else dec_j.cvR2_E7.median(),
                           jaccard_cvR2_D7_all=np.nan if dec_j is None else dec_j.cvR2_D7.median())
                rows.append(row)
                pd.DataFrame(rows).to_csv(os.path.join(a.out, "a3_a4_summary.csv"), index=False)  # saved after every configuration
                print(f"{tag:40s} dist={row['reference_distance_median_abs_z']:.2f} "
                      f"overlap={row['overlap_top_zizj_median']:.3f} (chance {row['chance']:.2f}) "
                      f"cvR2 D7={row['enesh_cvR2_D7_all']:.2f} E7={row['enesh_cvR2_E7_all']:.2f}", flush=True)
                if a.save_enesh:
                    safe = tag.replace("|", "_")
                    en.to_csv(os.path.join(a.out, f"enesh_{safe}.csv"))
                    jd.to_csv(os.path.join(a.out, f"jaccard_{safe}.csv"))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(a.out, "a3_a4_summary.csv"), index=False)
    pd.concat(per_patient).to_csv(os.path.join(a.out, "a3_a4_per_patient_overlap.csv"), index=False)
    plot(summ, os.path.join(a.out, "a3_reference_distance.png"))


def plot(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for meth, mk in (("spcc", "o"), ("ssn", "s")):
        d = s[s.method == meth]
        if d.empty:
            continue
        axes[0].scatter(d.reference_distance_median_abs_z, d.overlap_top_zizj_median / d.chance,
                        marker=mk, label=meth)
        axes[1].scatter(d.reference_distance_median_abs_z, d.enesh_cvR2_D7_all - d.enesh_cvR2_E7_all,
                        marker=mk, label=meth)
        for _, r in d.iterrows():
            axes[0].annotate(f"{r.reference} {r.density}", (r.reference_distance_median_abs_z,
                             r.overlap_top_zizj_median / r.chance), fontsize=6)
    axes[0].axhline(1, color="grey", lw=0.6)
    axes[0].set_xlabel("Reference distance (median mean |z|)")
    axes[0].set_ylabel("Retained-edge overlap with top |z_i z_j|\n(fold over chance)")
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].set_xlabel("Reference distance (median mean |z|)")
    axes[1].set_ylabel("Median CV R² gain: deviation (D7)\nminus matched expression (E7)")
    axes[0].legend(title="method", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)


if __name__ == "__main__":
    main()
