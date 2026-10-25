"""Shared functions for the revision analyses.

Conventions used by every script
--------------------------------
* Expression matrices: rows = genes (Ensembl IDs without version suffix),
  columns = samples, values = log2(CPM + 1) on the recount3 scale
  (i.e. the calibrated TCGA matrix and the native GTEx matrix used in the paper).
* Edge list: CSV with two columns ``gene_a, gene_b`` (the 90,959-edge
  STRING-constrained candidate universe).
* Anchors: plain text file, one gene ID per line (the 3,992 DEG anchors).
* Survival: CSV with columns ``sample, time, event`` (time in months, event 0/1).
* Sample lists: plain text file, one sample ID per line.

All formulas follow the manuscript: population SD (denominator N) for the
reference, Eq. 4 for add-one-in sPCC, Eqs. 5-11 for eNESH.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import sparse

# --------------------------------------------------------------------- I/O


def _clean_gene(ids):
    s = pd.Index(ids).astype(str).str.strip()
    s = s.str.replace(r"^[A-Z]+::", "", regex=True)          # e.g. "REWIRE::ENSG..."
    return s.where(~s.str.startswith("ENSG"), s.str.split(".").str[0])


def read_matrix(path: str) -> pd.DataFrame:
    """Read a matrix and return it as genes x samples, whatever its orientation.

    Works directly on the project files, e.g. STAGE_D_M2_EXPRESSION_RAW.csv
    (patients x genes) or STAGE_C_GTEX_334_REFERENCE_LOG_CPM_MTFIXED.csv (genes x samples).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".tsv", ".txt", ".tab"):
        df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    else:
        df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index, df.columns = _clean_gene(df.index), _clean_gene(df.columns)
    if df.columns.str.startswith("ENSG").mean() > 0.5 and df.index.str.startswith("ENSG").mean() < 0.5:
        df = df.T
    df = df.loc[~df.index.duplicated()]
    return df.apply(pd.to_numeric, errors="coerce")


def read_list(spec: str) -> list[str]:
    """IDs from a text file (one per line), or from a split table.

    ``STAGE_D_LOCKED_SPLIT.csv:development`` returns the patient_id values whose
    ``split`` column equals (or starts with) ``development``.
    """
    path, _, value = spec.rpartition(":") if not os.path.exists(spec) else (spec, "", "")
    if path.lower().endswith(".csv"):
        d = pd.read_csv(path)
        idcol = next((c for c in d.columns if c.lower() in ("patient_id", "sample", "gene", "gene_id", "id")),
                     d.columns[0])
        if value:
            scol = next(c for c in d.columns if c.lower() == "split")
            d = d[d[scol].astype(str).str.lower().str.startswith(value.lower())]
        items = d[idcol].astype(str).tolist()
    else:
        with open(path) as fh:
            items = [ln.strip().split(",")[0] for ln in fh if ln.strip()]
        if items and items[0].lower() in {"gene", "gene_id", "sample", "id", "patient_id"}:
            items = items[1:]
    return list(_clean_gene(items))


def read_edges(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    e = pd.read_csv(path, sep=sep, low_memory=False)
    pairs = [("gene1", "gene2"), ("gene_a", "gene_b"), ("source", "target"), ("node1", "node2")]
    cols = next(([a, b] for a, b in pairs if a in e.columns and b in e.columns), list(e.columns[:2]))
    e = e[cols].copy()
    e.columns = ["gene_a", "gene_b"]
    for c in e.columns:
        e[c] = _clean_gene(e[c])
    return e


def read_survival(path: str) -> pd.DataFrame:
    """Survival table -> index sample, columns time (months), event (0/1).

    Accepts the project column names (patient_id, OS_months, OS_event) or
    generic ones (sample, time, event)."""
    s = pd.read_csv(path)
    low = {c.lower(): c for c in s.columns}
    sid = next((low[c] for c in ("patient_id", "sample", "id") if c in low), s.columns[0])
    tcol = next((low[c] for c in ("os_months", "time", "os_time", "months") if c in low), None)
    ecol = next((low[c] for c in ("os_event", "event", "status", "os_status") if c in low), None)
    out = pd.DataFrame({"time": pd.to_numeric(s[tcol], errors="coerce"),
                        "event": pd.to_numeric(s[ecol], errors="coerce")})
    out.index = s[sid].astype(str)
    out = out.dropna()
    out["event"] = out["event"].astype(int)
    return out


class Network:
    """Gene universe, candidate edges and anchors, aligned to integer indices."""

    def __init__(self, edges: pd.DataFrame, genes_available, anchors):
        avail = set(genes_available)
        e = edges[edges.gene_a.isin(avail) & edges.gene_b.isin(avail)]
        e = e[e.gene_a != e.gene_b].drop_duplicates()
        self.genes = sorted(set(e.gene_a) | set(e.gene_b))
        self.gidx = {g: k for k, g in enumerate(self.genes)}
        self.ia = e.gene_a.map(self.gidx).to_numpy()
        self.ib = e.gene_b.map(self.gidx).to_numpy()
        self.edges = e.reset_index(drop=True)
        self.anchors = [a for a in anchors if a in self.gidx]
        self.aidx = np.array([self.gidx[a] for a in self.anchors])
        G, E = len(self.genes), len(self.ia)
        # gene x edge incidence (both endpoints)
        rows = np.concatenate([self.ia, self.ib])
        cols = np.concatenate([np.arange(E), np.arange(E)])
        self.inc = sparse.csr_matrix((np.ones(2 * E, np.float32), (rows, cols)), shape=(G, E))
        self.anchor_inc = self.inc[self.aidx]
        # gene x gene adjacency of the candidate universe
        adj = sparse.csr_matrix((np.ones(2 * E), (rows, np.concatenate([self.ib, self.ia]))), shape=(G, G))
        adj.data[:] = 1.0
        self.adj = adj
        self.degree = np.asarray(adj.sum(1)).ravel()

    def neighbour_mean(self, M: np.ndarray) -> np.ndarray:
        """Mean of M (genes x samples) over each gene's candidate-universe neighbours."""
        deg = np.where(self.degree == 0, 1, self.degree)
        return (self.adj @ M) / deg[:, None]


# ------------------------------------------------------ reference statistics


def ref_stats(ref: np.ndarray):
    mu = ref.mean(1)
    sd = ref.std(1)  # population SD, denominator N (as in the paper)
    return mu, sd


def edge_corr(ref: np.ndarray, ia, ib, chunk: int = 20000) -> np.ndarray:
    mu, sd = ref_stats(ref)
    sd = np.where(sd == 0, np.nan, sd)
    Z = (ref - mu[:, None]) / sd[:, None]
    out = np.empty(len(ia))
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = (Z[ia[s:s + chunk]] * Z[ib[s:s + chunk]]).mean(1)
    return out


def zscores(x: np.ndarray, mu, sd) -> np.ndarray:
    sd = np.where(sd == 0, np.nan, sd)
    return (x - mu[:, None]) / sd[:, None]


# ------------------------------------------------------- single-sample edges


def pcc_augmented(zi, zj, rho, N):
    """Correlation after adding one sample to an N-sample reference (Eq. S2)."""
    a = N / (N + 1.0)
    return (N * rho + a * zi * zj) / np.sqrt((N + a * zi ** 2) * (N + a * zj ** 2))


def spcc_exact(zi, zj, rho, N):
    """Add-one-in LIONESS::PCC edge (Eq. 4)."""
    return (N + 1.0) * pcc_augmented(zi, zj, rho, N) - N * rho


def spcc_first_order(zi, zj, rho):
    """First-order (influence-function) approximation, Eq. S3."""
    return rho + zi * zj - 0.5 * rho * (zi ** 2 + zj ** 2)


def ssn_z(zi, zj, rho, N):
    """SSN edge statistic (Liu et al. 2016): delta PCC / ((1 - rho^2)/(N - 1))."""
    delta = pcc_augmented(zi, zj, rho, N) - rho
    return delta / ((1.0 - rho ** 2) / (N - 1.0))


def patient_edges(Z: np.ndarray, net: Network, rho: np.ndarray, N, method: str = "spcc") -> np.ndarray:
    """Edge weights, edges x patients."""
    zi, zj = Z[net.ia], Z[net.ib]
    r = rho[:, None] if np.ndim(rho) == 1 else rho
    if method == "spcc":
        return spcc_exact(zi, zj, r, N)
    if method == "ssn":
        return ssn_z(zi, zj, r, N)
    raise ValueError(method)


def top_k_mask(W: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask keeping the k largest |W| per column (per patient) or overall (1-D)."""
    A = np.abs(np.nan_to_num(W, nan=-np.inf))
    if A.ndim == 1:
        m = np.zeros(A.shape, bool)
        m[np.argpartition(-A, k - 1)[:k]] = True
        return m
    m = np.zeros(A.shape, bool)
    idx = np.argpartition(-A, k - 1, axis=0)[:k]
    np.put_along_axis(m, idx, True, axis=0)
    return m


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def corr_pvalues(rho, N):
    from scipy import stats
    r = np.clip(rho, -0.999999, 0.999999)
    t = r * np.sqrt((N - 2) / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(t), N - 2)


# ------------------------------------------------------------------ eNESH


def enesh(net: Network, ref_keep, ref_sign, pat_keep, pat_sign, dmax=None):
    """eNESH (Eqs. 5-11) for every anchor (rows) and patient (columns).

    ref_keep, ref_sign : edges            (retained in reference, sign of rho)
    pat_keep, pat_sign : edges x patients (retained in patient, sign of sPCC)
    dmax               : largest reference neighbourhood; computed if None.
    """
    if dmax is None:
        deg = np.asarray(net.inc[:, ref_keep].sum(1)).ravel()
        dmax = deg.max()
    inR = ref_keep[:, None]
    same = pat_keep & (pat_sign == ref_sign[:, None])
    P = (inR & same).astype(np.float32)
    L = (inR & ~same).astype(np.float32)
    U = (inR | pat_keep).astype(np.float32)
    A = net.anchor_inc
    nP, nL, nU = A @ P, A @ L, A @ U
    with np.errstate(invalid="ignore", divide="ignore"):
        f = nP / nU
        h = nL / nU
    g = nL / dmax
    e = 1.0 - (f - (g + h))
    e[nU == 0] = 0.0
    return np.asarray(e), dmax


def build_network_features(ref_mat: np.ndarray, tum_mat: np.ndarray, net: Network,
                           density: float = 0.20, method: str = "spcc",
                           ref_rule: str = "density", dmax=None, loo: bool = False):
    """Reference network + patient networks + eNESH, under one configuration.

    loo=True treats tum_mat as its own reference (standard leave-one-out LIONESS
    within a tumour cohort): for patient p the reference is the other n-1 tumours.
    Returns dict with z, weights, masks, eNESH and diagnostics.
    """
    E = len(net.ia)
    k = int(round(density * E))
    if not loo:
        N = ref_mat.shape[1]
        mu, sd = ref_stats(ref_mat)
        rho = edge_corr(ref_mat, net.ia, net.ib)
        Z = zscores(tum_mat, mu, sd)
        W = patient_edges(Z, net, rho, N, method)
        rho_ref = rho
    else:
        X = tum_mat
        n = X.shape[1]
        N = n - 1
        S1 = X.sum(1, keepdims=True)
        S2 = (X ** 2).sum(1, keepdims=True)
        mu = (S1 - X) / N                                  # genes x patients
        var = (S2 - X ** 2) / N - mu ** 2
        sd = np.sqrt(np.clip(var, 0, None))
        sd_safe = np.where(sd == 0, np.nan, sd)
        Z = (X - mu) / sd_safe
        Sxy = (X[net.ia] * X[net.ib]).sum(1, keepdims=True)
        cov = (Sxy - X[net.ia] * X[net.ib]) / N - mu[net.ia] * mu[net.ib]
        rho_loo = cov / (sd_safe[net.ia] * sd_safe[net.ib])
        W = patient_edges(Z, net, rho_loo, N, method)
        rho_ref = edge_corr(X, net.ia, net.ib)              # aggregate tumour network
    if ref_rule == "fdr":
        q = bh_fdr(corr_pvalues(np.nan_to_num(rho_ref), N))
        ref_keep = q < 0.05
        k = int(ref_keep.sum())
    else:
        ref_keep = top_k_mask(rho_ref, k)
    pat_keep = top_k_mask(W, k)
    ref_sign = np.sign(rho_ref)
    pat_sign = np.sign(W)
    e, dmax = enesh(net, ref_keep, ref_sign, pat_keep, pat_sign, dmax)
    return dict(Z=Z, W=W, rho=rho_ref, ref_keep=ref_keep, pat_keep=pat_keep,
                enesh=e, dmax=dmax, k=k, N=N)


def overlap_diagnostics(feat: dict, net: Network) -> pd.DataFrame:
    """Per patient: overlap of retained edges with top-|zi zj| edges and with the reference."""
    Z = feat["Z"]
    k = feat["k"]
    zz = np.abs(Z[net.ia] * Z[net.ib])
    top_dev = top_k_mask(zz, k)
    keep = feat["pat_keep"]
    ov_dev = (keep & top_dev).sum(0) / k
    ov_ref = (keep & feat["ref_keep"][:, None]).sum(0) / k
    absz = np.nanmean(np.abs(Z), 0)
    return pd.DataFrame({"overlap_top_zizj": ov_dev, "overlap_reference": ov_ref,
                         "mean_abs_z": absz, "expected_by_chance": k / len(net.ia)})


# ------------------------------------------------------------- C-index


def comparable_pairs(time, event):
    """Harrell-comparable pairs (i has the event and the shorter time; time ties
    count when i had the event and j did not), as in scikit-survival."""
    t = np.asarray(time, float)
    e = np.asarray(event).astype(bool)
    shorter = (t[:, None] < t[None, :]) & e[:, None]
    tie = (t[:, None] == t[None, :]) & e[:, None] & ~e[None, :]
    i, j = np.where(shorter | tie)
    return i, j


def cindex_columns(X: np.ndarray, time, event, chunk: int = 2000) -> np.ndarray:
    """Harrell's C for every column of X (patients x features), higher = higher risk."""
    i, j = comparable_pairs(time, event)
    out = np.empty(X.shape[1])
    for s in range(0, X.shape[1], chunk):
        d = X[i, s:s + chunk] - X[j, s:s + chunk]
        out[s:s + chunk] = ((d > 0).sum(0) + 0.5 * (d == 0).sum(0)) / len(i)
    return out


def cindex(risk, time, event):
    return float(cindex_columns(np.asarray(risk, float)[:, None], time, event)[0])


def pair_counts(risk, time, event):
    i, j = comparable_pairs(time, event)
    r = np.asarray(risk, float)
    d = r[i] - r[j]
    return dict(comparable=len(i), concordant=int((d > 0).sum()),
                discordant=int((d < 0).sum()), tied_risk=int((d == 0).sum()))


# ----------------------------------------------------------- regression


def ols_r2(y: np.ndarray, X: np.ndarray, cv_folds: int = 5, seed: int = 42):
    """In-sample adjusted R^2 and K-fold cross-validated R^2 for one response."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    y, X = y[ok], X[ok]
    n, p = X.shape
    if n <= p + 2 or np.var(y) == 0:
        return np.nan, np.nan
    D = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - D @ beta) ** 2).sum() / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % cv_folds
    press = 0.0
    for f in range(cv_folds):
        tr, te = folds != f, folds == f
        b, *_ = np.linalg.lstsq(D[tr], y[tr], rcond=None)
        press += ((y[te] - D[te] @ b) ** 2).sum()
    return adj, 1 - press / sst
