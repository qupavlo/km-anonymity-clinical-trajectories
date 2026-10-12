"""Geometric metrics of the MIMIC index: naive ones and ones aware of patient ownership.

For every chunk and every k:
  lid_all      LID (Levina-Bickel MLE) over the k nearest neighbours, zero distances skipped
  rk_all       distance to the k-th neighbour
  kocc_all     k-occurrence: how many chunks hold this one in their neighbourhood
               (anti-hubness = few)
  self_ratio   share of neighbours belonging to the same patient
  rk_foreign   distance to the k-th chunk of ANOTHER patient
  d_foreign1   distance to the nearest chunk of another patient
  lid_foreign  LID over foreign neighbours only
  dup_patients how many patients carry an identical text

Output: results/metrics_<model>.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
KS = (10, 20, 50)
KMAX = 256
EPS = 1e-6
BLOCK = 1024


def knn(emb, kmax=KMAX):
    x = torch.from_numpy(emb).to("mps")
    n = x.shape[0]
    idx = np.empty((n, kmax), dtype=np.int32)
    dist = np.empty((n, kmax), dtype=np.float32)
    for s in range(0, n, BLOCK):
        q = x[s:s + BLOCK]
        sim = q @ x.T
        rows = torch.arange(q.shape[0], device="mps")
        sim[rows, rows + s] = -10.0
        v, i = torch.topk(sim, kmax, dim=1)
        idx[s:s + BLOCK] = i.cpu().numpy()
        dist[s:s + BLOCK] = torch.sqrt(torch.clamp(2 - 2 * v, min=0)).cpu().numpy()
        if s % (BLOCK * 40) == 0:
            print(f"  kNN {s}/{n}", flush=True)
    return idx, dist


def lid_mle(d):
    d = np.where(d > EPS, d, np.nan)
    rk = np.nanmax(d, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(d / rk)
        cnt = np.sum(~np.isnan(d), axis=1)
        lid = -(cnt - 1) / np.nansum(logs, axis=1)
    lid[(cnt < 3) | ~np.isfinite(lid)] = np.nan
    return lid


def foreign_first_k(idx, dist, pat, kmax):
    """Vectorised: distances to the first kmax neighbours of a different patient (NaN if too few)."""
    n = idx.shape[0]
    foreign = pat[idx] != pat[:, None]
    rank = np.cumsum(foreign, axis=1) - 1
    fd = np.full((n, kmax), np.nan, dtype=np.float32)
    sel = foreign & (rank < kmax)
    r, c = np.nonzero(sel)
    fd[r, rank[r, c]] = dist[r, c]
    return fd


def main(model="bge"):
    df = pd.read_parquet(ROOT / "data" / "chunks.parquet", columns=["chunk_id", "subject_id", "text"])
    emb = np.load(ROOT / "data" / "emb" / f"{model}.npy")
    pat = pd.factorize(df.subject_id)[0].astype(np.int32)
    n = len(df)

    # how many patients carry an identical text
    tcode = pd.factorize(df.text)[0]
    tp = pd.DataFrame({"t": tcode, "p": pat}).groupby("t").p.nunique()
    dup_patients = tp.values[tcode]

    idx, dist = knn(emb)
    out = {"chunk_id": df.chunk_id.values, "r1": dist[:, 0], "dup_patients": dup_patients}
    same = pat[idx] == pat[:, None]
    fd = foreign_first_k(idx, dist, pat, max(KS))
    for k in KS:
        out[f"lid_all_k{k}"] = lid_mle(dist[:, :k])
        out[f"rk_all_k{k}"] = dist[:, k - 1]
        out[f"kocc_all_k{k}"] = np.bincount(idx[:, :k].ravel(), minlength=n)
        out[f"self_ratio_k{k}"] = same[:, :k].mean(axis=1)
        out[f"rk_foreign_k{k}"] = fd[:, k - 1]
        out[f"lid_foreign_k{k}"] = lid_mle(np.nan_to_num(fd[:, :k], nan=0.0))
        out[f"lam_k{k}"] = 1.0 / (out[f"lid_all_k{k}"] + 1.0)
    out["d_foreign1"] = fd[:, 0]
    res = pd.DataFrame(out)
    (ROOT / "results").mkdir(exist_ok=True)
    res.to_parquet(ROOT / "results" / f"metrics_{model}.parquet", index=False)
    np.save(ROOT / "data" / "emb" / f"knn_idx_{model}.npy", idx[:, :50])
    print(f"{model}: saved; NaN in LID {np.isnan(out['lid_all_k20']).mean():.4f}; "
          f"no other-patient neighbour in top-{KMAX}: {np.isnan(fd[:, 0]).mean():.4f}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["bge"]:
        main(m)
