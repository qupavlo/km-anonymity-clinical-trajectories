"""R3 на MIMIC: чи здатне адитивне збурення сховати ізольований запис.

Ізольованість визначаємо відносно: верхній дециль за відстанню до 20-го фрагмента іншого пацієнта.
Порівнюємо розподіли λ за однакового середнього бюджету шуму.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "r3"; RES.mkdir(parents=True, exist_ok=True)
BUDGETS = (0.05, 0.10, 0.20, 0.35, 0.50)
SEED = 20261005; BLOCK = 2048; TOPK = 10; NQ = 2000


def alloc(name, m):
    if name == "uniform": return np.ones(len(m))
    if name == "lid":     return 1.0 / (m.lid_all_k20.fillna(m.lid_all_k20.median()).values + 1.0)
    rk = m.rk_foreign_k20.values
    return rk if name == "rk" else pd.Series(rk).rank(pct=True).values


def perturb(emb, lam, rng):
    z = rng.standard_normal(emb.shape).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    out = emb + lam[:, None].astype(np.float32) * z
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def rank1(pert, orig):
    a = torch.from_numpy(pert).to("mps"); b = torch.from_numpy(orig).to("mps")
    hit = np.empty(len(pert), dtype=bool)
    for s in range(0, len(pert), BLOCK):
        hit[s:s+BLOCK] = (torch.argmax(a[s:s+BLOCK] @ b.T, 1).cpu().numpy()
                          == np.arange(s, min(s+BLOCK, len(pert))))
    return hit


def utility(pert, orig, q):
    a = torch.from_numpy(orig[q]).to("mps"); p = torch.from_numpy(pert).to("mps"); o = torch.from_numpy(orig).to("mps")
    ov = []
    for s in range(0, len(q), 256):
        qq = a[s:s+256]
        c = torch.topk(qq @ o.T, TOPK, 1).indices.cpu().numpy()
        d = torch.topk(qq @ p.T, TOPK, 1).indices.cpu().numpy()
        ov += [len(set(x) & set(y))/TOPK for x, y in zip(c, d)]
    return float(np.mean(ov))


def main(model="bge"):
    m = pd.read_parquet(ROOT/"results"/f"metrics_{model}.parquet")
    emb = np.load(ROOT/"data"/"emb"/f"{model}.npy")
    txt = pd.read_parquet(ROOT/"data"/"chunks.parquet", columns=["text"]).text
    keep = ~txt.duplicated().values
    emb, m = emb[keep], m[keep].reset_index(drop=True)
    iso = m.rk_foreign_k20.values >= np.quantile(m.rk_foreign_k20.values, 0.90)
    print(f"індекс {len(emb)}, ізольованих (верхній дециль) {iso.sum()}", flush=True)
    rng = np.random.default_rng(SEED); q = rng.choice(len(emb), NQ, replace=False)
    rows = []
    for name in ("uniform", "lid", "rk", "rk_rank"):
        w = alloc(name, m); w = w / w.mean()
        for B in BUDGETS:
            lam = (w*B).astype(np.float32)
            p = perturb(emb, lam, np.random.default_rng(SEED))
            h = rank1(p, emb)
            rows.append(dict(policy=name, budget=B,
                             lam_iso=float(lam[iso].mean()), lam_rest=float(lam[~iso].mean()),
                             link_iso=float(h[iso].mean()), link_rest=float(h[~iso].mean()),
                             utility=utility(p, emb, q)))
            print(f"{name:8s} B={B:.2f} λ ізол/решта {rows[-1]['lam_iso']/rows[-1]['lam_rest']:.2f} "
                  f"зіставлено ізол {rows[-1]['link_iso']:.3f} решта {rows[-1]['link_rest']:.3f} "
                  f"корисність {rows[-1]['utility']:.3f}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(RES/f"lambda_{model}.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    for mm in sys.argv[1:] or ["bge"]: main(mm)
