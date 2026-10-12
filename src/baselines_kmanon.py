"""Baselines under the CORRECT k^m-anonymity (all subsets are checked).

The previous baseline table compared variants of the weakened scheme that an adaptive adversary
breaks. Here every policy is scored against the same correct criterion: the guarantee must hold
for any subset of size <= m.

Policies (all spend the same number of generalization levels, the amount that greedy needs):
  greedy    raise the code of the violating subset with the largest gain in support
  random    raise codes at random
  rarest    raise the codes that are rarest in the population
  uniform   raise ALL codes by one level (an approximation of global recoding)

Output: results/unit/baselines_kmanon.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kmanon import gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
K_TARGET, MAXLVL, MAX_STEPS = 5, 2, 40
M_VALUES = (2, 3)
N_SAMPLE = 500
N_BOOT = 300
SEED = 20261008


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    pop = dx.groupby("code").subject_id.nunique().to_dict()

    print("index...", flush=True)
    idx = {l: defaultdict(set) for l in range(MAXLVL + 1)}
    for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
        for c in cs:
            for l in range(MAXLVL + 1):
                idx[l][(g, b, gen_code(c, l))].add(sid)

    def support(g, b, items):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if len(out) < K_TARGET:
                return len(out)
        return len(out)

    def support_exact(g, b, items):
        """No early exit: the exact size is required when the value is used as the gain in the
        greedy step and not merely for a boolean comparison against K."""
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s_ in sets[1:]:
            out = out & s_
        return len(out)

    def worst(g, b, cs, lvl, m):
        sup1 = {c: support(g, b, [(c, lvl[c])]) for c in cs}
        w, wset = min(sup1.values()), [min(sup1, key=sup1.get)]
        pool = sorted(cs, key=lambda c: sup1[c])[:14]
        for j in range(2, m + 1):
            bj, bs = len(u), None
            for comb in combinations(pool, j):
                s = support(g, b, [(c, lvl[c]) for c in comb])
                if s < bj:
                    bj, bs = s, list(comb)
                if bj == 0:
                    break
            if bs is not None and bj < w:
                w, wset = bj, bs
        return w, wset

    def run_greedy(g, b, cs, m):
        lvl = {c: 0 for c in cs}
        for _ in range(MAX_STEPS):
            w, wset = worst(g, b, cs, lvl, m)
            if w >= K_TARGET:
                break
            best, gain = None, -1
            for c in wset:
                if lvl[c] >= MAXLVL:
                    continue
                t = dict(lvl); t[c] += 1
                v = support_exact(g, b, [(x, t[x]) for x in wset])
                if v > gain:
                    best, gain = c, v
            if best is None:
                break
            lvl[best] += 1
        return lvl

    def spend(cs, budget, order):
        lvl = {c: 0 for c in cs}
        sp = i = 0
        while sp < budget and i < MAXLVL * len(cs) * 2:
            c = order[i % len(order)]
            if lvl[c] < MAXLVL:
                lvl[c] += 1; sp += 1
            i += 1
        return lvl

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for m in M_VALUES:
        res = defaultdict(list)
        for n, sid in enumerate(sample):
            if n % 50 == 0:
                print(f"  m={m}: {n}/{len(sample)}", flush=True)
            cs = sorted(u.at[sid, "codes"])
            g, b = u.at[sid, "gender"], u.at[sid, "band"]
            if len(cs) < m:
                continue
            lg = run_greedy(g, b, cs, m)
            budget = sum(lg.values())
            res["greedy"].append(worst(g, b, cs, lg, m)[0] >= K_TARGET)
            res["cost"].append(budget)
            lr = spend(cs, budget, list(rng.permutation(cs)))
            res["random"].append(worst(g, b, cs, lr, m)[0] >= K_TARGET)
            lx = spend(cs, budget, sorted(cs, key=lambda c: pop.get(c, 0)))
            res["rarest"].append(worst(g, b, cs, lx, m)[0] >= K_TARGET)
            lu = {c: 1 for c in cs}
            res["uniform"].append(worst(g, b, cs, lu, m)[0] >= K_TARGET)
            res["uniform_cost"].append(len(cs))
        row = dict(m=m, n=len(res["greedy"]), mean_cost=float(np.mean(res["cost"])),
                   uniform_cost=float(np.mean(res["uniform_cost"])))
        for k in ("greedy", "random", "rarest", "uniform"):
            a = np.array(res[k], dtype=float)
            boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(N_BOOT)]
            row[k] = a.mean()
            row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(boot, [2.5, 97.5])
        rows.append(row)
        print(f"m={m}: greedy {row['greedy']:.3f} random {row['random']:.3f} "
              f"rarest {row['rarest']:.3f} uniform {row['uniform']:.3f} "
              f"(cost greedy {row['mean_cost']:.1f}, uniform {row['uniform_cost']:.1f})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "baselines_kmanon.csv", index=False)
    print()
    print(out[["m", "n", "greedy", "random", "rarest", "uniform", "mean_cost"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
