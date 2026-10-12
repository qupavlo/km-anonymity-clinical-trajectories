"""Correct k^m-anonymity at K=20, plus an honest baseline for the selection rule.

Two gaps found by the review.

1. K=20 had been run only for the WEAKENED scheme (`protect.csv`, `baselines.csv`). Under the
   correct k^m only K=5 existed, even though K=20 is closer to the thresholds actually used in
   real releases.

2. The greedy-versus-random comparison in `baselines_kmanon.py` really contrasts feedback with
   the absence of feedback rather than two selection rules: `spend()` hands out the budget
   blindly and never checks the criterion, whereas greedy checks it at every step. An honest
   baseline, `random_in_violating`, is added here: the same feedback loop and the same number of
   steps, but the code to raise is chosen AT RANDOM from the violating set instead of by maximum
   gain. The difference between it and greedy is the contribution of the selection rule alone.

Output: results/unit/kmanon_k20.csv
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
MAXLVL, MAX_STEPS, POOL = 2, 40, 14
M_VALUES = (2, 3)
K_VALUES = (5, 20)
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

    print("population index...", flush=True)
    idx = {l: defaultdict(set) for l in range(MAXLVL + 1)}
    for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
        for c in cs:
            for l in range(MAXLVL + 1):
                idx[l][(g, b, gen_code(c, l))].add(sid)
    print("done", flush=True)

    def sup_exact(g, b, items):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
        return len(out)

    def worst(g, b, cs, lvl, m, K):
        sup1 = {c: sup_exact(g, b, [(c, lvl[c])]) for c in cs}
        w, wset = min(sup1.values()), [min(sup1, key=sup1.get)]
        pool = sorted(cs, key=lambda c: sup1[c])[:POOL]
        for j in range(2, m + 1):
            for comb in combinations(pool, j):
                s = sup_exact(g, b, [(c, lvl[c]) for c in comb])
                if s < w:
                    w, wset = s, list(comb)
                if w == 0:
                    break
        return w, wset

    def run(g, b, cs, m, K, rule):
        """rule: 'greedy' takes the maximum gain; 'random' takes a random code from the violating
        set. Both see the same feedback and share the same step limit."""
        lvl = {c: 0 for c in cs}
        steps = 0
        while steps < MAX_STEPS:
            w, wset = worst(g, b, cs, lvl, m, K)
            if w >= K:
                break
            cand = [c for c in wset if lvl[c] < MAXLVL]
            if not cand:
                break
            if rule == "greedy":
                best, gain = None, -1
                for c in cand:
                    t = dict(lvl); t[c] += 1
                    v = sup_exact(g, b, [(x, t[x]) for x in wset])
                    if v > gain:
                        best, gain = c, v
            else:
                best = cand[int(rng.integers(len(cand)))]
            lvl[best] += 1
            steps += 1
        w, _ = worst(g, b, cs, lvl, m, K)
        return w >= K, sum(lvl.values()), steps

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for K in K_VALUES:
        for m in M_VALUES:
            res = defaultdict(list)
            for n, sid in enumerate(sample):
                if n % 100 == 0:
                    print(f"  K={K} m={m}: {n}/{len(sample)}", flush=True)
                cs = sorted(u.at[sid, "codes"])
                g, b = u.at[sid, "gender"], u.at[sid, "band"]
                if len(cs) < m:
                    continue
                ok_g, cost_g, st_g = run(g, b, cs, m, K, "greedy")
                ok_r, cost_r, st_r = run(g, b, cs, m, K, "random")
                res["greedy"].append(ok_g); res["cost_greedy"].append(cost_g)
                res["random_in_viol"].append(ok_r); res["cost_random"].append(cost_r)
            row = dict(K=K, m=m, n=len(res["greedy"]),
                       mean_cost_greedy=float(np.mean(res["cost_greedy"])),
                       mean_cost_random=float(np.mean(res["cost_random"])))
            for k in ("greedy", "random_in_viol"):
                a = np.array(res[k], dtype=float)
                boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(N_BOOT)]
                row[k] = float(a.mean())
                row[f"{k}_lo"], row[f"{k}_hi"] = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            rows.append(row)
            print(f"K={K} m={m}: greedy {row['greedy']:.3f} "
                  f"[{row['greedy_lo']:.3f},{row['greedy_hi']:.3f}]  "
                  f"random-in-violating {row['random_in_viol']:.3f} "
                  f"[{row['random_in_viol_lo']:.3f},{row['random_in_viol_hi']:.3f}]  "
                  f"cost {row['mean_cost_greedy']:.1f} / {row['mean_cost_random']:.1f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "kmanon_k20.csv", index=False)
    print()
    print(out[["K", "m", "n", "greedy", "random_in_viol",
               "mean_cost_greedy", "mean_cost_random"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
