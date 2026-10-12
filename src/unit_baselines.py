"""Baselines for targeted generalization of a unit.

We check whether the greedy choice gains anything over simpler strategies AT EQUAL COST.
Cost = the number of levels raised (total depth), so the comparison is fair.

  greedy      our method: raise the code that gains the most candidates
  random      raise random codes among those known to the adversary, same depth
  rarest      raise the rarest code (a simple heuristic, no search)
  all_rare    generalize ALL rare codes of the unit by 1 level (the classical approach)
  suppress    remove codes from those known to the adversary instead of generalizing

Output: results/unit/baselines.csv
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RES.mkdir(parents=True, exist_ok=True)
Q_KNOWN = (2, 3)
K_TARGET = (5, 20)
N_SAMPLE = 3000
N_BOOT = 200
RARE_PAT = 10
SEED = 20261006
MAXLVL = 2


def generalize(code, level):
    ver, c = code.split(":", 1)
    if level <= 0:
        return code
    if ver == "10":
        return f"{ver}:{c[:3] if level == 1 else c[:1]}"
    return f"{ver}:{c[:3] if level == 1 else c[:2]}"


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz")
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    unit = dx.groupby("subject_id").code.apply(set)
    u = pd.DataFrame({"codes": unit}).join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    pop = dx.groupby("code").subject_id.nunique()
    rare = set(pop[pop <= RARE_PAT].index)

    print("population index...", flush=True)
    idx = {}
    for lvl in range(MAXLVL + 1):
        m = defaultdict(set)
        for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
            for c in cs:
                m[(g, b, generalize(c, lvl))].add(sid)
        idx[lvl] = m

    def cand(g, b, chosen):
        """chosen: list of (code, level); level None = the code is suppressed (the adversary does not see it)."""
        sets = [idx[l].get((g, b, generalize(c, l)), set()) for c, l in chosen if l is not None]
        return len(set.intersection(*sets)) if sets else len(u)

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)

    def run_greedy(g, b, known, K):
        lvl = {c: 0 for c in known}
        for _ in range(MAXLVL * len(known)):
            if cand(g, b, [(c, lvl[c]) for c in known]) >= K:
                break
            best, gain = None, -1
            for c in known:
                if lvl[c] >= MAXLVL:
                    continue
                v = cand(g, b, [(x, lvl[x] + (x == c)) for x in known])
                if v > gain:
                    best, gain = c, v
            if best is None:
                break
            lvl[best] += 1
        return lvl

    def apply_budget(g, b, known, budget, order):
        """Raise codes in the given order until `budget` levels have been spent."""
        lvl = {c: 0 for c in known}
        spent = 0
        i = 0
        while spent < budget and i < MAXLVL * len(known):
            c = order[i % len(order)]
            if lvl[c] < MAXLVL:
                lvl[c] += 1
                spent += 1
            i += 1
        return lvl

    rows = []
    for q in Q_KNOWN:
        for K in K_TARGET:
            res = defaultdict(list)
            for sid in sample:
                cs = list(u.at[sid, "codes"])
                g, b = u.at[sid, "gender"], u.at[sid, "band"]
                known = sorted(cs, key=lambda c: pop.get(c, 0))[:q]

                lvl_g = run_greedy(g, b, known, K)
                budget = sum(lvl_g.values())
                res["greedy"].append(cand(g, b, [(c, lvl_g[c]) for c in known]) >= K)
                res["greedy_cost"].append(budget)

                # random: same depth, random order
                order = list(rng.permutation(known))
                lvl_r = apply_budget(g, b, known, budget, order)
                res["random"].append(cand(g, b, [(c, lvl_r[c]) for c in known]) >= K)

                # rarest-first: same depth, starting from the rarest code
                order = sorted(known, key=lambda c: pop.get(c, 0))
                lvl_x = apply_budget(g, b, known, budget, order)
                res["rarest"].append(cand(g, b, [(c, lvl_x[c]) for c in known]) >= K)

                # all_rare: every rare code of the unit by 1 level (its cost is counted separately)
                lvl_a = {c: (1 if c in rare else 0) for c in known}
                res["all_rare"].append(cand(g, b, [(c, lvl_a[c]) for c in known]) >= K)
                res["all_rare_cost"].append(sum(1 for c in cs if c in rare))

                # suppress: suppress as many known codes as the levels greedy spent
                n_sup = min(len(known), max(1, budget // MAXLVL))
                sup = sorted(known, key=lambda c: pop.get(c, 0))[:n_sup]
                chosen = [(c, None if c in sup else 0) for c in known]
                res["suppress"].append(cand(g, b, chosen) >= K)
                res["suppress_cost"].append(n_sup)

            row = dict(q_known=q, k_target=K, n=len(sample))
            for m in ("greedy", "random", "rarest", "all_rare", "suppress"):
                a = np.array(res[m], dtype=float)
                boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(N_BOOT)]
                row[m] = a.mean()
                row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boot, [2.5, 97.5])
            row["greedy_cost"] = float(np.mean(res["greedy_cost"]))
            row["all_rare_cost"] = float(np.mean(res["all_rare_cost"]))
            row["suppress_cost"] = float(np.mean(res["suppress_cost"]))
            rows.append(row)
            print(f"q={q} K={K}: greedy {row['greedy']:.3f}  random {row['random']:.3f}  "
                  f"rarest {row['rarest']:.3f}  all_rare {row['all_rare']:.3f} "
                  f"(cost {row['all_rare_cost']:.1f} vs {row['greedy_cost']:.2f})  "
                  f"suppress {row['suppress']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "baselines.csv", index=False)
    print()
    cols = ["q_known", "k_target", "greedy", "random", "rarest", "all_rare", "suppress"]
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
