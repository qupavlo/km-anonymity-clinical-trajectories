"""Independent verifier: does "reached" mean k^m holds, or only that the pool found nothing?

Review claim (CRITICAL). The greedy algorithm looks for the worst subset only among the 14 codes
with the smallest SINGLE-code support. The capped pool is a heuristic, not sound Apriori pruning:
anti-monotonicity says the support of a superset is <= the support of a subset, but NOT that the
pair with minimal support is made of codes with small single support. Two common codes that
hardly ever co-occur give |C| < K and never enter the pool. Since the same truncated search is
both the optimizer and the audit, "reached" may only mean "the heuristic found no violation".

The previous completeness check (`diag_completeness.py`) was tautological: of 94 units only 27 had
more than 14 codes, so for the rest the pool was the whole set. No unit had more than 22 codes,
whereas 72 % of the refused units have more than 22.

Here we test exactly the dangerous regime: LONG trajectories, and not by exhaustive enumeration
(impossible) but by RANDOM sampling of subsets. The randomized search is an independent audit, but
it is itself incomplete: it proves nothing about correctness and the violations it finds are a
lower bound on the true count. What it does give is an honest upper bound on the miss rate (via
the rule of three, below) instead of the claimed "0.0 %".

Output: results/unit/verify_pool.csv
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
M = 3
POOL = 14
MIN_CODES = 23            # exactly the regime the previous check was missing
N_UNITS = 120
N_RANDOM = 30000          # random subsets per unit
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

    def sup(g, b, items):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not out:
                return 0
        return len(out)

    def worst_pool(g, b, cs, lvl, m):
        """What the paper's algorithm does: enumerate only within the pool."""
        sup1 = {c: sup(g, b, [(c, lvl[c])]) for c in cs}
        w, wset = min(sup1.values()), [min(sup1, key=sup1.get)]
        pool = sorted(cs, key=lambda c: sup1[c])[:POOL]
        for j in range(2, m + 1):
            for comb in combinations(pool, j):
                s = sup(g, b, [(c, lvl[c]) for c in comb])
                if s < w:
                    w, wset = s, list(comb)
        return w, wset

    def worst_random(g, b, cs, lvl, m, n_draw):
        """Independent search: random subsets drawn from the WHOLE set, with no truncation."""
        best, best_set = len(u), None
        arr = np.array(cs, dtype=object)
        for _ in range(n_draw):
            j = rng.integers(2, m + 1)
            comb = rng.choice(len(arr), size=int(j), replace=False)
            sel = [arr[i] for i in comb]
            s = sup(g, b, [(c, lvl[c]) for c in sel])
            if s < best:
                best, best_set = s, sel
        return best, best_set

    long_units = [sid for sid in u.index if len(u.at[sid, "codes"]) >= MIN_CODES]
    print(f"units with >= {MIN_CODES} codes: {len(long_units)}", flush=True)
    sample = rng.choice(long_units, min(N_UNITS, len(long_units)), replace=False)

    rows = []
    for n, sid in enumerate(sample):
        cs = sorted(u.at[sid, "codes"])
        g, b = u.at[sid, "gender"], u.at[sid, "band"]
        lvl = {c: 0 for c in cs}
        # run the greedy algorithm exactly as in the paper
        steps = 0
        while steps < MAX_STEPS:
            w, wset = worst_pool(g, b, cs, lvl, M)
            if w >= K_TARGET:
                break
            cand = [c for c in wset if lvl[c] < MAXLVL]
            if not cand:
                break
            best, gain = None, -1
            for c in cand:
                t = dict(lvl); t[c] += 1
                v = sup(g, b, [(x, t[x]) for x in wset])
                if v > gain:
                    best, gain = c, v
            lvl[best] += 1
            steps += 1
        w_pool, _ = worst_pool(g, b, cs, lvl, M)
        reached_pool = w_pool >= K_TARGET
        # INDEPENDENT check of that same final state
        w_rand, rset = worst_random(g, b, cs, lvl, M, N_RANDOM)
        rows.append(dict(sid=int(sid), n_codes=len(cs), steps=steps,
                         worst_pool=int(w_pool), worst_random=int(w_rand),
                         reached_pool=bool(reached_pool),
                         missed=bool(reached_pool and w_rand < K_TARGET)))
        if n % 10 == 0:
            print(f"  {n}/{len(sample)} codes={len(cs):3d} pool={w_pool:4d} rand={w_rand:4d}"
                  f"{'  <-- MISS' if rows[-1]['missed'] else ''}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "verify_pool.csv", index=False)
    ok = out[out.reached_pool]
    missed = int(ok.missed.sum())
    n_ok = len(ok)
    print()
    print(f"units checked: {len(out)}, median codes {out.n_codes.median():.0f}, "
          f"max {out.n_codes.max()}")
    print(f"algorithm declared reached: {n_ok}")
    print(f"of these, the randomized search found a violation: {missed} ({missed/max(n_ok,1):.1%})")
    if missed == 0 and n_ok:
        print(f"95% upper bound (rule of three): {3/n_ok:.1%}")
    print(f"mean understatement of worst support: "
          f"{(out.worst_pool - out.worst_random).mean():.2f}")


if __name__ == "__main__":
    main()
