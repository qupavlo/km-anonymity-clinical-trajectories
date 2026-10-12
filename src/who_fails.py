"""Who is left unprotected once the hierarchy is exhausted?

At m = 3, correct k^m-anonymity covers only 73 % of patients. If the rest are systematically more
vulnerable groups (rare diseases, the elderly, minorities), then refusing to release hits exactly
the people the protection was meant to shield most. This echoes Knolle et al. 2026 (Nature), where
under-represented groups are over-represented in the risk tail.

We compare the profiles of the protected and the unprotected: sex, age, race, insurance, number of
diagnoses, number of admissions, rarity of the rarest code.

Output: results/unit/who_fails.csv
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
N_SAMPLE = 3000
SEED = 20261007


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "insurance", "race"],
                      parse_dates=["admittime"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    first = adm.sort_values(["subject_id", "admittime"]).groupby("subject_id").first()

    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band", "anchor_age"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    u["race"] = first.race.reindex(u.index)
    u["insurance"] = first.insurance.reindex(u.index)
    u["n_adm"] = adm.groupby("subject_id").hadm_id.nunique().reindex(u.index)
    pop = dx.groupby("code").subject_id.nunique().to_dict()

    print("population index...", flush=True)
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
        """No early exit: the exact size is needed when the value serves as the gain of a greedy
        step, rather than only for a boolean comparison against K."""
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
        w, wset = (min(sup1.values()), [min(sup1, key=sup1.get)]) if sup1 else (len(u), [])
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

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for n, sid in enumerate(sample):
        if n % 300 == 0:
            print(f"  {n}/{len(sample)}", flush=True)
        cs = sorted(u.at[sid, "codes"])
        if len(cs) < M:
            continue
        g, b = u.at[sid, "gender"], u.at[sid, "band"]
        lvl = {c: 0 for c in cs}
        for _ in range(MAX_STEPS):
            w, wset = worst(g, b, cs, lvl, M)
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
        w, _ = worst(g, b, cs, lvl, M)
        rows.append(dict(sid=sid, protected=w >= K_TARGET, n_codes=len(cs),
                         age=u.at[sid, "anchor_age"], gender=g, race=u.at[sid, "race"],
                         insurance=u.at[sid, "insurance"], n_adm=u.at[sid, "n_adm"],
                         min_pop=min(pop.get(c, 0) for c in cs),
                         n_rare10=sum(1 for c in cs if pop.get(c, 0) <= 10)))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "who_fails.csv", index=False)

    print(f"\nn={len(d)}, protected {d.protected.mean():.1%}\n")
    print("NUMERIC FEATURES (median):")
    for c in ["n_codes", "age", "n_adm", "min_pop", "n_rare10"]:
        a, b_ = d[d.protected][c].median(), d[~d.protected][c].median()
        print(f"  {c:10s} protected {a:8.1f}   UNprotected {b_:8.1f}")
    print("\nSHARE UNPROTECTED WITHIN GROUP:")
    for c in ["gender", "race", "insurance"]:
        g = d.groupby(c).protected.agg(["mean", "size"])
        g = g[g["size"] >= 30].sort_values("mean")
        for k, r in g.head(6).iterrows():
            print(f"  {c:10s} {str(k)[:28]:30s} unprotected {1 - r['mean']:5.1%}  (n={int(r['size'])})")
    print("\nBY NUMBER OF RARE CODES:")
    d["rare_bin"] = pd.cut(d.n_rare10, [-1, 0, 1, 3, 100], labels=["0", "1", "2-3", "4+"])
    print(d.groupby("rare_bin", observed=True).protected.agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
