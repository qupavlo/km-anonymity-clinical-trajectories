"""Adaptive adversary: does the protection hold if the background knowledge is chosen cleverly?

So far the adversary's knowledge was fixed in advance: the q most frequent diagnoses (the
realistic scenario) or the q rarest ones (the worst case by population rarity). But an adversary
who sees the released set can pick the q codes that ACTUALLY narrow the candidate set the most
for this particular patient, i.e. turn our own logic against us.

Four knowledge-selection strategies are compared, each evaluated AFTER the protection is applied:
  common   the q most frequent codes in the population (realistic)
  rare     the q rarest codes in the population (our "worst case")
  adaptive the q codes that minimize the candidate set (exhaustive search over combinations)
  oracle   adaptive plus a free choice among ALL of the patient's codes, not only those the
           protection was built for

The key question: the protection was built assuming a specific set of q codes. Does it remain
valid if the adversary knows a different q?

Output: results/unit/adaptive.csv
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
Q_KNOWN = (2, 3)
K_TARGET = 5
N_SAMPLE = 1500
MAXLVL = 2
SEED = 20261007
MAX_COMB = 2000          # cap on the number of combinations enumerated


def gen_code(code, lvl):
    if lvl <= 0:
        return code
    ver, c = code.split(":", 1)
    return f"{ver}:{c[:3] if lvl == 1 else c[:1 if ver == '10' else 2]}"


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

    print("population index...", flush=True)
    idx = {l: defaultdict(set) for l in range(MAXLVL + 1)}
    for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
        for c in cs:
            for l in range(MAXLVL + 1):
                idx[l][(g, b, gen_code(c, l))].add(sid)

    def cand(g, b, chosen):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in chosen]
        if not sets:
            return len(u)
        sets.sort(key=len)
        return len(set.intersection(*sets))

    def protect(g, b, known, K):
        """Greedy generalization against a GIVEN set of known codes (as in the main method)."""
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

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for q in Q_KNOWN:
        res = defaultdict(list)
        for sid in sample:
            cs = list(u.at[sid, "codes"])
            g, b = u.at[sid, "gender"], u.at[sid, "band"]
            if len(cs) < q:
                continue
            # the protection is built assuming the "rarest" codes are known, as in the main method
            assumed = sorted(cs, key=lambda c: pop.get(c, 0))[:q]
            lvl_all = protect(g, b, assumed, K_TARGET)
            # levels apply to the whole unit: a code the protection never touched stays at 0
            lvl_full = {c: lvl_all.get(c, 0) for c in cs}

            res["assumed"].append(cand(g, b, [(c, lvl_full[c]) for c in assumed]))

            # the adversary picks a different q codes from the PROTECTED unit
            for name, pick in (("common", sorted(cs, key=lambda c: -pop.get(c, 0))[:q]),
                               ("rare", sorted(cs, key=lambda c: pop.get(c, 0))[:q])):
                res[name].append(cand(g, b, [(c, lvl_full[c]) for c in pick]))

            # adaptive: enumerate combinations and keep the worst case for the patient.
            # Sort by the ACTUAL support of the generalized code, not by pop of the original one:
            # pop contains no generalized codes, so the old key returned 10**9 for every candidate.
            pool = sorted(cs, key=lambda c: len(idx[lvl_full[c]].get(
                (g, b, gen_code(c, lvl_full[c])), ())))[:14]
            best = 10**9
            n = 0
            for comb in combinations(pool, q):
                best = min(best, cand(g, b, [(c, lvl_full[c]) for c in comb]))
                n += 1
                if n >= MAX_COMB:
                    break
            res["adaptive"].append(best)
        row = dict(q=q, n=len(res["assumed"]), k_target=K_TARGET)
        for m in ("assumed", "common", "rare", "adaptive"):
            a = np.array(res[m])
            row[f"{m}_median"] = float(np.median(a))
            row[f"{m}_safe"] = float(np.mean(a >= K_TARGET))
        rows.append(row)
        print(f"q={q}: protection built for 'rare' -> safe {row['assumed_safe']:.1%}; "
              f"if the adversary knows common {row['common_safe']:.1%}, "
              f"rare {row['rare_safe']:.1%}, adaptive {row['adaptive_safe']:.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "adaptive.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
