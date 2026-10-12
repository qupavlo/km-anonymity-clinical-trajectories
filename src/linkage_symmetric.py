"""Symmetric recomputation of the risk table: separate RARITY from COUNT.

Raised in review. In `linkage_fixed.py` scenarios S1-S3 take the patient's MOST COMMON codes
(`common_q`), while S6 takes the RAREST one. That pits a best case for the defender against a
worst case for the adversary, so the contrast "1 rare code beats 3 common ones" may be an
artefact of the selection rather than a property of the data. On top of that, S6 was computed
only over patients who have a rare code (n=34,216), while the table caption claimed n=223,291 —
so the shares were not comparable.

Here we use a full factorial design: selection policy (common | rare) × q (1,2,3).
Then:
  rare_q vs common_q at the SAME q            -> the pure effect of RARITY
  q=1 vs q=2 vs q=3 under the SAME policy     -> the pure effect of COUNT
No "best case against worst case" comparison anywhere.

In addition, every row is computed against TWO denominators:
  n_cond  patients for whom the scenario is defined (as before)
  share_cohort  the same share referred to the FULL cohort -> comparable across rows

Output: results/unit/linkage_symmetric.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RARE_PAT = 10


def main():
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    u["n_adm"] = adm.groupby("subject_id").hadm_id.nunique().reindex(u.index)
    pop = dx.groupby("code").subject_id.nunique().to_dict()
    N = len(u)
    print(f"patients {N}", flush=True)

    demo = defaultdict(set)
    for sid, g, b in zip(u.index, u.gender, u.band):
        demo[(g, b)].add(sid)
    code_idx = defaultdict(set)
    for sid, cs in zip(u.index, u.codes):
        for c in cs:
            code_idx[c].add(sid)
    nadm_idx = defaultdict(set)
    for sid, k in zip(u.index, u.n_adm):
        nadm_idx[k].add(sid)
    print("indexes ready", flush=True)

    def inter(sets):
        sets = sorted(sets, key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not out:
                break
        return len(out)

    def pick(cs, q, policy):
        """q codes of the patient: the most common or the rarest ones.

        None when the patient has fewer than q codes: previously such a patient was handed their
        whole code set under the label "the q most common", which is not the same scenario.
        """
        if len(cs) < q:
            return None
        key = (lambda c: -pop.get(c, 0)) if policy == "common" else (lambda c: pop.get(c, 0))
        return sorted(cs, key=key)[:q]

    rows = []

    # baseline without diagnoses
    sizes = np.array([len(demo[(r.gender, r.band)]) for r in u.itertuples()])
    rows.append(dict(policy="-", q=0, n_cond=len(sizes), median=float(np.median(sizes)),
                     unique=float(np.mean(sizes == 1)),
                     share_cohort=float(np.mean(sizes == 1)),
                     lt5=float(np.mean(sizes < 5))))
    print(f"{'sex+age':22s} n={len(sizes):6d} median {np.median(sizes):7.0f} "
          f"ident. {np.mean(sizes == 1):6.2%}", flush=True)

    # full factorial: policy × q
    for policy in ("common", "rare"):
        for q in (1, 2, 3):
            sizes, n_undef = [], 0
            for r in u.itertuples():
                sel = pick(r.codes, q, policy)
                if sel is None:
                    n_undef += 1
                    continue
                sizes.append(inter([demo[(r.gender, r.band)]] + [code_idx[c] for c in sel]))
            a = np.array(sizes)
            uniq = float(np.mean(a == 1))
            rows.append(dict(policy=policy, q=q, n_cond=len(a), median=float(np.median(a)),
                             unique=uniq,
                             share_cohort=uniq * len(a) / N,   # referred to the full cohort
                             lt5=float(np.mean(a < 5))))
            print(f"{policy+' x'+str(q):22s} n={len(a):6d} median {np.median(a):7.0f} "
                  f"ident. {uniq:6.2%}  (of cohort {uniq*len(a)/N:6.2%})", flush=True)

    # + number of admissions, on top of the q codes (same filter as above)
    for policy in ("common", "rare"):
        for q in (3,):
            sizes = []
            for r in u.itertuples():
                sel = pick(r.codes, q, policy)
                if sel is None:
                    continue
                sizes.append(inter([demo[(r.gender, r.band)], nadm_idx[r.n_adm]]
                                   + [code_idx[c] for c in sel]))
            a = np.array(sizes)
            uniq = float(np.mean(a == 1))
            rows.append(dict(policy=policy + "+nadm", q=q, n_cond=len(a),
                             median=float(np.median(a)), unique=uniq,
                             share_cohort=uniq * len(a) / N, lt5=float(np.mean(a < 5))))
            print(f"{policy+'x'+str(q)+' +n_adm':22s} n={len(a):6d} median {np.median(a):7.0f} "
                  f"ident. {uniq:6.2%}", flush=True)

    # separately: exactly one code rare in ABSOLUTE terms (pop <= RARE_PAT), as S6 had it
    rare = {c for c, n in pop.items() if n <= RARE_PAT}
    sizes = []
    for r in u.itertuples():
        rr = [c for c in r.codes if c in rare]
        if not rr:
            continue
        sizes.append(inter([demo[(r.gender, r.band)],
                            code_idx[min(rr, key=lambda c: pop.get(c, 0))]]))
    a = np.array(sizes)
    uniq = float(np.mean(a == 1))
    rows.append(dict(policy="rare_abs", q=1, n_cond=len(a), median=float(np.median(a)),
                     unique=uniq, share_cohort=uniq * len(a) / N,
                     lt5=float(np.mean(a < 5))))
    print(f"{'one code pop<=10':22s} n={len(a):6d} median {np.median(a):7.0f} "
          f"ident. {uniq:6.2%}  (of cohort {uniq*len(a)/N:6.2%})", flush=True)
    print(f"   share of cohort carrying such a code at all: {len(a)/N:.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "linkage_symmetric.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))

    # direct answers to the questions posed in the paper
    g = {(r.policy, r.q): r for r in out.itertuples()}
    print("\n--- RARITY EFFECT (same q, different policy) ---")
    for q in (1, 2, 3):
        c, r_ = g[("common", q)], g[("rare", q)]
        print(f"  q={q}: common {c.unique:6.2%} -> rare {r_.unique:6.2%} "
              f"(×{r_.unique/max(c.unique,1e-9):.1f})")
    print("--- CARDINALITY EFFECT (same policy, different q) ---")
    for policy in ("common", "rare"):
        v = [g[(policy, q)].unique for q in (1, 2, 3)]
        print(f"  {policy}: {v[0]:6.2%} -> {v[1]:6.2%} -> {v[2]:6.2%} "
              f"(×{v[2]/max(v[0],1e-9):.1f} from q=1 to q=3)")


if __name__ == "__main__":
    main()
