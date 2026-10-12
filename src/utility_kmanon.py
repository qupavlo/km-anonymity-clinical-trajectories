"""Utility of CORRECT k^m-anonymity.

The earlier utility measurements covered a weakened version (only q pre-chosen codes were
generalized, 2 out of 12). The correct version coarsens 37-54 % of the trajectory, so the cost
has to be re-measured.

Task: 30-day readmission from the codes of the previous admission — the same as in
`unit_utility2.py`, so that the numbers stay comparable.

Models:
  lr_cv   logistic regression with C tuned by cross-validation (so the regularization effect does
          not mask the cost)
  lr_fix  logistic regression with C=1 (as in the earlier measurements, for comparability)

Policies: original / km2 / km3 / random_km3 (random generalization of the same depth).

Note: `support()` here exits early once the partial intersection drops below K, and that
truncated value is used as the gain of a greedy step. See `utility_refusal.py`, which fixes this
with `support_exact()`; the levels computed here are therefore not the ones the paper costs.

Output: results/unit/utility_kmanon.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from kmanon import gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
K_TARGET, MAXLVL, MAX_STEPS = 5, 2, 40
N_PATIENTS = 8000
N_REPEATS = 5
WINDOW = 30
SEED = 20261007


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                               "hospital_expire_flag"], parse_dates=["admittime", "dischtime"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    adm["next_admit"] = adm.groupby("subject_id").admittime.shift(-1)
    adm["gap"] = (adm.next_admit - adm.dischtime).dt.total_seconds() / 86400
    adm["y"] = ((adm.gap >= 0) & (adm.gap <= WINDOW)).astype(int)
    adm = adm[adm.hospital_expire_flag == 0]

    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]

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

    def km_levels(sid, m):
        cs = sorted(u.at[sid, "codes"])
        g, b = u.at[sid, "gender"], u.at[sid, "band"]
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
                v = support(g, b, [(x, t[x]) for x in wset])
                if v > gain:
                    best, gain = c, v
            if best is None:
                break
            lvl[best] += 1
        return lvl

    sample = rng.choice(u.index.values, min(N_PATIENTS, len(u)), replace=False)
    u = u.loc[sample]
    print(f"subsample {len(u)}; computing levels...", flush=True)
    LV = {}
    for m in (2, 3):
        LV[m] = {}
        for n, sid in enumerate(u.index):
            if n % 1000 == 0:
                print(f"  m={m}: {n}/{len(u)}", flush=True)
            LV[m][sid] = km_levels(sid, m)
    # random policy of the same depth as km3
    LV["rand"] = {}
    for sid in u.index:
        cs = sorted(u.at[sid, "codes"])
        budget = sum(LV[3][sid].values())
        lv = {c: 0 for c in cs}
        order = list(rng.permutation(cs))
        spent = i = 0
        while spent < budget and i < MAXLVL * len(cs):
            c = order[i % len(order)]
            if lv[c] < MAXLVL:
                lv[c] += 1; spent += 1
            i += 1
        LV["rand"][sid] = lv

    by = dx.groupby("hadm_id").code.apply(list)
    adm = adm[adm.hadm_id.isin(by.index) & adm.subject_id.isin(u.index)].reset_index(drop=True)
    print(f"admissions {len(adm)}, readmissions {adm.y.mean():.4f}", flush=True)

    def docs(policy):
        out = []
        for sid, hid in zip(adm.subject_id, adm.hadm_id):
            cs = by[hid]
            lv = {} if policy == "original" else LV[policy].get(sid, {})
            out.append(" ".join(gen_code(c, lv.get(c, 0)).replace(":", "_") for c in cs))
        return out

    y, groups = adm.y.values, adm.subject_id.values
    rows = []
    for policy in ("original", 2, 3, "rand"):
        X = docs(policy)
        name = {"original": "original", 2: "km_m2", 3: "km_m3", "rand": "random_same_cost"}[policy]
        for mdl in ("lr_fix", "lr_cv"):
            aucs, aps = [], []
            for rep in range(N_REPEATS):
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                                random_state=SEED + rep).split(X, y, groups))
                vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3)
                Xtr = vec.fit_transform([X[i] for i in tr])
                clf = (LogisticRegression(max_iter=2000, n_jobs=-1) if mdl == "lr_fix"
                       else LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=3, max_iter=1000,
                                                 n_jobs=-1, scoring="roc_auc"))
                clf.fit(Xtr, y[tr])
                p = clf.predict_proba(vec.transform([X[i] for i in te]))[:, 1]
                aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
            rows.append(dict(policy=name, model=mdl, auroc=np.mean(aucs), auroc_sd=np.std(aucs),
                             ap=np.mean(aps), vocab=len(set(w for d in X for w in d.split()))))
            print(f"{name:18s} {mdl:7s} AUROC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
                  f"AP {np.mean(aps):.4f}", flush=True)

    out = pd.DataFrame(rows)
    base = out[out.policy == "original"].set_index("model").auroc
    out["auroc_drop"] = out.apply(lambda r: base[r.model] - r.auroc, axis=1)
    out.to_csv(RES / "utility_kmanon.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
