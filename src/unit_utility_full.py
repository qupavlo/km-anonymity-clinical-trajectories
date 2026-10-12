"""Cost of the EXTENDED method in model quality.

The previous measurement covered generalization of the diagnoses alone. The method now coarsens
the year, admission type, race, insurance, marital status and procedures as well, so the cost has
to be re-measured.

The task is the same (30-day readmission from the codes of the previous admission) so that the
numbers stay comparable with `unit_utility2.py`.

Features: diagnosis and procedure codes plus the categorical attributes as separate tokens. That
way coarsening the attributes also affects the model, not only coarsening the codes.

Output: results/unit/utility_full.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).parent))
from unit_protect_full import SPEC, gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
SEED, Q_KNOWN, K_TARGET, N_REPEATS = 20261007, 3, 5, 5
N_PATIENTS = 6000    # the greedy search is expensive; intersections are sped up by sorting on size
WINDOW = 30


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pr = pd.read_csv(ROOT / "raw" / "hosp" / "procedures_icd.csv.gz", dtype={"icd_code": str})
    pr["code"] = "P" + pr.icd_version.astype(str) + ":" + pr.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
                               "insurance", "marital_status", "race", "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    adm["next_admit"] = adm.groupby("subject_id").admittime.shift(-1)
    adm["gap"] = (adm.next_admit - adm.dischtime).dt.total_seconds() / 86400
    adm["y"] = ((adm.gap >= 0) & (adm.gap <= WINDOW)).astype(int)
    adm = adm[adm.hospital_expire_flag == 0]
    first = adm.groupby("subject_id").first()

    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    u["year"] = first.admittime.dt.year.reindex(u.index)
    u["adm_type"] = first.admission_type.reindex(u.index)
    u["insurance"] = first.insurance.reindex(u.index)
    u["marital"] = first.marital_status.reindex(u.index)
    u["race"] = first.race.reindex(u.index)
    u["procs"] = pr.groupby("subject_id").code.apply(set).reindex(u.index)
    dpop = dx.groupby("code").subject_id.nunique()
    ppop = pr.groupby("code").subject_id.nunique()
    allpop = {**dpop.to_dict(), **ppop.to_dict()}

    print("population index...", flush=True)
    code_idx = {l: defaultdict(set) for l in range(3)}
    for sid, cs, ps in zip(u.index, u.codes, u.procs):
        for c in set(cs) | (ps if isinstance(ps, set) else set()):
            for l in range(3):
                code_idx[l][gen_code(c, l)].add(sid)
    attr_idx = {a: {l: defaultdict(set) for l in range(m + 1)} for a, (_, m) in SPEC.items()}
    for sid in u.index:
        for a, (fn, m) in SPEC.items():
            for l in range(m + 1):
                attr_idx[a][l][fn(u.at[sid, a], l)].add(sid)
    gender_idx = defaultdict(set)
    for sid, g in zip(u.index, u.gender):
        gender_idx[g].add(sid)

    def cand(sid, known, lvl):
        sets = [gender_idx[u.at[sid, "gender"]]]
        sets += [code_idx[lvl[("code", c)]][gen_code(c, lvl[("code", c)])] for c in known]
        sets += [attr_idx[a][lvl[("attr", a)]][fn(u.at[sid, a], lvl[("attr", a)])]
                 for a, (fn, _) in SPEC.items()]
        sets.sort(key=len)          # start from the smallest set: the intersection is faster
        return len(set.intersection(*sets))

    sample = rng.choice(u.index.values, min(N_PATIENTS, len(u)), replace=False)
    u = u.loc[sample]
    print(f"subsample {len(u)} patients; computing policies...", flush=True)
    levels = {}
    for n, sid in enumerate(u.index):
        if n % 2000 == 0:
            print(f"  {n}/{len(u)}", flush=True)
        cs, ps = u.at[sid, "codes"], u.at[sid, "procs"]
        known = sorted(cs, key=lambda c: -allpop.get(c, 0))[:Q_KNOWN]
        if isinstance(ps, set) and ps:
            known = list(dict.fromkeys(known + [min(ps, key=lambda c: ppop.get(c, 0))]))
        lvl = {("code", c): 0 for c in known}
        lvl.update({("attr", a): 0 for a in SPEC})
        maxlv = {("code", c): 2 for c in known}
        maxlv.update({("attr", a): m for a, (_, m) in SPEC.items()})
        steps = 0
        while cand(sid, known, lvl) < K_TARGET and steps < 20:
            best, gain = None, -1
            for key in lvl:
                if lvl[key] >= maxlv[key]:
                    continue
                t = dict(lvl); t[key] += 1
                v = cand(sid, known, t)
                if v > gain:
                    best, gain = key, v
            if best is None:
                break
            lvl[best] += 1; steps += 1
        levels[sid] = lvl

    by_dx = dx.groupby("hadm_id").code.apply(list)
    by_pr = pr.groupby("hadm_id").code.apply(list)
    adm = adm[adm.hadm_id.isin(by_dx.index) & adm.subject_id.isin(u.index)].reset_index(drop=True)
    print(f"admissions {len(adm)}, readmissions {adm.y.mean():.4f}", flush=True)

    def docs(policy):
        out = []
        for sid, hid in zip(adm.subject_id, adm.hadm_id):
            items = list(by_dx.get(hid, [])) + list(by_pr.get(hid, []))
            lvl = levels.get(sid, {})
            if policy == "original":
                toks = items + [f"{a}={u.at[sid, a]}" for a in SPEC]
            elif policy == "full":
                toks = [gen_code(c, lvl.get(("code", c), 0)) for c in items]
                toks += [f"{a}={fn(u.at[sid, a], lvl.get(('attr', a), 0))}" for a, (fn, _) in SPEC.items()]
            else:  # codes_only — the previous method, kept for comparison
                toks = [gen_code(c, lvl.get(("code", c), 0)) for c in items]
                toks += [f"{a}={u.at[sid, a]}" for a in SPEC]
            out.append(" ".join(str(t).replace(":", "_").replace(" ", "_") for t in toks))
        return out

    y, groups = adm.y.values, adm.subject_id.values
    rows = []
    for policy in ("original", "codes_only", "full"):
        X = docs(policy)
        aucs, aps = [], []
        for rep in range(N_REPEATS):
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED + rep).split(X, y, groups))
            vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3)
            Xtr = vec.fit_transform([X[i] for i in tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, y[tr])
            p = clf.predict_proba(vec.transform([X[i] for i in te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
        rows.append(dict(policy=policy, auroc=np.mean(aucs), auroc_sd=np.std(aucs),
                         ap=np.mean(aps), vocab=len(set(w for d in X for w in d.split()))))
        print(f"{policy:12s} AUROC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"AP {np.mean(aps):.4f}  vocab {rows[-1]['vocab']}", flush=True)

    out = pd.DataFrame(rows)
    out["auroc_drop"] = out.auroc.iloc[0] - out.auroc
    out.to_csv(RES / "utility_full.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
