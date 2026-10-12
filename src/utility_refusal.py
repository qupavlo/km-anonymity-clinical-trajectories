"""Utility of a release that ACTUALLY honours the refusal policy (closes M7).

The earlier measurement (`utility_kmanon.py`) computed AUROC on a cohort that still contained the
units the method would refuse to release. It therefore measured the cost of GENERALIZATION, not
the cost of the RELEASE POLICY. That overstates utility and, worse, hides the effect §V-H points
at: refused patients have substantially more admissions, so removing 26-27 % of patients removes
a DISPROPORTIONATELY larger share of the readmission events themselves, i.e. of the target.

Four arms separate two effects that used to be conflated:
  original              all patients, raw codes             (baseline)
  km_m3                 all patients, coarsened             (= what was measured until now)
  original_kept         kept patients only, raw codes       (isolates the effect of the FILTERING)
  km_m3_kept            kept patients only, coarsened       (the real release)

km_m3_kept against original is the full cost of the policy. original_kept against original is the
part of it explained by the filtering alone, with no coarsening at all.

We additionally report what share of the readmission EVENTS the refusal removes, against the
share of patients.

Known defect, corrected in `utility_trainset.py`: the `*_kept` arms here restrict both the
training AND the test part to kept patients, so the arms are scored on different populations with
different base rates. The AUROC gap then partly measures how much easier one task is than the
other rather than what the controller loses.

Output: results/unit/utility_refusal.csv
"""
import pickle
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
N_REPEATS = 25        # 5 gave an MDE of ~0.024 AUROC, i.e. the size of the effect being measured
WINDOW = 30
SEED = 20261007          # same as in utility_kmanon.py, so the cohort coincides
M = 3


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

    def support_exact(g, b, items):
        """No early exit. Needed wherever the value serves as the GAIN of a greedy step:
        `support()` returns the size of the partial intersection as soon as it drops below K,
        i.e. an inflated number in 0..K-1, and the greedy step may then pick the worse code.
        That bug is still present in `utility_kmanon.py`, so the levels there are not the ones
        whose cost the paper claims."""
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

    def km_levels(sid, m):
        """Returns (levels, whether the guarantee was reached, reason, steps).

        The reason separates two cases that used to collapse into a single "unreachable" count:
          hierarchy  the hierarchy is exhausted, nothing left to lift (what the paper claims)
          max_steps  cut off at the step limit, which is an entirely different claim
        A unit with 276 codes cannot finish within 40 steps even in principle.
        """
        cs = sorted(u.at[sid, "codes"])
        g, b = u.at[sid, "gender"], u.at[sid, "band"]
        lvl = {c: 0 for c in cs}
        for step in range(MAX_STEPS):
            w, wset = worst(g, b, cs, lvl, m)
            if w >= K_TARGET:
                return lvl, True, "reached", step
            best, gain = None, -1
            for c in wset:
                if lvl[c] >= MAXLVL:
                    continue
                t = dict(lvl); t[c] += 1
                v = support_exact(g, b, [(x, t[x]) for x in wset])   # NOT support(): see above
                if v > gain:
                    best, gain = c, v
            if best is None:
                return lvl, False, "hierarchy", step
            lvl[best] += 1
        w, _ = worst(g, b, cs, lvl, m)
        return lvl, w >= K_TARGET, ("reached" if w >= K_TARGET else "max_steps"), MAX_STEPS

    sample = rng.choice(u.index.values, min(N_PATIENTS, len(u)), replace=False)
    u = u.loc[sample]
    print(f"subsample {len(u)}; computing levels m={M}...", flush=True)
    cache = RES / f"levels_m{M}_n{N_PATIENTS}_s{SEED}.pkl"
    if cache.exists():
        LV, KEPT, WHY, STEPS = pickle.loads(cache.read_bytes())
        print(f"levels from cache {cache.name}", flush=True)
    else:
        LV, KEPT, WHY, STEPS = {}, {}, {}, {}
        for n, sid in enumerate(u.index):
            if n % 500 == 0:
                print(f"  {n}/{len(u)}", flush=True)
            LV[sid], KEPT[sid], WHY[sid], STEPS[sid] = km_levels(sid, M)
        cache.write_bytes(pickle.dumps((LV, KEPT, WHY, STEPS)))

    kept_ids = {sid for sid, ok in KEPT.items() if ok}
    refused = len(u) - len(kept_ids)
    n_hier = sum(1 for s in WHY.values() if s == "hierarchy")
    n_cap = sum(1 for s in WHY.values() if s == "max_steps")
    print(f"\nrefused {refused}/{len(u)} = {refused/len(u):.1%} patients", flush=True)
    print(f"  of these, hierarchy exhausted: {n_hier} ({n_hier/max(refused,1):.1%} refusals)")
    print(f"  of these, stopped at MAX_STEPS={MAX_STEPS}: {n_cap} ({n_cap/max(refused,1):.1%} refusals)",
          flush=True)
    pd.DataFrame([dict(sid=s, kept=KEPT[s], why=WHY[s], steps=STEPS[s],
                       n_codes=len(u.at[s, "codes"])) for s in u.index]
                 ).to_csv(RES / "refusal_reason.csv", index=False)

    by = dx.groupby("hadm_id").code.apply(list)
    adm = adm[adm.hadm_id.isin(by.index) & adm.subject_id.isin(u.index)].reset_index(drop=True)

    # --- key number for §V-G: how many EVENTS the refusal removes ---
    in_kept = adm.subject_id.isin(kept_ids)
    ev_all, ev_kept = int(adm.y.sum()), int(adm[in_kept].y.sum())
    adm_all, adm_kept = len(adm), int(in_kept.sum())
    print(f"patients kept {len(kept_ids)}/{len(u)} = {len(kept_ids)/len(u):.1%}")
    print(f"admissions          {adm_kept}/{adm_all} = {adm_kept/adm_all:.1%}")
    print(f"READMISSION EVENTS  {ev_kept}/{ev_all} = {ev_kept/ev_all:.1%}")
    print(f"base rate: all {adm.y.mean():.4f}, admitted {adm[in_kept].y.mean():.4f}",
          flush=True)

    def docs(rows, coarsen):
        out = []
        for sid, hid in zip(rows.subject_id, rows.hadm_id):
            lv = LV[sid] if coarsen else {}
            out.append(" ".join(gen_code(c, lv.get(c, 0)).replace(":", "_") for c in by[hid]))
        return out

    arms = [("original", adm, False), ("km_m3", adm, True),
            ("original_kept", adm[in_kept].reset_index(drop=True), False),
            ("km_m3_kept", adm[in_kept].reset_index(drop=True), True)]

    rows, PER_REP = [], {}
    for name, rws, coarsen in arms:
        X, y, groups = docs(rws, coarsen), rws.y.values, rws.subject_id.values
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
            a = np.array(aucs)
            PER_REP[(name, mdl)] = a
            rows.append(dict(arm=name, model=mdl, n_adm=len(rws), n_pat=rws.subject_id.nunique(),
                             base_rate=float(y.mean()), auroc=float(a.mean()),
                             auroc_sd=float(a.std(ddof=1)),          # ddof=1: sample sd over repeats
                             auroc_lo=float(a.mean() - 1.96 * a.std(ddof=1) / np.sqrt(len(a))),
                             auroc_hi=float(a.mean() + 1.96 * a.std(ddof=1) / np.sqrt(len(a))),
                             ap=float(np.mean(aps))))
            print(f"{name:16s} {mdl:7s} n_adm={len(rws):6d} AUROC {a.mean():.4f} "
                  f"± {a.std(ddof=1):.4f}", flush=True)

    # --- PAIRED test where the rows coincide (same splits, same admissions) ---
    from scipy import stats
    pairs = [("km_m3", "original", "coarsening on the full population"),
             ("km_m3_kept", "original_kept", "coarsening within the release")]
    prows = []
    for mdl in ("lr_fix", "lr_cv"):
        for a, b, label in pairs:
            d = PER_REP[(a, mdl)] - PER_REP[(b, mdl)]
            t, pv = stats.ttest_rel(PER_REP[(a, mdl)], PER_REP[(b, mdl)])
            ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            prows.append(dict(model=mdl, test="paired", a=a, b=b, mean_diff=float(d.mean()),
                              ci_lo=float(ci[0]), ci_hi=float(ci[1]), t=float(t), p=float(pv),
                              n_rep=len(d), note=label))
            print(f"[paired] {mdl} {a} − {b}: Δ={d.mean():+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] t={t:.2f} p={pv:.3f}", flush=True)
        # effect of the REFUSAL: different populations, so a paired test does not apply; unpaired,
        # and flagged as such in the output
        for a, b, label in [("original_kept", "original", "effect of exclusion (different populations)")]:
            x, y = PER_REP[(a, mdl)], PER_REP[(b, mdl)]
            t, pv = stats.ttest_ind(x, y, equal_var=False)
            diff = x.mean() - y.mean()
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            prows.append(dict(model=mdl, test="unpaired_welch", a=a, b=b, mean_diff=float(diff),
                              ci_lo=float(diff - 1.96 * se), ci_hi=float(diff + 1.96 * se),
                              t=float(t), p=float(pv), n_rep=len(x), note=label))
            print(f"[unpaired] {mdl} {a} − {b}: Δ={diff:+.4f} "
                  f"CI[{diff-1.96*se:+.4f},{diff+1.96*se:+.4f}] t={t:.2f} p={pv:.4f}", flush=True)
    pd.DataFrame(prows).to_csv(RES / "utility_refusal_tests.csv", index=False)

    out = pd.DataFrame(rows)
    meta = pd.DataFrame([dict(arm="_meta", model="-", n_adm=adm_all, n_pat=len(u),
                              base_rate=float(adm.y.mean()), auroc=np.nan, auroc_sd=np.nan,
                              auroc_lo=refused / len(u),      # share of refused patients
                              auroc_hi=1 - ev_kept / ev_all,  # share of removed EVENTS
                              ap=np.nan)])
    pd.concat([out, meta]).to_csv(RES / "utility_refusal.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))
    print(f"\n_meta: patients refused {refused/len(u):.1%}, "
          f"readmission events removed {1 - ev_kept/ev_all:.1%}")


if __name__ == "__main__":
    main()
