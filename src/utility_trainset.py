"""Cost of REFUSAL measured on a shared test set (closes the CRITICAL point of the ACPS review).

Defect of the earlier measurement. In `utility_refusal.py` the `*_kept` arms restricted both the
training AND the TEST part to kept patients. So `original` was scored on a test set with a base
rate of 21.3 % and `released` on one with 11.1 %. The AUROC difference between them measures how
much easier one task is than the other, not how much the controller loses. The paper admitted
this itself ("a different and easier population") and still reported 0.033 as the cost, in the
title, the abstract and the conclusions.

The right question: the model is trained on what may be released and then applied to ALL patients.
So the test set must stay complete and only the training set may vary.

Four arms, ALL on one and the same test set (full population, same coarsening):
  train_all        trained on all patients            <- baseline
  train_kept       trained on kept patients only      <- cost of REFUSAL (what we want)
  train_all_raw    trained on all, raw codes          <- cost of COARSENING
  train_kept_raw   trained on kept patients, raw      <- control

The difference train_kept - train_all on the shared test set is the cost of refusal.

Output: results/unit/utility_trainset.csv, utility_trainset_tests.csv
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from kmanon import gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
N_PATIENTS, N_REPEATS, WINDOW, SEED, M = 8000, 25, 30, 20261007, 3


def main():
    cache = RES / f"levels_m{M}_n{N_PATIENTS}_s{SEED}.pkl"
    if not cache.exists():
        raise SystemExit(f"no level cache {cache.name}; run utility_refusal.py first")
    LV, KEPT, WHY, STEPS = pickle.loads(cache.read_bytes())
    kept_ids = {sid for sid, ok in KEPT.items() if ok}
    print(f"levels from cache: {len(LV)} patients, admitted {len(kept_ids)}", flush=True)

    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                               "hospital_expire_flag"], parse_dates=["admittime", "dischtime"])
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    adm["next_admit"] = adm.groupby("subject_id").admittime.shift(-1)
    adm["gap"] = (adm.next_admit - adm.dischtime).dt.total_seconds() / 86400
    adm["y"] = ((adm.gap >= 0) & (adm.gap <= WINDOW)).astype(int)
    adm = adm[adm.hospital_expire_flag == 0]

    by = dx.groupby("hadm_id").code.apply(list)
    adm = adm[adm.hadm_id.isin(by.index) & adm.subject_id.isin(LV.keys())].reset_index(drop=True)
    print(f"admissions {len(adm)}, base rate {adm.y.mean():.4f}", flush=True)

    def docs(rows, coarsen):
        return [" ".join(gen_code(c, (LV[sid] if coarsen else {}).get(c, 0)).replace(":", "_")
                         for c in by[hid])
                for sid, hid in zip(rows.subject_id, rows.hadm_id)]

    y, groups = adm.y.values, adm.subject_id.values
    X_gen, X_raw = docs(adm, True), docs(adm, False)
    in_kept = adm.subject_id.isin(kept_ids).values

    # CONTROL: the refusal does two things at once — it removes a biased subset
    # AND it halves the training set. To separate those effects we add an arm
    # that removes the same share of patients AT RANDOM.
    rng = np.random.default_rng(SEED)
    pat = adm.subject_id.unique()
    n_keep = len(set(pat) & kept_ids)
    rand_keep = set(rng.choice(pat, n_keep, replace=False))
    in_rand = adm.subject_id.isin(rand_keep).values
    print(f"control: randomly kept {n_keep} patients with {len(pat)}", flush=True)

    arms = [("train_all", X_gen, None), ("train_kept", X_gen, in_kept),
            ("train_random", X_gen, in_rand),
            ("train_all_raw", X_raw, None), ("train_kept_raw", X_raw, in_kept)]

    rows, PER = [], {}
    for name, X, mask in arms:
        aucs, aps, ntr = [], [], []
        for rep in range(N_REPEATS):
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED + rep).split(X, y, groups))
            # the REFUSAL applies to the training part only; the test set is always complete
            tr_use = tr[mask[tr]] if mask is not None else tr
            vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3)
            Xtr = vec.fit_transform([X[i] for i in tr_use])
            clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr_use])
            p = clf.predict_proba(vec.transform([X[i] for i in te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
            ntr.append(len(tr_use))
        a = np.array(aucs)
        PER[name] = a
        rows.append(dict(arm=name, n_train=int(np.mean(ntr)), n_test_fixed=len(y) - int(np.mean(ntr)),
                         test_base_rate=float(y.mean()), auroc=float(a.mean()),
                         auroc_sd=float(a.std(ddof=1)), ap=float(np.mean(aps))))
        print(f"{name:16s} n_train≈{np.mean(ntr):7.0f}  AUROC {a.mean():.4f} ± {a.std(ddof=1):.4f}"
              f"  AP {np.mean(aps):.4f}", flush=True)

    prows = []
    for a, b, label in [("train_kept", "train_all", "cost of REFUSAL (shared test set)"),
                        ("train_random", "train_all", "CONTROL: same count, random selection"),
                        ("train_kept", "train_random", "pure SELECTION effect, size matched"),
                        ("train_all", "train_all_raw", "cost of COARSENING (shared test set)"),
                        ("train_kept_raw", "train_all_raw", "cost of refusal without coarsening")]:
        d = PER[a] - PER[b]
        t, pv = stats.ttest_rel(PER[a], PER[b])
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
        prows.append(dict(a=a, b=b, mean_diff=float(d.mean()), ci_lo=float(ci[0]),
                          ci_hi=float(ci[1]), t=float(t), p=float(pv), n_rep=len(d), note=label))
        print(f"[paired] {a} − {b}: Δ={d.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"p={pv:.4f}   {label}", flush=True)

    pd.DataFrame(rows).to_csv(RES / "utility_trainset.csv", index=False)
    pd.DataFrame(prows).to_csv(RES / "utility_trainset_tests.csv", index=False)
    print("\nwrote utility_trainset.csv and utility_trainset_tests.csv")


if __name__ == "__main__":
    main()
