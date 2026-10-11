"""Ціна ВІДМОВИ, виміряна на спільній тестовій вибірці (закриває CRITICAL рецензії ACPS).

Дефект попереднього заміру. У `utility_refusal.py` плечі `*_kept` обмежували і навчальну, і
ТЕСТОВУ частину до допущених пацієнтів. Отже `original` оцінювався на тесті з базовою часткою
21,3 %, а `released` — на тесті з 11,1 %. Різниця AUROC між ними вимірює, наскільки одна задача
легша за іншу, а не скільки втрачає контролер. Стаття сама це визнавала («a different and easier
population») і все одно подавала 0,033 як ціну — у заголовку, анотації й висновках.

Правильне питання: модель навчають на тому, що дозволено випустити, і застосовують до ВСІХ
пацієнтів. Тож тестова вибірка має лишатися повною, а змінюватись має лише навчальна.

Чотири плеча, УСІ на одному й тому самому тесті (повна популяція, однакове огрублення):
  train_all        навчання на всіх пацієнтах        <- базова лінія
  train_kept       навчання лише на допущених        <- ціна ВІДМОВИ (те, що треба)
  train_all_raw    навчання на всіх, сирі коди       <- ціна ОГРУБЛЕННЯ
  train_kept_raw   навчання на допущених, сирі коди  <- контроль

Різниця train_kept - train_all на спільному тесті і є ціною відмови.

Вихід: results/unit/utility_trainset.csv, utility_trainset_tests.csv
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
        raise SystemExit(f"немає кешу рівнів {cache.name}; спершу запустіть utility_refusal.py")
    LV, KEPT, WHY, STEPS = pickle.loads(cache.read_bytes())
    kept_ids = {sid for sid, ok in KEPT.items() if ok}
    print(f"рівні з кешу: {len(LV)} пацієнтів, допущено {len(kept_ids)}", flush=True)

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
    print(f"госпіталізацій {len(adm)}, базова частка {adm.y.mean():.4f}", flush=True)

    def docs(rows, coarsen):
        return [" ".join(gen_code(c, (LV[sid] if coarsen else {}).get(c, 0)).replace(":", "_")
                         for c in by[hid])
                for sid, hid in zip(rows.subject_id, rows.hadm_id)]

    y, groups = adm.y.values, adm.subject_id.values
    X_gen, X_raw = docs(adm, True), docs(adm, False)
    in_kept = adm.subject_id.isin(kept_ids).values

    # КОНТРОЛЬ: відмова робить дві речі одночасно — прибирає зміщену підмножину
    # І вдвічі зменшує навчальну вибірку. Щоб розділити ці ефекти, додаємо плече
    # з ВИПАДКОВИМ вилученням такої самої частки пацієнтів.
    rng = np.random.default_rng(SEED)
    pat = adm.subject_id.unique()
    n_keep = len(set(pat) & kept_ids)
    rand_keep = set(rng.choice(pat, n_keep, replace=False))
    in_rand = adm.subject_id.isin(rand_keep).values
    print(f"контроль: випадково збережено {n_keep} пацієнтів з {len(pat)}", flush=True)

    arms = [("train_all", X_gen, None), ("train_kept", X_gen, in_kept),
            ("train_random", X_gen, in_rand),
            ("train_all_raw", X_raw, None), ("train_kept_raw", X_raw, in_kept)]

    rows, PER = [], {}
    for name, X, mask in arms:
        aucs, aps, ntr = [], [], []
        for rep in range(N_REPEATS):
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED + rep).split(X, y, groups))
            # ВІДМОВА діє лише на навчальну частину; тест завжди повний
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
    for a, b, label in [("train_kept", "train_all", "ціна ВІДМОВИ (спільний тест)"),
                        ("train_random", "train_all", "КОНТРОЛЬ: та сама к-ть, випадковий відбір"),
                        ("train_kept", "train_random", "чистий ефект ВІДБОРУ, розмір зрівняно"),
                        ("train_all", "train_all_raw", "ціна ОГРУБЛЕННЯ (спільний тест)"),
                        ("train_kept_raw", "train_all_raw", "ціна відмови без огрублення")]:
        d = PER[a] - PER[b]
        t, pv = stats.ttest_rel(PER[a], PER[b])
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
        prows.append(dict(a=a, b=b, mean_diff=float(d.mean()), ci_lo=float(ci[0]),
                          ci_hi=float(ci[1]), t=float(t), p=float(pv), n_rep=len(d), note=label))
        print(f"[парний] {a} − {b}: Δ={d.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"p={pv:.4f}   {label}", flush=True)

    pd.DataFrame(rows).to_csv(RES / "utility_trainset.csv", index=False)
    pd.DataFrame(prows).to_csv(RES / "utility_trainset_tests.csv", index=False)
    print("\nзбережено utility_trainset.csv і utility_trainset_tests.csv")


if __name__ == "__main__":
    main()
