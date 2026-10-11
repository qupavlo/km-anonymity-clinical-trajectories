"""Корисність релізу, який РЕАЛЬНО дотримується політики відмови (закриває M7).

Попередній замір (`utility_kmanon.py`) рахував AUROC на когорті, що містить і ті юніти, яким
метод відмовив би у випуску. Тобто міряно ціну УЗАГАЛЬНЕННЯ, а не ціну RELEASE POLICY.
Це завищує корисність і, що гірше, ховає ефект, на який вказує §V-H: відмовлені пацієнти мають
істотно більше госпіталізацій, тож вилучення 26-27 % пацієнтів вилучає НЕПРОПОРЦІЙНО більшу
частку самих подій реадмісії — цільової змінної.

Чотири лінії розділяють два ефекти, які досі були злиті:
  original              всі пацієнти, сирі коди            (базова лінія)
  km_m3                 всі пацієнти, огрублено            (= те, що міряли досі)
  original_kept         лише допущені, сирі коди           (ізолює ефект ВІДСІВУ)
  km_m3_kept            лише допущені, огрублено           (реальний реліз)

Порівняння km_m3_kept проти original — повна ціна політики. Порівняння original_kept проти
original — та її частина, що пояснюється самим відсівом, без жодного огрублення.

Додатково звітуємо, яку частку ПОДІЙ реадмісії прибирає відмова, проти частки пацієнтів.

Вихід: results/unit/utility_refusal.csv
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
N_REPEATS = 25        # 5 давало MDE ~0.024 AUROC — на межі ефекту, що міряємо
WINDOW = 30
SEED = 20261007          # той самий, що в utility_kmanon.py — когорта збігається
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

    print("індекс популяції...", flush=True)
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
        """Без раннього відсікання. Потрібен там, де значення йде як ПРИРІСТ у жадібному кроці:
        `support()` повертає розмір часткового перетину щойно той падає нижче K, тобто завищене
        число 0..K-1, і жадібний крок може обрати гірший код. У `utility_kmanon.py` цей баг
        лишився, тому рівні там не ті, чию вартість заявляє стаття."""
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
        """Повертає (рівні, чи досягнуто гарантію, причина, кроки).

        Причина розділяє два випадки, які досі зливалися в одне число «unreachable»:
          hierarchy  ієрархію вичерпано — підняти вже нічого (це те, що стаття стверджує)
          max_steps  обрив на ліміті кроків — зовсім інше твердження
        Юніт із 276 кодами не може завершитися за 40 кроків у принципі.
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
                v = support_exact(g, b, [(x, t[x]) for x in wset])   # НЕ support(): див. вище
                if v > gain:
                    best, gain = c, v
            if best is None:
                return lvl, False, "hierarchy", step
            lvl[best] += 1
        w, _ = worst(g, b, cs, lvl, m)
        return lvl, w >= K_TARGET, ("reached" if w >= K_TARGET else "max_steps"), MAX_STEPS

    sample = rng.choice(u.index.values, min(N_PATIENTS, len(u)), replace=False)
    u = u.loc[sample]
    print(f"підвибірка {len(u)}; обчислюю рівні m={M}...", flush=True)
    cache = RES / f"levels_m{M}_n{N_PATIENTS}_s{SEED}.pkl"
    if cache.exists():
        LV, KEPT, WHY, STEPS = pickle.loads(cache.read_bytes())
        print(f"рівні з кешу {cache.name}", flush=True)
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
    print(f"\nвідмовлено {refused}/{len(u)} = {refused/len(u):.1%} пацієнтів", flush=True)
    print(f"  з них ієрархію вичерпано: {n_hier} ({n_hier/max(refused,1):.1%} відмов)")
    print(f"  з них обрив на MAX_STEPS={MAX_STEPS}: {n_cap} ({n_cap/max(refused,1):.1%} відмов)",
          flush=True)
    pd.DataFrame([dict(sid=s, kept=KEPT[s], why=WHY[s], steps=STEPS[s],
                       n_codes=len(u.at[s, "codes"])) for s in u.index]
                 ).to_csv(RES / "refusal_reason.csv", index=False)

    by = dx.groupby("hadm_id").code.apply(list)
    adm = adm[adm.hadm_id.isin(by.index) & adm.subject_id.isin(u.index)].reset_index(drop=True)

    # --- ключове число для §V-G: скільки ПОДІЙ прибирає відмова ---
    in_kept = adm.subject_id.isin(kept_ids)
    ev_all, ev_kept = int(adm.y.sum()), int(adm[in_kept].y.sum())
    adm_all, adm_kept = len(adm), int(in_kept.sum())
    print(f"пацієнтів збережено {len(kept_ids)}/{len(u)} = {len(kept_ids)/len(u):.1%}")
    print(f"госпіталізацій      {adm_kept}/{adm_all} = {adm_kept/adm_all:.1%}")
    print(f"ПОДІЙ реадмісії     {ev_kept}/{ev_all} = {ev_kept/ev_all:.1%}")
    print(f"базова частота: всі {adm.y.mean():.4f}, допущені {adm[in_kept].y.mean():.4f}",
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
                             auroc_sd=float(a.std(ddof=1)),          # ddof=1: 5 повторів
                             auroc_lo=float(a.mean() - 1.96 * a.std(ddof=1) / np.sqrt(len(a))),
                             auroc_hi=float(a.mean() + 1.96 * a.std(ddof=1) / np.sqrt(len(a))),
                             ap=float(np.mean(aps))))
            print(f"{name:16s} {mdl:7s} n_adm={len(rws):6d} AUROC {a.mean():.4f} "
                  f"± {a.std(ddof=1):.4f}", flush=True)

    # --- ПАРНИЙ тест там, де рядки збігаються (ті самі спліти, ті самі госпіталізації) ---
    from scipy import stats
    pairs = [("km_m3", "original", "огрублення на повній популяції"),
             ("km_m3_kept", "original_kept", "огрублення всередині релізу")]
    prows = []
    for mdl in ("lr_fix", "lr_cv"):
        for a, b, label in pairs:
            d = PER_REP[(a, mdl)] - PER_REP[(b, mdl)]
            t, pv = stats.ttest_rel(PER_REP[(a, mdl)], PER_REP[(b, mdl)])
            ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
            prows.append(dict(model=mdl, test="paired", a=a, b=b, mean_diff=float(d.mean()),
                              ci_lo=float(ci[0]), ci_hi=float(ci[1]), t=float(t), p=float(pv),
                              n_rep=len(d), note=label))
            print(f"[парний] {mdl} {a} − {b}: Δ={d.mean():+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] t={t:.2f} p={pv:.3f}", flush=True)
        # ефект ВІДМОВИ: різні популяції, тож парний тест незастосовний — непарний + явна позначка
        for a, b, label in [("original_kept", "original", "ефект відсіву (різні популяції)")]:
            x, y = PER_REP[(a, mdl)], PER_REP[(b, mdl)]
            t, pv = stats.ttest_ind(x, y, equal_var=False)
            diff = x.mean() - y.mean()
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            prows.append(dict(model=mdl, test="unpaired_welch", a=a, b=b, mean_diff=float(diff),
                              ci_lo=float(diff - 1.96 * se), ci_hi=float(diff + 1.96 * se),
                              t=float(t), p=float(pv), n_rep=len(x), note=label))
            print(f"[непарний] {mdl} {a} − {b}: Δ={diff:+.4f} "
                  f"CI[{diff-1.96*se:+.4f},{diff+1.96*se:+.4f}] t={t:.2f} p={pv:.4f}", flush=True)
    pd.DataFrame(prows).to_csv(RES / "utility_refusal_tests.csv", index=False)

    out = pd.DataFrame(rows)
    meta = pd.DataFrame([dict(arm="_meta", model="-", n_adm=adm_all, n_pat=len(u),
                              base_rate=float(adm.y.mean()), auroc=np.nan, auroc_sd=np.nan,
                              auroc_lo=refused / len(u),      # частка відмовлених пацієнтів
                              auroc_hi=1 - ev_kept / ev_all,  # частка вилучених ПОДІЙ
                              ap=np.nan)])
    pd.concat([out, meta]).to_csv(RES / "utility_refusal.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))
    print(f"\n_meta: відмовлено пацієнтів {refused/len(u):.1%}, "
          f"вилучено подій реадмісії {1 - ev_kept/ev_all:.1%}")


if __name__ == "__main__":
    main()
