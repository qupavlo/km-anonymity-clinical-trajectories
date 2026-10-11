"""Ціна захисту в якості навчання, задача без витоку.

Задача: чи буде ПОВТОРНА госпіталізація протягом 30 днів після виписки.
Ознаки — коди ПОПЕРЕДНЬОЇ госпіталізації, ціль лежить у майбутньому відносно ознак,
тому кодів, що фіксують результат, тут бути не може (на відміну від летальності,
де нагору вийшли «паліативна допомога» і «не реанімувати»).

Виключаємо госпіталізації, що закінчилися смертю: для них повторної бути не може.

Порівнюємо ті самі політики, що й раніше: original / greedy / random / all_level1.

Вихід: results/unit/utility_readmission.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
SEED = 20261006
Q_KNOWN, K_TARGET, MAXLVL, N_REPEATS = 3, 5, 2, 5
WINDOW_DAYS = 30


def generalize(code, level):
    ver, c = code.split(":", 1)
    if level <= 0:
        return code
    if ver == "10":
        return f"{ver}:{c[:3] if level == 1 else c[:1]}"
    return f"{ver}:{c[:3] if level == 1 else c[:2]}"


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

    # ціль: наступна госпіталізація того самого пацієнта протягом 30 днів
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    adm["next_admit"] = adm.groupby("subject_id").admittime.shift(-1)
    adm["gap"] = (adm.next_admit - adm.dischtime).dt.total_seconds() / 86400
    adm["y"] = ((adm.gap >= 0) & (adm.gap <= WINDOW_DAYS)).astype(int)
    adm = adm[adm.hospital_expire_flag == 0]          # померлі не можуть повернутися

    unit = dx.groupby("subject_id").code.apply(set)
    u = pd.DataFrame({"codes": unit}).join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    pop = dx.groupby("code").subject_id.nunique()

    print("індекс популяції...", flush=True)
    idx = {}
    for lvl in range(MAXLVL + 1):
        m = defaultdict(set)
        for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
            for c in cs:
                m[(g, b, generalize(c, lvl))].add(sid)
        idx[lvl] = m

    def cand(g, b, chosen):
        sets = [idx[l].get((g, b, generalize(c, l)), set()) for c, l in chosen]
        return len(set.intersection(*sets)) if sets else len(u)

    print("обчислюю політики...", flush=True)
    lv_greedy, lv_random = {}, {}
    for sid in u.index:
        cs = list(u.at[sid, "codes"])
        g, b = u.at[sid, "gender"], u.at[sid, "band"]
        known = sorted(cs, key=lambda c: pop.get(c, 0))[:Q_KNOWN]
        lvl = {c: 0 for c in known}
        for _ in range(MAXLVL * len(known)):
            if cand(g, b, [(c, lvl[c]) for c in known]) >= K_TARGET:
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
        lv_greedy[sid] = lvl
        budget, rl = sum(lvl.values()), {c: 0 for c in known}
        order = list(rng.permutation(known))
        spent, i = 0, 0
        while spent < budget and i < MAXLVL * len(known):
            c = order[i % len(order)]
            if rl[c] < MAXLVL:
                rl[c] += 1
                spent += 1
            i += 1
        lv_random[sid] = rl

    by_adm = dx[dx.subject_id.isin(u.index)].groupby("hadm_id").code.apply(list)
    adm = adm[adm.hadm_id.isin(by_adm.index) & adm.subject_id.isin(u.index)].reset_index(drop=True)
    print(f"госпіталізацій {len(adm)}, повторних протягом {WINDOW_DAYS} днів "
          f"{adm.y.mean():.4f} ({adm.y.sum()})", flush=True)

    def docs(policy):
        out = []
        for sid, hid in zip(adm.subject_id, adm.hadm_id):
            cs = by_adm[hid]
            if policy == "original":
                g = cs
            elif policy == "all_level1":
                g = [generalize(c, 1) for c in cs]
            else:
                lv = (lv_greedy if policy == "greedy" else lv_random)[sid]
                g = [generalize(c, lv.get(c, 0)) for c in cs]
            out.append(" ".join(x.replace(":", "_") for x in g))
        return out

    y, groups = adm.y.values, adm.subject_id.values
    rows = []
    for policy in ("original", "greedy", "random", "all_level1"):
        X_txt = docs(policy)
        aucs, aps = [], []
        for rep in range(N_REPEATS):
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=SEED + rep).split(X_txt, y, groups))
            vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3)
            Xtr = vec.fit_transform([X_txt[i] for i in tr])
            clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, y[tr])
            p = clf.predict_proba(vec.transform([X_txt[i] for i in te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
        rows.append(dict(policy=policy, auroc=np.mean(aucs), auroc_sd=np.std(aucs),
                         ap=np.mean(aps), vocab=len(set(w for d in X_txt for w in d.split()))))
        print(f"{policy:12s} AUROC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"AP {np.mean(aps):.4f}  словник {rows[-1]['vocab']}", flush=True)

    out = pd.DataFrame(rows)
    out["auroc_drop"] = out.auroc.iloc[0] - out.auroc
    out.to_csv(RES / "utility_readmission.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
