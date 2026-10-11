"""Коди проти спільного простору — ПАРНЕ порівняння на одній вибірці (закриває M2-2).

Претензія рецензії. Табл. VII статті порівнювала 73,3 % (лише коди, `kmanon.py`, n=737, POOL=14)
з 81,3 % (спільний простір, `kmanon_full.py`, n=400, POOL=12). Це різні вибірки, різні гілки коду
і різні бюджети пошуку. Менший POOL перебирає меншу частку простору підмножин і тому СЕРТИФІКУЄ
БІЛЬШЕ юнітів — тобто артефакт діє в той самий бік, що й спостережений ефект, і «інтервали не
перетинаються» нічого не доводить.

Тут обидві конфігурації рахуються для ОДНИХ І ТИХ САМИХ пацієнтів, з однаковими POOL, MAX_STEPS,
K і m. Звітуємо парну різницю з довірчим інтервалом по парних спостереженнях, а не два незалежні
відсотки.

Вихід: results/unit/joint_paired.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from unit_protect_full import SPEC, gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
K_TARGET = 5
M_VALUES = (2, 3)
N_SAMPLE = 500
MAX_STEPS = 60        # спільний для обох плечей
POOL = 12             # спільний для обох плечей
SEED = 20261009


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "admittime", "admission_type", "insurance",
                               "marital_status", "race"], parse_dates=["admittime"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    first = adm.sort_values(["subject_id", "admittime"]).groupby("subject_id").first()

    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    u["year"] = first.admittime.dt.year.reindex(u.index)
    u["adm_type"] = first.admission_type.reindex(u.index)
    u["insurance"] = first.insurance.reindex(u.index)
    u["marital"] = first.marital_status.reindex(u.index)
    u["race"] = first.race.reindex(u.index)
    u = u.dropna(subset=["year", "adm_type", "insurance", "marital", "race"])

    ASPEC = {a: v for a, v in SPEC.items() if a != "band"}
    BASE = ["gender", "band"]
    print(f"пацієнтів {len(u)}; будую індекс...", flush=True)

    code_idx = {l: defaultdict(set) for l in range(3)}
    for sid, cs in zip(u.index, u.codes):
        for c in cs:
            for l in range(3):
                code_idx[l][gen_code(c, l)].add(sid)
    base_idx = {a: defaultdict(set) for a in BASE}
    for sid in u.index:
        for a in BASE:
            base_idx[a][u.at[sid, a]].add(sid)
    attr_idx = {a: {l: defaultdict(set) for l in range(m + 1)} for a, (_, m) in ASPEC.items()}
    for sid in u.index:
        for a, (fn, m) in ASPEC.items():
            for l in range(m + 1):
                attr_idx[a][l][fn(u.at[sid, a], l)].add(sid)
    print("готово", flush=True)

    def elem_set(sid, e, lvl):
        kind, name = e
        if kind == "c":
            return code_idx[lvl][gen_code(name, lvl)]
        fn, _ = ASPEC[name]
        return attr_idx[name][lvl][fn(u.at[sid, name], lvl)]

    def sup(sid, elems, lvl, exact):
        sets = [base_idx[a][u.at[sid, a]] for a in BASE]
        sets += [elem_set(sid, e, lvl[e]) for e in elems]
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not exact and len(out) < K_TARGET:
                return len(out)
        return len(out)

    def worst(sid, elems, lvl, m):
        sup1 = {e: sup(sid, [e], lvl, False) for e in elems}
        w, wset = min(sup1.values()), [min(sup1, key=sup1.get)]
        pool = sorted(elems, key=lambda e: sup1[e])[:POOL]
        for j in range(2, m + 1):
            for comb in combinations(pool, j):
                s = sup(sid, list(comb), lvl, False)
                if s < w:
                    w, wset = s, list(comb)
                if w == 0:
                    break
        return w, wset

    def enforce(sid, elems, maxlv, m):
        lvl = {e: 0 for e in elems}
        steps = 0
        while steps < MAX_STEPS:
            w, wset = worst(sid, elems, lvl, m)
            if w >= K_TARGET:
                break
            best, gain = None, -1
            for e in wset:
                if lvl[e] >= maxlv[e]:
                    continue
                t = dict(lvl); t[e] += 1
                v = sup(sid, wset, t, True)
                if v > gain:
                    best, gain = e, v
            if best is None:
                break
            lvl[best] += 1
            steps += 1
        w, _ = worst(sid, elems, lvl, m)
        return w >= K_TARGET, sum(lvl.values()), steps

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows, pair_rows = [], []
    for m in M_VALUES:
        res = defaultdict(list)
        for n, sid in enumerate(sample):
            if n % 50 == 0:
                print(f"  m={m}: {n}/{len(sample)}", flush=True)
            cs = sorted(u.at[sid, "codes"])
            if len(cs) < m:
                continue
            # плече 1: лише коди
            e1 = [("c", c) for c in cs]
            mv1 = {e: 2 for e in e1}
            ok1, cost1, _ = enforce(sid, e1, mv1, m)
            # плече 2: коди + атрибути, ТА САМА вибірка, той самий POOL і ліміт
            e2 = e1 + [("a", a) for a in ASPEC]
            mv2 = dict(mv1); mv2.update({("a", a): ASPEC[a][1] for a in ASPEC})
            ok2, cost2, _ = enforce(sid, e2, mv2, m)
            res["codes"].append(ok1); res["joint"].append(ok2)
            res["cost_codes"].append(cost1); res["cost_joint"].append(cost2)
        a1 = np.array(res["codes"], dtype=float)
        a2 = np.array(res["joint"], dtype=float)
        d = a2 - a1
        # парний тест на бінарних парах: McNemar через біном на розбіжних парах
        b = int(((a2 == 1) & (a1 == 0)).sum())   # joint зміг, codes ні
        c = int(((a2 == 0) & (a1 == 1)).sum())   # codes зміг, joint ні
        pv = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d)) if d.std() > 0 else (0, 0)
        rows.append(dict(m=m, n=len(a1), codes=float(a1.mean()), joint=float(a2.mean()),
                         paired_diff=float(d.mean()), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                         discordant_joint_only=b, discordant_codes_only=c, mcnemar_p=float(pv),
                         cost_codes=float(np.mean(res["cost_codes"])),
                         cost_joint=float(np.mean(res["cost_joint"]))))
        r = rows[-1]
        print(f"m={m}: лише коди {r['codes']:.3f}  спільний {r['joint']:.3f}  "
              f"парна різниця {r['paired_diff']:+.3f} CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
              f"розбіжні пари {b}/{c}  McNemar p={pv:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "joint_paired.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
