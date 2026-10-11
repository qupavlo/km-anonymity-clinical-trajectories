"""Розширення методу на ВСІ квазіідентифікатори, а не лише коди діагнозів.

Попередній прогін показав, що узагальнення самих діагнозів не рятує: додавання року першої
госпіталізації або однієї процедури зводить захист нанівець (96-99 % ідентифіковано).

Тут жадібний алгоритм працює над спільним набором атрибутів, кожен зі своєю ієрархією:

  діагноз   повний код -> категорія (3 символи) -> блок (1-2 символи)
  процедура повний код -> категорія -> блок
  рік       точний -> п'ятиріччя -> десятиріччя -> прибрано
  тип госп. точний -> {невідкладна, планова} -> прибрано
  страховка точна -> прибрано
  сім. стан точний -> прибрано
  раса      точна -> укрупнена група -> прибрано
  вік       десятиріччя -> двадцятиріччя -> прибрано

Алгоритм незмінний: на кожному кроці піднімаємо той атрибут, що найбільше збільшує множину
кандидатів. Змінюється лише простір дій.

Вихід: results/unit/protect_full.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
Q_KNOWN = (2, 3)
K_TARGET = (5, 20)
N_SAMPLE = 2500
SEED = 20261007
RACE_GROUPS = {"WHITE": "WHITE", "BLACK": "BLACK", "HISPANIC": "HISPANIC", "ASIAN": "ASIAN"}


def gen_code(code, lvl):
    if lvl <= 0:
        return code
    tag, rest = (code[0], code[1:]) if code[0] == "P" else ("", code)
    ver, c = rest.split(":", 1)
    return f"{tag}{ver}:{c[:3] if lvl == 1 else c[:1 if ver == '10' else 2]}"


def gen_year(y, lvl):
    if pd.isna(y):
        return "?"
    y = int(y)
    return {0: str(y), 1: f"{y // 5 * 5}-5", 2: f"{y // 10 * 10}s"}.get(lvl, "*")


def gen_plain(v, lvl):
    return "*" if lvl >= 1 else str(v)


def gen_type(v, lvl):
    if lvl >= 2:
        return "*"
    if lvl == 1:
        return "URGENT" if isinstance(v, str) and ("EMER" in v or "URGENT" in v) else "PLANNED"
    return str(v)


def gen_race(v, lvl):
    if lvl >= 2:
        return "*"
    if lvl == 1:
        s = str(v).upper()
        return next((g for k, g in RACE_GROUPS.items() if k in s), "OTHER")
    return str(v)


def gen_age(b, lvl):
    if lvl >= 2:
        return "*"
    n = int(str(b).rstrip("s"))
    return f"{n // 20 * 20}s+" if lvl == 1 else str(b)


# атрибут -> (функція узагальнення, максимальний рівень)
SPEC = {
    "year": (gen_year, 3), "adm_type": (gen_type, 2), "insurance": (gen_plain, 1),
    "marital": (gen_plain, 1), "race": (gen_race, 2), "band": (gen_age, 2),
}


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pr = pd.read_csv(ROOT / "raw" / "hosp" / "procedures_icd.csv.gz", dtype={"icd_code": str})
    pr["code"] = "P" + pr.icd_version.astype(str) + ":" + pr.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "admission_type",
                               "insurance", "marital_status", "race"], parse_dates=["admittime"])
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
    u["procs"] = pr.groupby("subject_id").code.apply(set).reindex(u.index)
    dpop = dx.groupby("code").subject_id.nunique()
    ppop = pr.groupby("code").subject_id.nunique()
    allpop = {**dpop.to_dict(), **ppop.to_dict()}

    # індекс: (атрибут, рівень, значення) -> множина пацієнтів; для кодів окремо
    print("індекс популяції...", flush=True)
    code_idx = {l: defaultdict(set) for l in range(3)}
    for sid, cs, ps in zip(u.index, u.codes, u.procs):
        items = set(cs) | (ps if isinstance(ps, set) else set())
        for c in items:
            for l in range(3):
                code_idx[l][gen_code(c, l)].add(sid)
    attr_idx = {a: {l: defaultdict(set) for l in range(m + 1)} for a, (_, m) in SPEC.items()}
    for sid in u.index:
        for a, (fn, m) in SPEC.items():
            v = u.at[sid, a]
            for l in range(m + 1):
                attr_idx[a][l][fn(v, l)].add(sid)
    gender_idx = defaultdict(set)
    for sid, g in zip(u.index, u.gender):
        gender_idx[g].add(sid)
    print("готово", flush=True)

    def cand(sid, known_codes, lvl):
        sets = [gender_idx[u.at[sid, "gender"]]]
        for c in known_codes:
            sets.append(code_idx[lvl[("code", c)]][gen_code(c, lvl[("code", c)])])
        for a, (fn, _) in SPEC.items():
            l = lvl[("attr", a)]
            sets.append(attr_idx[a][l][fn(u.at[sid, a], l)])
        sets.sort(key=len)          # починаємо з найменшої множини — перетин швидший
        return len(set.intersection(*sets))

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for q in Q_KNOWN:
        for K in K_TARGET:
            reached, cost, before, raised = [], [], [], []
            for sid in sample:
                cs = u.at[sid, "codes"]
                ps = u.at[sid, "procs"]
                items = list(cs) + (sorted(ps, key=lambda c: ppop.get(c, 0))[:1] if isinstance(ps, set) and ps else [])
                known = sorted(items, key=lambda c: -allpop.get(c, 0))[:q]
                if isinstance(ps, set) and ps:                      # процедуру завжди вважаємо відомою
                    known = list(dict.fromkeys(known + [min(ps, key=lambda c: ppop.get(c, 0))]))
                lvl = {("code", c): 0 for c in known}
                lvl.update({("attr", a): 0 for a in SPEC})
                before.append(cand(sid, known, lvl))
                maxlv = {("code", c): 2 for c in known}
                maxlv.update({("attr", a): m for a, (_, m) in SPEC.items()})
                steps = 0
                while cand(sid, known, lvl) < K and steps < 20:
                    best, gain = None, -1
                    for key in lvl:
                        if lvl[key] >= maxlv[key]:
                            continue
                        trial = dict(lvl)
                        trial[key] += 1
                        v = cand(sid, known, trial)
                        if v > gain:
                            best, gain = key, v
                    if best is None:
                        break
                    lvl[best] += 1
                    steps += 1
                reached.append(cand(sid, known, lvl) >= K)
                cost.append(sum(lvl.values()))
                raised.append(sum(1 for k, v in lvl.items() if v > 0 and k[0] == "attr"))
            rows.append(dict(q_known=q, k_target=K, n=len(sample),
                             median_before=float(np.median(before)),
                             unsafe_before=float(np.mean(np.array(before) < K)),
                             reached=float(np.mean(reached)), mean_cost=float(np.mean(cost)),
                             mean_attrs_raised=float(np.mean(raised))))
            r = rows[-1]
            print(f"q={q} K={K}: було небезпечних {r['unsafe_before']:.1%}, досягнуто {r['reached']:.1%}, "
                  f"сумарна глибина {r['mean_cost']:.2f}, з них атрибутів {r['mean_attrs_raised']:.2f}",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "protect_full.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
