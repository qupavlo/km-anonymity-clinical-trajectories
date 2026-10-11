"""Правильна kᵐ-анонімність над СПІЛЬНИМ простором: коди діагнозів і атрибути разом.

Попередній прогін (`unit_protect_full.py`) захищав атрибути, але перевіряв умову лише для одного
наперед заданого набору відомих елементів — та сама послаблена версія, яку зламав адаптивний
супротивник. Тут гарантія вимагається для БУДЬ-ЯКОЇ підмножини розміру <= m, причому підмножина
може містити як коди, так і атрибути.

Простір елементів юніта: усі коди діагнозів + 7 атрибутів (стать, вікова група, рік першої
госпіталізації, тип госпіталізації, страховка, сімейний стан, раса). Кожен має свою ієрархію
узагальнення; стать узагальнюється лише до "*".

Вихід: results/unit/kmanon_full.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from unit_protect_full import SPEC, gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
K_TARGET = 5
M_VALUES = (2, 3)
N_SAMPLE = 400
# ASPEC: year, adm_type, insurance, marital, race (band і gender — завжди відомі)
MAX_STEPS = 60
POOL = 12          # скільки найтонших елементів беремо в перебір сполучень
SEED = 20261007


def gen_sex(v, lvl):
    return "*" if lvl >= 1 else str(v)


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

    # демографія (стать, вікова група) вважається ВІДОМОЮ завжди — як у прогоні лише по кодах,
    # щоб числа були прямо порівнянні. Решта атрибутів — елементи, які супротивник може знати.
    ASPEC = {a: v for a, v in SPEC.items() if a != "band"}
    BASE_ATTRS = ["gender", "band"]
    print(f"атрибутів {len(ASPEC)}, пацієнтів {len(u)}; будую індекс...", flush=True)

    code_idx = {l: defaultdict(set) for l in range(3)}
    for sid, cs in zip(u.index, u.codes):
        for c in cs:
            for l in range(3):
                code_idx[l][gen_code(c, l)].add(sid)
    attr_idx_base = {a: defaultdict(set) for a in ("gender", "band")}
    for sid in u.index:
        for a in ("gender", "band"):
            attr_idx_base[a][u.at[sid, a]].add(sid)
    attr_idx = {a: {l: defaultdict(set) for l in range(m + 1)} for a, (_, m) in ASPEC.items()}
    for sid in u.index:
        for a, (fn, m) in ASPEC.items():
            for l in range(m + 1):
                attr_idx[a][l][fn(u.at[sid, a], l)].add(sid)
    ALL = len(u)
    print("готово", flush=True)

    def elem_set(sid, e, lvl):
        """Множина пацієнтів, що збігаються з пацієнтом sid за елементом e на рівні lvl."""
        kind, name = e
        if kind == "c":
            return code_idx[lvl][gen_code(name, lvl)]
        fn, _ = ASPEC[name]
        return attr_idx[name][lvl][fn(u.at[sid, name], lvl)]

    def support_exact(sid, elems, lvl):
        """Без раннього відсікання — для приросту в жадібному кроці."""
        sets = [attr_idx_base[a][u.at[sid, a]] for a in BASE_ATTRS]
        sets += [elem_set(sid, e, lvl[e]) for e in elems]
        sets.sort(key=len)
        out = sets[0]
        for s_ in sets[1:]:
            out = out & s_
        return len(out)

    def support(sid, elems, lvl):
        # демографія завжди у фільтрі: супротивник її знає за припущенням
        sets = [attr_idx_base[a][u.at[sid, a]] for a in BASE_ATTRS]
        sets += [elem_set(sid, e, lvl[e]) for e in elems]
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if len(out) < K_TARGET:
                return len(out)
        return len(out)

    def worst(sid, elems, lvl, m):
        sup1 = {e: support(sid, [e], lvl) for e in elems}
        w = min(sup1.values()); wset = [min(sup1, key=sup1.get)]
        pool = sorted(elems, key=lambda e: sup1[e])[:POOL]
        for j in range(2, m + 1):
            bj, bs = ALL, None
            for comb in combinations(pool, j):
                s = support(sid, list(comb), lvl)
                if s < bj:
                    bj, bs = s, list(comb)
                if bj == 0:
                    break
            if bs is not None and bj < w:
                w, wset = bj, bs
        return w, wset

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for m in M_VALUES:
        reached, levels_used, n_el, before, attrs_raised, codes_raised = [], [], [], [], [], []
        for n, sid in enumerate(sample):
            if n % 50 == 0:
                print(f"  m={m}: {n}/{len(sample)}", flush=True)
            elems = [("c", c) for c in sorted(u.at[sid, "codes"])] + \
                    [("a", a) for a in ASPEC]
            maxlv = {e: (2 if e[0] == "c" else ASPEC[e[1]][1]) for e in elems}
            lvl = {e: 0 for e in elems}
            n_el.append(len(elems))
            w0, _ = worst(sid, elems, lvl, m)
            before.append(w0)
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
                    v = support_exact(sid, wset, t)
                    if v > gain:
                        best, gain = e, v
                if best is None:
                    break
                lvl[best] += 1
                steps += 1
            w, _ = worst(sid, elems, lvl, m)
            reached.append(w >= K_TARGET)
            levels_used.append(sum(lvl.values()))
            attrs_raised.append(sum(1 for e, v in lvl.items() if v > 0 and e[0] == "a"))
            codes_raised.append(sum(1 for e, v in lvl.items() if v > 0 and e[0] == "c"))
        rows.append(dict(m=m, k=K_TARGET, n=len(reached),
                         median_elements=float(np.median(n_el)),
                         unsafe_before=float(np.mean(np.array(before) < K_TARGET)),
                         reached=float(np.mean(reached)),
                         mean_levels=float(np.mean(levels_used)),
                         mean_codes_raised=float(np.mean(codes_raised)),
                         mean_attrs_raised=float(np.mean(attrs_raised)),
                         mean_attrs_total=len(ASPEC)))
        r = rows[-1]
        print(f"m={m}: небезпечних до {r['unsafe_before']:.1%}, досягнуто {r['reached']:.1%}, "
              f"піднято {r['mean_codes_raised']:.1f} кодів і {r['mean_attrs_raised']:.1f} з "
              f"{len(ASPEC)} атрибутів, сумарна глибина {r['mean_levels']:.1f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "kmanon_full.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
