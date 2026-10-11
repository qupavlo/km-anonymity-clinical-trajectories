"""Адаптивний супротивник: чи витримує захист, якщо знання обирають розумно.

Досі ми фіксували знання супротивника наперед: q найчастіших діагнозів (реалістичний сценарій)
або q найрідкісніших (найгірший за популяційною рідкісністю). Але супротивник, який бачить
випущений набір, може обрати ті q кодів, що РЕАЛЬНО найсильніше звужують коло саме для цього
пацієнта, тобто застосувати нашу ж логіку проти нас.

Порівнюємо чотири стратегії вибору знання, кожну — ПІСЛЯ застосування захисту:
  common   q найчастіших у популяції (реалістичний)
  rare     q найрідкісніших у популяції (наш "найгірший випадок")
  adaptive q кодів, що мінімізують множину кандидатів (повний перебір сполучень)
  oracle   adaptive + вільний вибір серед УСІХ кодів пацієнта, не лише відомих захисту

Ключове питання: захист будувався під припущення про конкретні q кодів. Чи лишається він
дійсним, якщо супротивник знає інші q?

Вихід: results/unit/adaptive.csv
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
Q_KNOWN = (2, 3)
K_TARGET = 5
N_SAMPLE = 1500
MAXLVL = 2
SEED = 20261007
MAX_COMB = 2000          # обмеження перебору сполучень


def gen_code(code, lvl):
    if lvl <= 0:
        return code
    ver, c = code.split(":", 1)
    return f"{ver}:{c[:3] if lvl == 1 else c[:1 if ver == '10' else 2]}"


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    pop = dx.groupby("code").subject_id.nunique().to_dict()

    print("індекс популяції...", flush=True)
    idx = {l: defaultdict(set) for l in range(MAXLVL + 1)}
    for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
        for c in cs:
            for l in range(MAXLVL + 1):
                idx[l][(g, b, gen_code(c, l))].add(sid)

    def cand(g, b, chosen):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in chosen]
        if not sets:
            return len(u)
        sets.sort(key=len)
        return len(set.intersection(*sets))

    def protect(g, b, known, K):
        """Жадібне узагальнення під ЗАДАНЕ знання (як у основному методі)."""
        lvl = {c: 0 for c in known}
        for _ in range(MAXLVL * len(known)):
            if cand(g, b, [(c, lvl[c]) for c in known]) >= K:
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
        return lvl

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for q in Q_KNOWN:
        res = defaultdict(list)
        for sid in sample:
            cs = list(u.at[sid, "codes"])
            g, b = u.at[sid, "gender"], u.at[sid, "band"]
            if len(cs) < q:
                continue
            # захист будуємо під "найрідкісніші" — як у основному методі
            assumed = sorted(cs, key=lambda c: pop.get(c, 0))[:q]
            lvl_all = protect(g, b, assumed, K_TARGET)
            # рівні застосовано до всього юніта: код, який захист не чіпав, лишається на 0
            lvl_full = {c: lvl_all.get(c, 0) for c in cs}

            res["assumed"].append(cand(g, b, [(c, lvl_full[c]) for c in assumed]))

            # супротивник обирає інші q кодів із ЗАХИЩЕНОГО юніта
            for name, pick in (("common", sorted(cs, key=lambda c: -pop.get(c, 0))[:q]),
                               ("rare", sorted(cs, key=lambda c: pop.get(c, 0))[:q])):
                res[name].append(cand(g, b, [(c, lvl_full[c]) for c in pick]))

            # adaptive: перебір сполучень, беремо найгірше для пацієнта
            # сортуємо за ФАКТИЧНИМ support узагальненого коду, а не за pop вихідного:
            # pop не містить узагальнених кодів, тож старий ключ давав усім 10**9
            pool = sorted(cs, key=lambda c: len(idx[lvl_full[c]].get(
                (g, b, gen_code(c, lvl_full[c])), ())))[:14]
            best = 10**9
            n = 0
            for comb in combinations(pool, q):
                best = min(best, cand(g, b, [(c, lvl_full[c]) for c in comb]))
                n += 1
                if n >= MAX_COMB:
                    break
            res["adaptive"].append(best)
        row = dict(q=q, n=len(res["assumed"]), k_target=K_TARGET)
        for m in ("assumed", "common", "rare", "adaptive"):
            a = np.array(res[m])
            row[f"{m}_median"] = float(np.median(a))
            row[f"{m}_safe"] = float(np.mean(a >= K_TARGET))
        rows.append(row)
        print(f"q={q}: захист будувався під 'rare' → безпечних {row['assumed_safe']:.1%}; "
              f"якщо супротивник знає common {row['common_safe']:.1%}, "
              f"rare {row['rare_safe']:.1%}, adaptive {row['adaptive_safe']:.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "adaptive.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
