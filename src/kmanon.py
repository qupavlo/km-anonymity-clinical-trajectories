"""Коректна kᵐ-анонімність: гарантія для БУДЬ-ЯКОЇ підмножини розміру <= m.

Попередня реалізація забезпечувала умову лише для однієї наперед заданої множини відомих кодів,
і адаптивний супротивник її ламав (21 % безпечних замість 99 %). Правильне означення
(Terrovitis et al.): для кожної множини з m або менше елементів має існувати щонайменше k
пацієнтів, що її містять.

Жадібний алгоритм з перевіркою всіх сполучень. Застосовуємо монотонність Apriori: якщо множина
розміру j уже порушує умову, порушують і всі її надмножини, тому шукаємо ПОРУШЕННЯ знизу вгору
і зупиняємося на першому знайденому.

Вихід: results/unit/kmanon.csv
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
M_VALUES = (2, 3)
K_TARGET = 5
N_SAMPLE = 800
MAXLVL = 2
MAX_STEPS = 40
SEED = 20261007


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
    print("готово", flush=True)

    def support(g, b, items):
        """items: список (код, рівень)."""
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if len(out) < K_TARGET:        # раннє відсікання
                return len(out)
        return len(out)

    def support_exact(g, b, items):
        """Без раннього відсікання: точний розмір потрібен, коли значення використовується
        як приріст у жадібному кроці, а не лише для булевого порівняння з K."""
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s_ in sets[1:]:
            out = out & s_
        return len(out)

    def worst_violation(g, b, cs, lvl, m):
        """Знайти підмножину розміру <= m з найменшою підтримкою. Apriori: шукаємо знизу вгору,
        і до рівня j+1 беремо лише коди, що входили в "тонкі" множини рівня j."""
        # рівень 1
        sup1 = {c: support(g, b, [(c, lvl[c])]) for c in cs}
        worst = min(sup1.values()) if sup1 else len(u)
        worst_set = [min(sup1, key=sup1.get)] if sup1 else []
        if worst < K_TARGET and m == 1:
            return worst, worst_set
        # кандидати: коди з найменшою підтримкою (решта не дадуть порушення швидше)
        pool = sorted(cs, key=lambda c: sup1[c])[:14]
        for j in range(2, m + 1):
            best_j, best_set = len(u), None
            for comb in combinations(pool, j):
                s = support(g, b, [(c, lvl[c]) for c in comb])
                if s < best_j:
                    best_j, best_set = s, list(comb)
                if best_j == 0:
                    break
            if best_set is not None and best_j < worst:
                worst, worst_set = best_j, best_set
        return worst, worst_set

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for m in M_VALUES:
        reached, cost, n_changed, before, ncodes = [], [], [], [], []
        for n, sid in enumerate(sample):
            if n % 100 == 0:
                print(f"  m={m}: {n}/{len(sample)}", flush=True)
            cs = sorted(u.at[sid, "codes"])
            g, b = u.at[sid, "gender"], u.at[sid, "band"]
            if len(cs) < m:
                continue
            ncodes.append(len(cs))
            lvl = {c: 0 for c in cs}
            w0, _ = worst_violation(g, b, cs, lvl, m)
            before.append(w0)
            steps = 0
            while steps < MAX_STEPS:
                w, wset = worst_violation(g, b, cs, lvl, m)
                if w >= K_TARGET:
                    break
                # піднімаємо той код з порушеної множини, що дає найбільший приріст
                best, gain = None, -1
                for c in wset:
                    if lvl[c] >= MAXLVL:
                        continue
                    t = dict(lvl); t[c] += 1
                    v = support_exact(g, b, [(x, t[x]) for x in wset])
                    if v > gain:
                        best, gain = c, v
                if best is None:
                    break
                lvl[best] += 1
                steps += 1
            w, _ = worst_violation(g, b, cs, lvl, m)
            reached.append(w >= K_TARGET)
            cost.append(sum(lvl.values()))
            n_changed.append(sum(1 for v in lvl.values() if v > 0))
        rows.append(dict(m=m, k=K_TARGET, n=len(reached),
                         median_codes=float(np.median(ncodes)),
                         median_worst_before=float(np.median(before)),
                         unsafe_before=float(np.mean(np.array(before) < K_TARGET)),
                         reached=float(np.mean(reached)),
                         mean_levels=float(np.mean(cost)),
                         mean_codes_changed=float(np.mean(n_changed)),
                         share_codes_changed=float(np.mean(np.array(n_changed) / np.array(ncodes)))))
        r = rows[-1]
        print(f"m={m}: небезпечних до {r['unsafe_before']:.1%}, досягнуто {r['reached']:.1%}, "
              f"змінено {r['mean_codes_changed']:.1f} з {r['median_codes']:.0f} кодів "
              f"({r['share_codes_changed']:.1%})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "kmanon.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
