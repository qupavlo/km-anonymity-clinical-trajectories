"""Незалежний верифікатор: чи «reached» означає виконання kᵐ, чи лише «pool нічого не знайшов».

Претензія рецензії (CRITICAL). Жадібний алгоритм шукає найгіршу підмножину лише серед 14 кодів із
найменшим ОДИНИЧНИМ support. Anti-monotonicity цього не виправдовує: вона каже, що support
надмножини <= support підмножини, але НЕ що пара з мінімальним support складається з кодів із малим
одиничним support. Два поширені коди, які майже не співіснують, дають |C| < K і в pool не
потрапляють. Оскільки той самий обрізаний пошук є і оптимізатором, і аудитом, «reached» може
означати «евристика не знайшла порушення».

Попередня перевірка повноти (`diag_completeness.py`) була тавтологічною: з 94 юнітів лише 27 мали
понад 14 кодів, тож для решти pool дорівнював усьому набору. Жоден юніт не мав понад 22 коди, тоді
як 72 % відмовлених мають понад 22.

Тут перевіряємо саме небезпечний режим: ДОВГІ траєкторії, і не повним перебором (неможливо), а
ВИПАДКОВОЮ вибіркою підмножин. Випадковий пошук не доводить коректності, але дає чесну верхню межу
на частку пропусків замість заявлених «0,0 %».

Вихід: results/unit/verify_pool.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kmanon import gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
K_TARGET, MAXLVL, MAX_STEPS = 5, 2, 40
M = 3
POOL = 14
MIN_CODES = 23            # саме той режим, якого бракувало перевірці
N_UNITS = 120
N_RANDOM = 30000          # випадкових підмножин на юніт
SEED = 20261008


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

    print("індекс популяції...", flush=True)
    idx = {l: defaultdict(set) for l in range(MAXLVL + 1)}
    for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
        for c in cs:
            for l in range(MAXLVL + 1):
                idx[l][(g, b, gen_code(c, l))].add(sid)

    def sup(g, b, items):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not out:
                return 0
        return len(out)

    def worst_pool(g, b, cs, lvl, m):
        """Те, що робить алгоритм статті: перебір лише в межах pool."""
        sup1 = {c: sup(g, b, [(c, lvl[c])]) for c in cs}
        w, wset = min(sup1.values()), [min(sup1, key=sup1.get)]
        pool = sorted(cs, key=lambda c: sup1[c])[:POOL]
        for j in range(2, m + 1):
            for comb in combinations(pool, j):
                s = sup(g, b, [(c, lvl[c]) for c in comb])
                if s < w:
                    w, wset = s, list(comb)
        return w, wset

    def worst_random(g, b, cs, lvl, m, n_draw):
        """Незалежний пошук: випадкові підмножини з УСЬОГО набору, без обрізання."""
        best, best_set = len(u), None
        arr = np.array(cs, dtype=object)
        for _ in range(n_draw):
            j = rng.integers(2, m + 1)
            comb = rng.choice(len(arr), size=int(j), replace=False)
            sel = [arr[i] for i in comb]
            s = sup(g, b, [(c, lvl[c]) for c in sel])
            if s < best:
                best, best_set = s, sel
        return best, best_set

    long_units = [sid for sid in u.index if len(u.at[sid, "codes"]) >= MIN_CODES]
    print(f"юнітів з >= {MIN_CODES} кодами: {len(long_units)}", flush=True)
    sample = rng.choice(long_units, min(N_UNITS, len(long_units)), replace=False)

    rows = []
    for n, sid in enumerate(sample):
        cs = sorted(u.at[sid, "codes"])
        g, b = u.at[sid, "gender"], u.at[sid, "band"]
        lvl = {c: 0 for c in cs}
        # прогнати жадібний алгоритм рівно так, як у статті
        steps = 0
        while steps < MAX_STEPS:
            w, wset = worst_pool(g, b, cs, lvl, M)
            if w >= K_TARGET:
                break
            cand = [c for c in wset if lvl[c] < MAXLVL]
            if not cand:
                break
            best, gain = None, -1
            for c in cand:
                t = dict(lvl); t[c] += 1
                v = sup(g, b, [(x, t[x]) for x in wset])
                if v > gain:
                    best, gain = c, v
            lvl[best] += 1
            steps += 1
        w_pool, _ = worst_pool(g, b, cs, lvl, M)
        reached_pool = w_pool >= K_TARGET
        # НЕЗАЛЕЖНА перевірка того самого кінцевого стану
        w_rand, rset = worst_random(g, b, cs, lvl, M, N_RANDOM)
        rows.append(dict(sid=int(sid), n_codes=len(cs), steps=steps,
                         worst_pool=int(w_pool), worst_random=int(w_rand),
                         reached_pool=bool(reached_pool),
                         missed=bool(reached_pool and w_rand < K_TARGET)))
        if n % 10 == 0:
            print(f"  {n}/{len(sample)} codes={len(cs):3d} pool={w_pool:4d} rand={w_rand:4d}"
                  f"{'  <-- ПРОПУСК' if rows[-1]['missed'] else ''}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "verify_pool.csv", index=False)
    ok = out[out.reached_pool]
    missed = int(ok.missed.sum())
    n_ok = len(ok)
    print()
    print(f"юнітів перевірено: {len(out)}, медіана кодів {out.n_codes.median():.0f}, "
          f"максимум {out.n_codes.max()}")
    print(f"алгоритм оголосив reached: {n_ok}")
    print(f"з них випадковий пошук знайшов порушення: {missed} ({missed/max(n_ok,1):.1%})")
    if missed == 0 and n_ok:
        print(f"верхня межа 95 % (правило трьох): {3/n_ok:.1%}")
    print(f"середнє заниження найгіршого support: "
          f"{(out.worst_pool - out.worst_random).mean():.2f}")


if __name__ == "__main__":
    main()
