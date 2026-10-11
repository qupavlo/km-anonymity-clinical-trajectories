"""Вплив обрізання перебору, виміряний ПАРНО (закриває третю ітерацію тієї самої помилки).

Історія дефекту: раунд 1 — C7, раунд 2 — M2-2, і в тексті, написаному для закриття M2-2, та сама
помилка відтворена втретє. Твердження «звуження пулу з 14 до 12 піднімає видиму захищеність на
3–6 пунктів» було побудоване на зіставленні `kmanon.csv` (n=780/737, POOL=14) з `joint_paired.csv`
(n=483/464, POOL=12) — різні вибірки, різні прогони, непарне порівняння. Виправлення симптому
зберегло спосіб міркування.

Тут POOL=14 і POOL=12 рахуються для ОДНИХ І ТИХ САМИХ пацієнтів, решта параметрів спільна.
Звітуємо парну різницю, розбіжні пари і McNemar.

Вихід: results/unit/pool_paired.csv
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from kmanon import gen_code  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
K_TARGET, MAXLVL, MAX_STEPS = 5, 2, 40
M_VALUES = (2, 3)
POOLS = (14, 12)
N_SAMPLE = 500
SEED = 20261011


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
    print("готово", flush=True)

    def sup(g, b, items, exact):
        sets = [idx[l][(g, b, gen_code(c, l))] for c, l in items]
        if not sets:
            return len(u)
        sets.sort(key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not exact and len(out) < K_TARGET:
                return len(out)
        return len(out)

    def worst(g, b, cs, lvl, m, pool_size):
        sup1 = {c: sup(g, b, [(c, lvl[c])], False) for c in cs}
        w, wset = min(sup1.values()), [min(sup1, key=sup1.get)]
        pool = sorted(cs, key=lambda c: sup1[c])[:pool_size]
        for j in range(2, m + 1):
            for comb in combinations(pool, j):
                s = sup(g, b, [(c, lvl[c]) for c in comb], False)
                if s < w:
                    w, wset = s, list(comb)
                if w == 0:
                    break
        return w, wset

    def enforce(g, b, cs, m, pool_size):
        lvl = {c: 0 for c in cs}
        for _ in range(MAX_STEPS):
            w, wset = worst(g, b, cs, lvl, m, pool_size)
            if w >= K_TARGET:
                break
            cand = [c for c in wset if lvl[c] < MAXLVL]
            if not cand:
                break
            best, gain = None, -1
            for c in cand:
                t = dict(lvl); t[c] += 1
                v = sup(g, b, [(x, t[x]) for x in wset], True)
                if v > gain:
                    best, gain = c, v
            lvl[best] += 1
        w, _ = worst(g, b, cs, lvl, m, pool_size)
        return w >= K_TARGET, sum(lvl.values())

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for m in M_VALUES:
        res = defaultdict(list)
        for n, sid in enumerate(sample):
            if n % 50 == 0:
                print(f"  m={m}: {n}/{len(sample)}", flush=True)
            cs = sorted(u.at[sid, "codes"])
            g, b = u.at[sid, "gender"], u.at[sid, "band"]
            if len(cs) < m:
                continue
            for P in POOLS:
                ok, cost = enforce(g, b, cs, m, P)
                res[f"ok{P}"].append(ok); res[f"cost{P}"].append(cost)
            res["n_codes"].append(len(cs))
        a14 = np.array(res["ok14"], dtype=float)
        a12 = np.array(res["ok12"], dtype=float)
        d = a12 - a14
        b_ = int(((a12 == 1) & (a14 == 0)).sum())   # лише POOL=12 сертифікував
        c_ = int(((a12 == 0) & (a14 == 1)).sum())
        pv = stats.binomtest(b_, b_ + c_, 0.5).pvalue if (b_ + c_) else 1.0
        ci = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d)) if d.std() > 0 else (0.0, 0.0)
        rows.append(dict(m=m, n=len(a14), pool14=float(a14.mean()), pool12=float(a12.mean()),
                         paired_diff=float(d.mean()), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                         only_pool12=b_, only_pool14=c_, mcnemar_p=float(pv),
                         cost14=float(np.mean(res["cost14"])), cost12=float(np.mean(res["cost12"])),
                         median_codes=float(np.median(res["n_codes"]))))
        r = rows[-1]
        print(f"m={m}: POOL14 {r['pool14']:.3f}  POOL12 {r['pool12']:.3f}  "
              f"парна різниця {r['paired_diff']:+.3f} CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
              f"лише12/лише14 {b_}/{c_}  p={pv:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "pool_paired.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
