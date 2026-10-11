"""Базові лінії для цільового узагальнення юніта.

Перевіряємо, чи дає жадібний вибір виграш проти простіших стратегій за ОДНАКОВОЇ ціни.
Ціна = кількість піднятих рівнів (сумарна глибина), тож порівняння чесне.

  greedy      наш метод: піднімаємо код з найбільшим приростом множини кандидатів
  random      піднімаємо випадкові коди з відомих супротивнику, та сама глибина
  rarest      піднімаємо найрідкісніший код (проста евристика без перебору)
  all_rare    узагальнюємо ВСІ рідкісні коди юніта на 1 рівень (класичний підхід)
  suppress    вилучаємо з відомих супротивнику коди замість узагальнення

Вихід: results/unit/baselines.csv
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RES.mkdir(parents=True, exist_ok=True)
Q_KNOWN = (2, 3)
K_TARGET = (5, 20)
N_SAMPLE = 3000
N_BOOT = 200
RARE_PAT = 10
SEED = 20261006
MAXLVL = 2


def generalize(code, level):
    ver, c = code.split(":", 1)
    if level <= 0:
        return code
    if ver == "10":
        return f"{ver}:{c[:3] if level == 1 else c[:1]}"
    return f"{ver}:{c[:3] if level == 1 else c[:2]}"


def main():
    rng = np.random.default_rng(SEED)
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz")
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"
    unit = dx.groupby("subject_id").code.apply(set)
    u = pd.DataFrame({"codes": unit}).join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    pop = dx.groupby("code").subject_id.nunique()
    rare = set(pop[pop <= RARE_PAT].index)

    print("індекс популяції...", flush=True)
    idx = {}
    for lvl in range(MAXLVL + 1):
        m = defaultdict(set)
        for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
            for c in cs:
                m[(g, b, generalize(c, lvl))].add(sid)
        idx[lvl] = m

    def cand(g, b, chosen):
        """chosen: список (код, рівень); рівень None = код придушено (супротивник його не бачить)."""
        sets = [idx[l].get((g, b, generalize(c, l)), set()) for c, l in chosen if l is not None]
        return len(set.intersection(*sets)) if sets else len(u)

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)

    def run_greedy(g, b, known, K):
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

    def apply_budget(g, b, known, budget, order):
        """Підняти коди в заданому порядку, доки не витрачено budget рівнів."""
        lvl = {c: 0 for c in known}
        spent = 0
        i = 0
        while spent < budget and i < MAXLVL * len(known):
            c = order[i % len(order)]
            if lvl[c] < MAXLVL:
                lvl[c] += 1
                spent += 1
            i += 1
        return lvl

    rows = []
    for q in Q_KNOWN:
        for K in K_TARGET:
            res = defaultdict(list)
            for sid in sample:
                cs = list(u.at[sid, "codes"])
                g, b = u.at[sid, "gender"], u.at[sid, "band"]
                known = sorted(cs, key=lambda c: pop.get(c, 0))[:q]

                lvl_g = run_greedy(g, b, known, K)
                budget = sum(lvl_g.values())
                res["greedy"].append(cand(g, b, [(c, lvl_g[c]) for c in known]) >= K)
                res["greedy_cost"].append(budget)

                # random: та сама глибина, випадковий порядок
                order = list(rng.permutation(known))
                lvl_r = apply_budget(g, b, known, budget, order)
                res["random"].append(cand(g, b, [(c, lvl_r[c]) for c in known]) >= K)

                # rarest-first: та сама глибина, починаємо з найрідкіснішого
                order = sorted(known, key=lambda c: pop.get(c, 0))
                lvl_x = apply_budget(g, b, known, budget, order)
                res["rarest"].append(cand(g, b, [(c, lvl_x[c]) for c in known]) >= K)

                # all_rare: усі рідкісні коди юніта на 1 рівень (ціна рахується окремо)
                lvl_a = {c: (1 if c in rare else 0) for c in known}
                res["all_rare"].append(cand(g, b, [(c, lvl_a[c]) for c in known]) >= K)
                res["all_rare_cost"].append(sum(1 for c in cs if c in rare))

                # suppress: придушуємо стільки відомих кодів, скільки рівнів витратив greedy
                n_sup = min(len(known), max(1, budget // MAXLVL))
                sup = sorted(known, key=lambda c: pop.get(c, 0))[:n_sup]
                chosen = [(c, None if c in sup else 0) for c in known]
                res["suppress"].append(cand(g, b, chosen) >= K)
                res["suppress_cost"].append(n_sup)

            row = dict(q_known=q, k_target=K, n=len(sample))
            for m in ("greedy", "random", "rarest", "all_rare", "suppress"):
                a = np.array(res[m], dtype=float)
                boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(N_BOOT)]
                row[m] = a.mean()
                row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(boot, [2.5, 97.5])
            row["greedy_cost"] = float(np.mean(res["greedy_cost"]))
            row["all_rare_cost"] = float(np.mean(res["all_rare_cost"]))
            row["suppress_cost"] = float(np.mean(res["suppress_cost"]))
            rows.append(row)
            print(f"q={q} K={K}: greedy {row['greedy']:.3f}  random {row['random']:.3f}  "
                  f"rarest {row['rarest']:.3f}  all_rare {row['all_rare']:.3f} "
                  f"(ціна {row['all_rare_cost']:.1f} проти {row['greedy_cost']:.2f})  "
                  f"suppress {row['suppress']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "baselines.csv", index=False)
    print()
    cols = ["q_known", "k_target", "greedy", "random", "rarest", "all_rare", "suppress"]
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
