"""Чи можна зробити юніт пацієнта безпечним, узагальнивши кілька кодів?

Юніт випускається цілісним (уся траєкторія потрібна для навчання). Узагальнення = підняття коду
ICD до категорії: ICD-10 'E1165' -> 'E11' -> 'E'; ICD-9 '25000' -> '250' -> '25'.

Загроза: супротивник знає стать, вікову групу і q діагнозів пацієнта. Беремо найгірший для
пацієнта випадок — ту комбінацію q кодів, що звужує множину кандидатів найсильніше
(найрідкісніші в популяції коди).

Жадібний алгоритм: доки множина кандидатів < K, узагальнюємо той код, який дає найбільший приріст
множини. Міряємо ціну: скільки кодів змінено і на скільки рівнів.

Вихід: results/unit/protect.csv
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RES.mkdir(parents=True, exist_ok=True)
K_TARGET = (5, 20)
Q_KNOWN = (2, 3)        # скільки діагнозів знає супротивник
N_SAMPLE = 4000         # пацієнтів у прогоні (жадібний пошук дорогий)
SEED = 20261005


def generalize(code, level):
    """level 0 — без змін; 1 — категорія; 2 — блок."""
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
    pinfo = pts.set_index("subject_id")[["gender", "band"]]

    unit = dx.groupby("subject_id").code.apply(set)
    u = pd.DataFrame({"codes": unit}).join(pinfo, how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    pop = dx.groupby("code").subject_id.nunique()

    # індекс популяції: для кожного рівня узагальнення — множина пацієнтів на код
    print("будую індекс популяції...", flush=True)
    idx = {}
    for lvl in (0, 1, 2):
        m = defaultdict(set)
        for sid, cs in zip(u.index, u.codes):
            g = u.at[sid, "gender"]; b = u.at[sid, "band"]
            for c in cs:
                m[(g, b, generalize(c, lvl))].add(sid)
        idx[lvl] = m
        print(f"  рівень {lvl}: {len(m)} ключів", flush=True)

    def candidates(gender, band, chosen):
        """Скільки пацієнтів мають УСІ задані (код, рівень) при тих самих статі й віку."""
        sets = [idx[lvl].get((gender, band, generalize(c, lvl)), set()) for c, lvl in chosen]
        if not sets:
            return len(u)
        out = set.intersection(*sets)
        return len(out)

    sample = rng.choice(u.index.values, min(N_SAMPLE, len(u)), replace=False)
    rows = []
    for q in Q_KNOWN:
        for K in K_TARGET:
            n_ok = 0; changed = []; depth = []; before = []
            for sid in sample:
                cs = list(u.at[sid, "codes"])
                g, b = u.at[sid, "gender"], u.at[sid, "band"]
                # найгірша для пацієнта комбінація: q найрідкісніших у популяції кодів
                known = sorted(cs, key=lambda c: pop.get(c, 0))[:q]
                lvl = {c: 0 for c in known}
                k0 = candidates(g, b, [(c, 0) for c in known])
                before.append(k0)
                for _ in range(2 * q):                      # не більше ніж 2 рівні на код
                    cur = candidates(g, b, [(c, lvl[c]) for c in known])
                    if cur >= K:
                        break
                    best, gain = None, -1
                    for c in known:
                        if lvl[c] >= 2:
                            continue
                        trial = [(x, lvl[x] + 1 if x == c else lvl[x]) for x in known]
                        val = candidates(g, b, trial)
                        if val > gain:
                            best, gain = c, val
                    if best is None:
                        break
                    lvl[best] += 1
                fin = candidates(g, b, [(c, lvl[c]) for c in known])
                if fin >= K:
                    n_ok += 1
                changed.append(sum(1 for c in known if lvl[c] > 0))
                depth.append(sum(lvl.values()))
            rows.append(dict(
                q_known=q, k_target=K, n=len(sample),
                median_before=float(np.median(before)),
                share_unsafe_before=float(np.mean(np.array(before) < K)),
                share_reached=n_ok / len(sample),
                median_codes_changed=float(np.median(changed)),
                mean_codes_changed=float(np.mean(changed)),
                mean_depth=float(np.mean(depth)),
                median_codes_total=float(np.median([len(u.at[s, "codes"]) for s in sample])),
            ))
            r = rows[-1]
            print(f"знає {q} діагнози, ціль k>={K}: до захисту небезпечних {r['share_unsafe_before']:.1%}, "
                  f"досягнуто {r['share_reached']:.1%}, змінено кодів у середньому {r['mean_codes_changed']:.2f} "
                  f"з {r['median_codes_total']:.0f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "protect.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
