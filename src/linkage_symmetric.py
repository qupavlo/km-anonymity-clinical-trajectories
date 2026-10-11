"""Симетричний переобрахунок таблиці ризику: розділити РІДКІСНІСТЬ і КІЛЬКІСТЬ (C2-4).

Претензія рецензії. У `linkage_fixed.py` сценарії S1-S3 беруть НАЙЧАСТІШІ коди пацієнта
(`common_q`), а S6 — НАЙРІДКІСНІШИЙ. Це найкращий випадок для захисника проти найгіршого для
супротивника, тож контраст «1 рідкісний б'є 3 звичайні» може бути артефактом вибору, а не
властивістю даних. До того ж S6 рахувався лише на пацієнтах, що мають рідкісний код
(n=34 216), а підпис таблиці заявляв n=223 291 — отже частки непорівнянні.

Тут робимо повний факторний дизайн: політика вибору (common | rare) × q (1,2,3).
Тоді:
  rare_q проти common_q за ОДНАКОВОГО q      -> чистий ефект РІДКІСНОСТІ
  q=1 проти q=2 проти q=3 за ОДНАКОВОЇ політики -> чистий ефект КІЛЬКОСТІ
Жодного порівняння «найкраще проти найгіршого».

Додатково кожен рядок рахується на ДВОХ знаменниках:
  n_cond  пацієнти, для яких сценарій визначений (як було)
  share_cohort  та сама частка, віднесена до ПОВНОЇ когорти -> порівнянна між рядками

Вихід: results/unit/linkage_symmetric.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RARE_PAT = 10


def main():
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    u = pd.DataFrame({"codes": dx.groupby("subject_id").code.apply(set)})
    u = u.join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    u["n_adm"] = adm.groupby("subject_id").hadm_id.nunique().reindex(u.index)
    pop = dx.groupby("code").subject_id.nunique().to_dict()
    N = len(u)
    print(f"пацієнтів {N}", flush=True)

    demo = defaultdict(set)
    for sid, g, b in zip(u.index, u.gender, u.band):
        demo[(g, b)].add(sid)
    code_idx = defaultdict(set)
    for sid, cs in zip(u.index, u.codes):
        for c in cs:
            code_idx[c].add(sid)
    nadm_idx = defaultdict(set)
    for sid, k in zip(u.index, u.n_adm):
        nadm_idx[k].add(sid)
    print("індекси готові", flush=True)

    def inter(sets):
        sets = sorted(sets, key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not out:
                break
        return len(out)

    def pick(cs, q, policy):
        """q кодів пацієнта: найчастіші або найрідкісніші. None, якщо кодів менше q."""
        if len(cs) < q:
            return None
        key = (lambda c: -pop.get(c, 0)) if policy == "common" else (lambda c: pop.get(c, 0))
        return sorted(cs, key=key)[:q]

    rows = []

    # базова лінія без діагнозів
    sizes = np.array([len(demo[(r.gender, r.band)]) for r in u.itertuples()])
    rows.append(dict(policy="-", q=0, n_cond=len(sizes), median=float(np.median(sizes)),
                     unique=float(np.mean(sizes == 1)),
                     share_cohort=float(np.mean(sizes == 1)),
                     lt5=float(np.mean(sizes < 5))))
    print(f"{'sex+age':22s} n={len(sizes):6d} медіана {np.median(sizes):7.0f} "
          f"ідент. {np.mean(sizes == 1):6.2%}", flush=True)

    # повний факторний: політика × q
    for policy in ("common", "rare"):
        for q in (1, 2, 3):
            sizes, n_undef = [], 0
            for r in u.itertuples():
                sel = pick(r.codes, q, policy)
                if sel is None:
                    n_undef += 1
                    continue
                sizes.append(inter([demo[(r.gender, r.band)]] + [code_idx[c] for c in sel]))
            a = np.array(sizes)
            uniq = float(np.mean(a == 1))
            rows.append(dict(policy=policy, q=q, n_cond=len(a), median=float(np.median(a)),
                             unique=uniq,
                             share_cohort=uniq * len(a) / N,   # віднесено до повної когорти
                             lt5=float(np.mean(a < 5))))
            print(f"{policy+' x'+str(q):22s} n={len(a):6d} медіана {np.median(a):7.0f} "
                  f"ідент. {uniq:6.2%}  (від когорти {uniq*len(a)/N:6.2%})", flush=True)

    # + кількість госпіталізацій, поверх q кодів (той самий фільтр, що й вище)
    for policy in ("common", "rare"):
        for q in (3,):
            sizes = []
            for r in u.itertuples():
                sel = pick(r.codes, q, policy)
                if sel is None:
                    continue
                sizes.append(inter([demo[(r.gender, r.band)], nadm_idx[r.n_adm]]
                                   + [code_idx[c] for c in sel]))
            a = np.array(sizes)
            uniq = float(np.mean(a == 1))
            rows.append(dict(policy=policy + "+nadm", q=q, n_cond=len(a),
                             median=float(np.median(a)), unique=uniq,
                             share_cohort=uniq * len(a) / N, lt5=float(np.mean(a < 5))))
            print(f"{policy+'x'+str(q)+' +n_adm':22s} n={len(a):6d} медіана {np.median(a):7.0f} "
                  f"ідент. {uniq:6.2%}", flush=True)

    # окремо: рівно один БЕЗУМОВНО рідкісний код (pop <= RARE_PAT) — як було в S6
    rare = {c for c, n in pop.items() if n <= RARE_PAT}
    sizes = []
    for r in u.itertuples():
        rr = [c for c in r.codes if c in rare]
        if not rr:
            continue
        sizes.append(inter([demo[(r.gender, r.band)],
                            code_idx[min(rr, key=lambda c: pop.get(c, 0))]]))
    a = np.array(sizes)
    uniq = float(np.mean(a == 1))
    rows.append(dict(policy="rare_abs", q=1, n_cond=len(a), median=float(np.median(a)),
                     unique=uniq, share_cohort=uniq * len(a) / N,
                     lt5=float(np.mean(a < 5))))
    print(f"{'one code pop<=10':22s} n={len(a):6d} медіана {np.median(a):7.0f} "
          f"ідент. {uniq:6.2%}  (від когорти {uniq*len(a)/N:6.2%})", flush=True)
    print(f"   частка когорти, що взагалі має такий код: {len(a)/N:.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "linkage_symmetric.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))

    # прямі відповіді на питання статті
    g = {(r.policy, r.q): r for r in out.itertuples()}
    print("\n--- ЕФЕКТ РІДКІСНОСТІ (той самий q, інша політика) ---")
    for q in (1, 2, 3):
        c, r_ = g[("common", q)], g[("rare", q)]
        print(f"  q={q}: common {c.unique:6.2%} -> rare {r_.unique:6.2%} "
              f"(×{r_.unique/max(c.unique,1e-9):.1f})")
    print("--- ЕФЕКТ КІЛЬКОСТІ (та сама політика, інше q) ---")
    for policy in ("common", "rare"):
        v = [g[(policy, q)].unique for q in (1, 2, 3)]
        print(f"  {policy}: {v[0]:6.2%} -> {v[1]:6.2%} -> {v[2]:6.2%} "
              f"(×{v[2]/max(v[0],1e-9):.1f} від q=1 до q=3)")


if __name__ == "__main__":
    main()
