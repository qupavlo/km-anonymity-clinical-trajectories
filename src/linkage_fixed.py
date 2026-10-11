"""Виправлений розрахунок множини кандидатів: ВКЛЮЧЕННЯ, а не рівність ключа.

Помилка попередньої версії (`unit_linkage.py`): пацієнти групувалися за точним збігом кортежа
(стать, вікова група, q відомих кодів). Тобто кандидатом вважався лише той, чиї власні q
найчастіших кодів збігалися з відомими. Стаття ж означує множину кандидатів як
    C = { p' : A_p' = A_p AND Q_p ⊆ U_p' }
тобто пацієнт є кандидатом, якщо він МІСТИТЬ відомі коди, незалежно від решти своєї траєкторії.

Рівність занижує |C| і тому завищує ризик. Тут рахуємо коректно, через інвертований індекс.

Вихід: results/unit/linkage_fixed.csv, results/unit/attributes_fixed.csv
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
    u["n_adm"] = adm.groupby("subject_id").hadm_id.nunique().reindex(u.index)
    u["procs"] = pr.groupby("subject_id").code.apply(set).reindex(u.index)
    pop = dx.groupby("code").subject_id.nunique().to_dict()
    ppop = pr.groupby("code").subject_id.nunique().to_dict()
    N = len(u)
    print(f"пацієнтів {N}", flush=True)

    # інвертовані індекси
    demo = defaultdict(set)
    for sid, g, b in zip(u.index, u.gender, u.band):
        demo[(g, b)].add(sid)
    code_idx = defaultdict(set)
    for sid, cs in zip(u.index, u.codes):
        for c in cs:
            code_idx[c].add(sid)
    proc_idx = defaultdict(set)
    for sid, ps in zip(u.index, u.procs):
        if isinstance(ps, set):
            for c in ps:
                proc_idx[c].add(sid)
    attr_idx = {a: defaultdict(set) for a in ("year", "adm_type", "insurance", "marital", "race", "n_adm")}
    for sid in u.index:
        for a in attr_idx:
            attr_idx[a][u.at[sid, a]].add(sid)
    rare = {c for c, n in pop.items() if n <= RARE_PAT}
    print("індекси готові", flush=True)

    def inter(sets):
        sets = sorted(sets, key=len)
        out = sets[0]
        for s in sets[1:]:
            out = out & s
            if not out:
                break
        return len(out)

    def common_q(cs, q):
        return sorted(cs, key=lambda c: -pop.get(c, 0))[:q]

    scen = {
        "S0 sex+age": lambda sid, r: [],
        "S1 +1 common dx": lambda sid, r: [code_idx[c] for c in common_q(r.codes, 1)],
        "S2 +2 common dx": lambda sid, r: [code_idx[c] for c in common_q(r.codes, 2)],
        "S3 +3 common dx": lambda sid, r: [code_idx[c] for c in common_q(r.codes, 3)],
        "S4 S1+n_adm": lambda sid, r: [code_idx[c] for c in common_q(r.codes, 1)] + [attr_idx["n_adm"][r.n_adm]],
        "S5 S3+n_adm": lambda sid, r: [code_idx[c] for c in common_q(r.codes, 3)] + [attr_idx["n_adm"][r.n_adm]],
        "S6 one rare dx": lambda sid, r: [code_idx[min((c for c in r.codes if c in rare),
                                                       key=lambda c: pop.get(c, 0))]]
                          if any(c in rare for c in r.codes) else None,
        "S7 full code set": lambda sid, r: [code_idx[c] for c in r.codes],
    }
    rows = []
    for name, fn in scen.items():
        sizes = []
        for r in u.itertuples():
            parts = fn(r.Index, r)
            if parts is None:
                continue
            sizes.append(inter([demo[(r.gender, r.band)]] + parts))
        a = np.array(sizes)
        rows.append(dict(scenario=name, n=len(a), median=float(np.median(a)),
                         unique=float(np.mean(a == 1)), lt5=float(np.mean(a < 5)),
                         ge100=float(np.mean(a >= 100))))
        print(f"{name:20s} n={len(a):6d} медіана {np.median(a):8.0f} "
              f"ідентифіковано {np.mean(a == 1):6.1%}  <5 {np.mean(a < 5):6.1%}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "linkage_fixed.csv", index=False)

    # атрибути: до двох найчастіших діагнозів додаємо по одному атрибуту
    print("\nАТРИБУТИ (до 2 найчастіших діагнозів):", flush=True)
    extra = {
        "— (diagnoses only)": lambda r: [],
        "+ year of admission": lambda r: [attr_idx["year"][r.year]],
        "+ admission type": lambda r: [attr_idx["adm_type"][r.adm_type]],
        "+ insurance": lambda r: [attr_idx["insurance"][r.insurance]],
        "+ marital status": lambda r: [attr_idx["marital"][r.marital]],
        "+ race": lambda r: [attr_idx["race"][r.race]],
        "+ one procedure": lambda r: [proc_idx[min(r.procs, key=lambda c: ppop.get(c, 0))]]
                           if isinstance(r.procs, set) and r.procs else [],
        "+ all of the above": lambda r: [attr_idx[a][getattr(r, a)] for a in
                                         ("year", "adm_type", "insurance", "marital", "race")]
                              + ([proc_idx[min(r.procs, key=lambda c: ppop.get(c, 0))]]
                                 if isinstance(r.procs, set) and r.procs else []),
    }
    arows = []
    for name, fn in extra.items():
        sizes = []
        for r in u.itertuples():
            parts = [code_idx[c] for c in common_q(r.codes, 2)] + fn(r)
            sizes.append(inter([demo[(r.gender, r.band)]] + parts))
        a = np.array(sizes)
        arows.append(dict(extra=name, n=len(a), median=float(np.median(a)),
                          unique=float(np.mean(a == 1)), lt5=float(np.mean(a < 5))))
        print(f"  {name:22s} медіана {np.median(a):7.0f}  ідентифіковано {np.mean(a == 1):6.1%}", flush=True)
    pd.DataFrame(arows).to_csv(RES / "attributes_fixed.csv", index=False)


if __name__ == "__main__":
    main()
