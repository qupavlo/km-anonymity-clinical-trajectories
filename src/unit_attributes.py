"""РЕАЛІСТИЧНИЙ сценарій: супротивник знає q НАЙЧАСТІШИХ діагнозів пацієнта.

Наскільки додаткові атрибути датасету звужують множину кандидатів.

Наші попередні числа пораховано ЛИШЕ за діагнозами, тож вони є нижньою межею ризику.
Реальний датасет для навчання містить більше: процедури, тип і час госпіталізації,
страховку, мову, сімейний стан. Кожен атрибут звужує коло.

Міряємо приріст ризику: від базового сценарію (стать, вік, q діагнозів) додаємо по одному
атрибуту і дивимося, як змінюється медіана множини кандидатів і частка ідентифікованих.

Вихід: results/unit/attributes.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RES.mkdir(parents=True, exist_ok=True)
Q_KNOWN = (2, 3)


def main():
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz", dtype={"icd_code": str})
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pr = pd.read_csv(ROOT / "raw" / "hosp" / "procedures_icd.csv.gz", dtype={"icd_code": str})
    pr["pcode"] = "P" + pr.icd_version.astype(str) + ":" + pr.icd_code.str.strip()
    adm = pd.read_csv(ROOT / "raw" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "hadm_id", "admittime", "admission_type",
                               "insurance", "language", "marital_status", "race"],
                      parse_dates=["admittime"])
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    pop = dx.groupby("code").subject_id.nunique()
    unit = dx.groupby("subject_id").code.apply(set)
    u = pd.DataFrame({"codes": unit}).join(pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]

    # атрибути на рівні пацієнта: беремо з ПЕРШОЇ госпіталізації (що знав би супротивник)
    adm = adm.sort_values(["subject_id", "admittime"])
    first = adm.groupby("subject_id").first()
    u = u.join(first[["admission_type", "insurance", "language", "marital_status", "race"]])
    u["year"] = first.admittime.dt.year.reindex(u.index)
    u["n_adm"] = adm.groupby("subject_id").hadm_id.nunique().reindex(u.index)
    u["procs"] = pr.groupby("subject_id").pcode.apply(set).reindex(u.index)
    u["n_proc"] = u.procs.map(lambda s: len(s) if isinstance(s, set) else 0)
    print(f"пацієнтів {len(u)}; з процедурами {(u.n_proc > 0).mean():.1%}", flush=True)

    def top_rare(cs, q):
        return tuple(sorted(sorted(cs, key=lambda c: -pop.get(c, 0))[:q]))

    extras = {
        "— (лише діагнози)": lambda r: (),
        "+ рік першої госпіталізації": lambda r: (r.year,),
        "+ тип госпіталізації": lambda r: (r.admission_type,),
        "+ кількість госпіталізацій": lambda r: (r.n_adm,),
        "+ страховка": lambda r: (r.insurance,),
        "+ сімейний стан": lambda r: (r.marital_status,),
        "+ раса": lambda r: (r.race,),
        "+ 1 процедура": lambda r: (min(r.procs, key=lambda c: pr_pop.get(c, 0))
                                    if isinstance(r.procs, set) and r.procs else None,),
        "+ рік і тип": lambda r: (r.year, r.admission_type),
        "+ усі демографічні": lambda r: (r.insurance, r.marital_status, r.race, r.language),
        "+ усе разом": lambda r: (r.year, r.admission_type, r.n_adm, r.insurance,
                                  r.marital_status, r.race),
    }
    pr_pop = pr.groupby("pcode").subject_id.nunique().to_dict()

    rows = []
    for q in Q_KNOWN:
        base_key = [top_rare(cs, q) for cs in u.codes]
        for name, fn in extras.items():
            keys = [(g, b, k) + tuple(fn(r)) for g, b, k, r in
                    zip(u.gender, u.band, base_key, u.itertuples())]
            cnt = defaultdict(int)
            for k in keys:
                cnt[k] += 1
            size = np.array([cnt[k] for k in keys])
            rows.append(dict(q_known=q, extra=name, median=float(np.median(size)),
                             unique=float(np.mean(size == 1)), lt5=float(np.mean(size < 5)),
                             ge20=float(np.mean(size >= 20))))
            print(f"q={q} {name:30s} медіана {np.median(size):7.0f}  "
                  f"ідентифіковано {np.mean(size == 1):6.1%}  <5 {np.mean(size < 5):6.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "attributes.csv", index=False)


if __name__ == "__main__":
    main()
