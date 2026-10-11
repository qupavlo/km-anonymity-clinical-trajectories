"""Ризик зв'язування ЮНІТА пацієнта за часткового зовнішнього знання.

Юніт = уся траєкторія пацієнта: усі діагнози всіх його госпіталізацій.
Для навчання юніт треба випустити цілісним, тож унікальність самого набору неминуча.

Питання не «чи унікальний юніт», а «скільки юнітів відповідають тому, що супротивник реально знає».
Сценарії зовнішнього знання, від слабкого до сильного:

  S0  стать + вікова група
  S1  + найчастіший діагноз пацієнта (те, що знають близькі: «лежав із серцем»)
  S2  + 2 найчастіші діагнози
  S3  + 3 найчастіші діагнози
  S4  S1 + кількість госпіталізацій
  S5  S3 + кількість госпіталізацій
  S6  один РІДКІСНИЙ діагноз пацієнта (найгірший випадок: супротивник знає про рідкісну хворобу)

Для кожного сценарію рахуємо розмір множини кандидатів — скільки пацієнтів у наборі відповідають
цьому знанню. Розмір 1 означає, що юніт ідентифіковано.

Вихід: results/unit/linkage.csv, summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RES.mkdir(parents=True, exist_ok=True)
RARE_PAT = 10          # діагноз рідкісний, якщо його мають <= стількох пацієнтів у всій базі


def main():
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz")
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    # популяційна частота діагнозу — за ВСІЄЮ базою MIMIC
    pop = dx.groupby("code").subject_id.nunique()
    rare = set(pop[pop <= RARE_PAT].index)

    # юніт пацієнта
    unit = dx.groupby("subject_id").code.apply(set)
    n_adm = dx.groupby("subject_id").hadm_id.nunique()
    u = pd.DataFrame({"codes": unit, "n_adm": n_adm}).join(
        pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    print(f"юнітів (пацієнтів) {len(u)}; медіана діагнозів {u.codes.map(len).median():.0f}, "
          f"госпіталізацій {u.n_adm.median():.0f}", flush=True)

    # найчастіші й найрідкісніші діагнози всередині юніта
    def top_common(cs, k):
        return tuple(sorted(sorted(cs, key=lambda c: -pop.get(c, 0))[:k]))

    def rarest(cs):
        r = [c for c in cs if c in rare]
        return min(r, key=lambda c: pop.get(c, 0)) if r else None

    u["c1"] = u.codes.map(lambda s: top_common(s, 1))
    u["c2"] = u.codes.map(lambda s: top_common(s, 2))
    u["c3"] = u.codes.map(lambda s: top_common(s, 3))
    u["rarest"] = u.codes.map(rarest)

    scen = {
        "S0 стать+вік": lambda r: (r.gender, r.band),
        "S1 +1 частий діагноз": lambda r: (r.gender, r.band, r.c1),
        "S2 +2 часті діагнози": lambda r: (r.gender, r.band, r.c2),
        "S3 +3 часті діагнози": lambda r: (r.gender, r.band, r.c3),
        "S4 S1+к-ть госпіталізацій": lambda r: (r.gender, r.band, r.c1, r.n_adm),
        "S5 S3+к-ть госпіталізацій": lambda r: (r.gender, r.band, r.c3, r.n_adm),
        "S6 один рідкісний діагноз": lambda r: (r.rarest,),
        "S7 повний набір (довідково)": lambda r: (r.gender, r.band, tuple(sorted(r.codes))),
    }
    rows = []
    for name, fn in scen.items():
        keys = [fn(r) for r in u.itertuples()]
        # pandas перетворює None на NaN, тому перевіряємо через pd.isna, а не "is not None"
        has = (np.array([not pd.isna(k[-1]) for k in keys]) if name.startswith("S6")
               else np.ones(len(u), bool))
        cnt = defaultdict(int)
        for k, ok in zip(keys, has):
            if ok:
                cnt[k] += 1
        size = np.array([cnt[k] for k, ok in zip(keys, has) if ok])
        rows.append(dict(
            scenario=name, n_units=int(has.sum()),
            median=float(np.median(size)), p10=float(np.percentile(size, 10)),
            unique=float(np.mean(size == 1)), lt5=float(np.mean(size < 5)),
            lt20=float(np.mean(size < 20)), ge100=float(np.mean(size >= 100))))
        print(f"{name:30s} медіана кандидатів {np.median(size):8.0f}  "
              f"унікальних {np.mean(size==1):6.1%}  <5 {np.mean(size<5):6.1%}  "
              f">=100 {np.mean(size>=100):6.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "linkage.csv", index=False)
    (RES / "summary.json").write_text(json.dumps(dict(
        n_units=int(len(u)), rare_threshold=RARE_PAT,
        n_rare_codes=len(rare), rows=out.to_dict("records")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
