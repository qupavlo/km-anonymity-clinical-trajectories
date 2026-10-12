"""Linkage risk for a patient UNIT under partial external knowledge.

SUPERSEDED, KNOWN-INCORRECT. Kept only because the paper reports what it got wrong; do NOT use
it as a reference implementation. It groups patients by exact equality of the key (sex, age band,
q known codes), so a patient counts as a candidate only when their own q codes coincide with the
known ones. The candidate set is defined by CONTAINMENT instead: a patient is a candidate if their
trajectory contains the known codes, whatever else it holds. Equality understates the candidate
set and therefore overstates risk. The corrected version is `linkage_fixed.py`.

A unit = the patient's whole trajectory: every diagnosis of every admission. Training needs the
unit released intact, so uniqueness of the set itself is unavoidable.

The question is not "is the unit unique" but "how many units match what the adversary actually
knows". External-knowledge scenarios, from weak to strong:

  S0  sex + age band
  S1  + the patient's most frequent diagnosis (what relatives know: "he was in with his heart")
  S2  + the 2 most frequent diagnoses
  S3  + the 3 most frequent diagnoses
  S4  S1 + number of admissions
  S5  S3 + number of admissions
  S6  one RARE diagnosis of the patient (worst case: the adversary knows about a rare disease)

For each scenario we count the candidate set size, i.e. how many patients in the data match that
knowledge. Size 1 means the unit is identified.

Output: results/unit/linkage.csv, summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
RES.mkdir(parents=True, exist_ok=True)
RARE_PAT = 10          # a diagnosis is rare if at most this many patients in the whole base have it


def main():
    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz")
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(int).astype(str) + "s"

    # population frequency of a diagnosis, over the WHOLE MIMIC base
    pop = dx.groupby("code").subject_id.nunique()
    rare = set(pop[pop <= RARE_PAT].index)

    unit = dx.groupby("subject_id").code.apply(set)
    n_adm = dx.groupby("subject_id").hadm_id.nunique()
    u = pd.DataFrame({"codes": unit, "n_adm": n_adm}).join(
        pts.set_index("subject_id")[["gender", "band"]], how="inner")
    u = u[u.gender.notna() & u.band.notna()]
    print(f"units (patients) {len(u)}; median diagnoses {u.codes.map(len).median():.0f}, "
          f"admissions {u.n_adm.median():.0f}", flush=True)

    # the most frequent and the rarest diagnoses inside the unit
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
        "S0 sex+age band": lambda r: (r.gender, r.band),
        "S1 +1 common diagnosis": lambda r: (r.gender, r.band, r.c1),
        "S2 +2 common diagnoses": lambda r: (r.gender, r.band, r.c2),
        "S3 +3 common diagnoses": lambda r: (r.gender, r.band, r.c3),
        "S4 S1+n_admissions": lambda r: (r.gender, r.band, r.c1, r.n_adm),
        "S5 S3+n_admissions": lambda r: (r.gender, r.band, r.c3, r.n_adm),
        "S6 one rare diagnosis": lambda r: (r.rarest,),
        "S7 full code set (reference)": lambda r: (r.gender, r.band, tuple(sorted(r.codes))),
    }
    rows = []
    for name, fn in scen.items():
        keys = [fn(r) for r in u.itertuples()]
        # pandas turns None into NaN, so test with pd.isna rather than "is not None"
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
        print(f"{name:30s} median candidates {np.median(size):8.0f}  "
              f"unique {np.mean(size==1):6.1%}  <5 {np.mean(size<5):6.1%}  "
              f">=100 {np.mean(size>=100):6.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "linkage.csv", index=False)
    (RES / "summary.json").write_text(json.dumps(dict(
        n_units=int(len(u)), rare_threshold=RARE_PAT,
        n_rare_codes=len(rare), rows=out.to_dict("records")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
