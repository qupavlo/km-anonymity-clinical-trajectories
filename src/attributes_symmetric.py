"""Attribute table with the filter fixed (the same defect as in linkage_fixed).

`linkage_fixed.py:132` calls `common_q(r.codes, 2)` without checking `len(codes) >= 2`. For a
patient with a single code this returns their ENTIRE code set disguised as "the two most frequent
diagnoses", and the full set identifies 87.9%. There are 5,888 such patients (2.6% of the cohort)
and they inflate the baseline against which each attribute's contribution is measured.

This script recomputes the same table on the cohort with >= 2 codes, so that the baseline and
every row refer to one and the same population.

Output: results/unit/attributes_symmetric.csv
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "unit"
Q = 2


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
    for col, src in (("year", first.admittime.dt.year), ("adm_type", first.admission_type),
                     ("insurance", first.insurance), ("marital", first.marital_status),
                     ("race", first.race)):
        u[col] = src.reindex(u.index)
    u["procs"] = pr.groupby("subject_id").code.apply(set).reindex(u.index)
    ppop = pr.groupby("code").subject_id.nunique().to_dict()
    pop = dx.groupby("code").subject_id.nunique().to_dict()

    N_all = len(u)
    u = u[u.codes.map(len) >= Q]          # <-- the fix: one shared population for every row
    N = len(u)
    print(f"cohort {N_all}, with >= {Q} codes {N} (excluded {N_all - N})", flush=True)

    demo = defaultdict(set)
    code_idx = defaultdict(set)
    proc_idx = defaultdict(set)
    for sid, cs, g, b in zip(u.index, u.codes, u.gender, u.band):
        demo[(g, b)].add(sid)
        for c in cs:
            code_idx[c].add(sid)
    for sid, ps in zip(u.index, u.procs):
        if isinstance(ps, set):
            for c in ps:
                proc_idx[c].add(sid)
    attr_idx = {a: defaultdict(set) for a in ("year", "adm_type", "insurance", "marital", "race")}
    for sid in u.index:
        for a in attr_idx:
            attr_idx[a][u.at[sid, a]].add(sid)
    print("indexes ready", flush=True)

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
    rows = []
    for name, fn in extra.items():
        sizes = [inter([demo[(r.gender, r.band)]]
                       + [code_idx[c] for c in common_q(r.codes, Q)] + fn(r))
                 for r in u.itertuples()]
        a = np.array(sizes)
        rows.append(dict(extra=name, n=len(a), median=float(np.median(a)),
                         unique=float(np.mean(a == 1)), lt5=float(np.mean(a < 5))))
        print(f"{name:22s} n={len(a):6d} median {np.median(a):7.0f} "
              f"ident. {np.mean(a == 1):6.2%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "attributes_symmetric.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
