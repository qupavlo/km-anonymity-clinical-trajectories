"""R1 and R4 on MIMIC-IV.

The reference support is taken from the ICD codes of the whole MIMIC database (223k patients),
not only from our subsample — so that rarity means rarity in the population, not in the sample.

R1  Does geometry find the chunks of admissions carrying a RARE ICD code
    (a code held by <= RARE_PAT patients in the whole database)?

R4  The gap between the structured and the textual representation.
    k_struct — how many patients share the same (sex, age band, set of admission codes);
    this is what classical disclosure control sees. The question: are there chunks that are safe
    by k_struct yet flagged as isolated by geometry, and is that a real risk.

Output: results/r1_r4/*.csv, summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "r1_r4"
RES.mkdir(parents=True, exist_ok=True)
RARE_PAT = 10        # a code is rare if <= 10 patients out of 223k have it
K_STRUCT = 5
DIAG_SECTIONS = ["Discharge Diagnosis", "Past Medical History", "History of Present Illness",
                 "Brief Hospital Course", "Chief Complaint", "Major Surgical or Invasive Procedure"]
RNG = np.random.default_rng(20261005)


def auc_ci(y, s, n=300):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.sum() < 5 or y.sum() == len(y):
        return np.nan, np.nan, np.nan
    a = roc_auc_score(y, s)
    pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
    bs = [roc_auc_score(y[i], s[i]) for i in
          (np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, min(len(neg), 20000))])
           for _ in range(n))]
    return a, *np.percentile(bs, [2.5, 97.5])


def main(model="bge"):
    ch = pd.read_parquet(ROOT / "data" / "chunks.parquet",
                         columns=["chunk_id", "subject_id", "hadm_id", "section"])
    m = pd.read_parquet(ROOT / "results" / f"metrics_{model}.parquet")
    d = ch.merge(m, on="chunk_id")

    dx = pd.read_csv(ROOT / "raw" / "hosp" / "diagnoses_icd.csv.gz")
    dx["code"] = dx.icd_version.astype(str) + ":" + dx.icd_code.str.strip()
    sup = dx.groupby("code").subject_id.nunique()                 # support across the whole database
    rare_codes = set(sup[sup <= RARE_PAT].index)
    print(f"codes in total {len(sup)}, rare (<= {RARE_PAT} patients) {len(rare_codes)}", flush=True)

    # rarity and the code set at the admission level
    dx_sub = dx[dx.subject_id.isin(set(ch.subject_id))]
    by_adm = dx_sub.groupby("hadm_id").code.apply(frozenset)
    adm_rare = by_adm.map(lambda s: bool(s & rare_codes))
    d["rare"] = d.hadm_id.map(adm_rare).fillna(False).astype(bool).values
    d["min_support"] = d.hadm_id.map(by_adm.map(lambda s: min((sup[c] for c in s), default=np.nan))).astype(float)

    # structural quasi-identifier
    pts = pd.read_csv(ROOT / "raw" / "hosp" / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    pts["band"] = (pts.anchor_age // 10 * 10).astype(str) + "s"
    pinfo = pts.set_index("subject_id")[["gender", "band"]]
    adm_df = pd.DataFrame({"hadm_id": by_adm.index, "codes": by_adm.values})
    adm_df["subject_id"] = dx_sub.groupby("hadm_id").subject_id.first().values
    adm_df = adm_df.join(pinfo, on="subject_id")
    qid = list(zip(adm_df.gender, adm_df.band, adm_df.codes))
    codes_q, _ = pd.factorize(pd.Series(qid, dtype=object))
    pat_by_q = defaultdict(set)
    for q, s in zip(codes_q, adm_df.subject_id):
        pat_by_q[q].add(s)
    adm_df["k_struct"] = [len(pat_by_q[q]) for q in codes_q]
    d["k_struct"] = d.hadm_id.map(adm_df.set_index("hadm_id").k_struct)

    feats = {
        "r_k other-patient (k=20)": d.rk_foreign_k20.values,
        "r_k all (k=20)": d.rk_all_k20.values,
        "anti-hubness (k=20)": -d.kocc_all_k20.values.astype(float),
        "d to nearest other-patient neighbour": d.d_foreign1.values,
        "LID (k=20)": d.lid_all_k20.values,
        "λ = 1/(LID+1)": d.lam_k20.values,
    }
    rk = pd.Series(d.rk_foreign_k20).rank(pct=True).values
    ah = 1 - pd.Series(d.kocc_all_k20).rank(pct=True).values
    feats["U = r_k + anti-hubness"] = (rk + ah) / 2

    rows = []
    for scope, mask in [("all sections", np.ones(len(d), bool)),
                        ("diagnostic sections", d.section.isin(DIAG_SECTIONS).values)]:
        rare = d.rare.values.astype(bool)
        y = rare & mask
        sel = mask.astype(bool)
        for name, v in feats.items():
            a, lo, hi = auc_ci(y[sel], np.asarray(v, dtype=float)[sel])
            rows.append(dict(task="R1 rare code", scope=scope, feature=name,
                             auroc=a, lo=lo, hi=hi, n_pos=int(y.sum())))
    r1 = pd.DataFrame(rows)
    r1.to_csv(RES / f"r1_{model}.csv", index=False)

    # R4: the gap
    ks = pd.to_numeric(d.k_struct, errors="coerce").values
    safe = (ks >= K_STRUCT) & np.isfinite(ks)
    iso = d.rk_foreign_k20.values >= np.nanquantile(d.rk_foreign_k20.values, 0.90)
    gap = safe & iso
    r4 = dict(
        n_chunks=int(len(d)),
        safe_by_struct=float(safe.mean()),
        k_struct_lt5=float(np.nanmean(ks < K_STRUCT)),
        isolated_top10pct=float(iso.mean()),
        gap_safe_but_isolated=float(gap.mean()),
        gap_n=int(gap.sum()),
        rare_among_gap=float(d.rare.values.astype(bool)[gap].mean()),
        rare_among_safe=float(d.rare.values.astype(bool)[safe].mean()),
    )
    (RES / f"r4_{model}.json").write_text(json.dumps(r4, ensure_ascii=False, indent=2))

    print("\nR1 - AUROC for 'admission carries a rare ICD code':")
    for scope, g in r1.groupby("scope"):
        print(f"  [{scope}] positives {g.n_pos.iat[0]}")
        for _, r in g.sort_values("auroc", ascending=False).iterrows():
            print(f"    {r.feature:28s} {r.auroc:.3f}  [{r.lo:.3f}; {r.hi:.3f}]")
    print("\nR4 - gap:")
    for k, v in r4.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    import sys
    for mm in sys.argv[1:] or ["bge"]:
        main(mm)
