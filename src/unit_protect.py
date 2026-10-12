"""Can a patient's unit be made safe by generalizing a few codes?

SUPERSEDED, KNOWN-INCORRECT. Kept only because the paper reports what it got wrong; do NOT use
it as a reference implementation. It fixes ONE assumed knowledge set per patient (the q codes
rarest in the population) and enforces the guarantee for that single set only. k^m-anonymity
requires the guarantee to hold for EVERY subset of at most m codes, so a different q-subset can
still leave the candidate set below K after this script has reported success; an adaptive
adversary breaks it. The correct, all-subsets version is `kmanon.py`.

A unit is released intact (the whole trajectory is needed for training). Generalization = lifting
an ICD code to its category: ICD-10 'E1165' -> 'E11' -> 'E'; ICD-9 '25000' -> '250' -> '25'.

Threat: the adversary knows sex, age band and q of the patient's diagnoses. We take the case worst
for the patient, i.e. the combination of q codes that narrows the candidate set the most (the codes
rarest in the population).

Greedy algorithm: while the candidate set is < K, generalize the code that yields the largest
increase of that set. Cost is measured as how many codes were changed and by how many levels.

Output: results/unit/protect.csv
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
Q_KNOWN = (2, 3)        # how many diagnoses the adversary knows
N_SAMPLE = 4000         # patients per run (the greedy search is expensive)
SEED = 20261005


def generalize(code, level):
    """Level 0 — unchanged; 1 — category; 2 — block."""
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

    # population index: for each generalization level, the set of patients per code
    print("building population index...", flush=True)
    idx = {}
    for lvl in (0, 1, 2):
        m = defaultdict(set)
        for sid, cs in zip(u.index, u.codes):
            g = u.at[sid, "gender"]; b = u.at[sid, "band"]
            for c in cs:
                m[(g, b, generalize(c, lvl))].add(sid)
        idx[lvl] = m
        print(f"  level {lvl}: {len(m)} keys", flush=True)

    def candidates(gender, band, chosen):
        """How many patients carry ALL the given (code, level) pairs at the same sex and age band."""
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
                # combination worst for the patient: the q codes rarest in the population
                known = sorted(cs, key=lambda c: pop.get(c, 0))[:q]
                lvl = {c: 0 for c in known}
                k0 = candidates(g, b, [(c, 0) for c in known])
                before.append(k0)
                for _ in range(2 * q):                      # at most 2 levels per code
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
            print(f"knows {q} diagnoses, target k>={K}: unsafe before protection {r['share_unsafe_before']:.1%}, "
                  f"reached {r['share_reached']:.1%}, codes changed on average {r['mean_codes_changed']:.2f} "
                  f"of {r['median_codes_total']:.0f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "protect.csv", index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
