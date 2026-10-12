# What enforcing *k^m*-anonymity costs on longitudinal patient records

Code and aggregate results for the paper *"Refusal, Not Coarsening: What Enforcing
k^m-Anonymity Costs on Longitudinal Patient Records, and Why Its Guarantee Resists
Verification."*

## What this repository contains

- `src/` — every script that produced a number in the paper.
- `results/` — the aggregate result files the paper's tables and figure are drawn from.
- `fig/make_risk_curve.py` — the figure script. It reads from `results/`, so regenerating the
  figure does not require the source data.

## What this repository does not contain, and will not

**No patient data, and no derived data at the patient level.**

The study uses MIMIC-IV v3.1 and MIMIC-IV-Note v2.2, distributed by PhysioNet under a credentialed
data use agreement that prohibits redistribution. Beyond the raw data, four intermediate files
produced by these scripts carry one row per patient and are therefore also withheld:

| File | Why withheld |
|---|---|
| `who_fails.csv` | subject id with age, sex, ethnicity, payer |
| `refusal_reason.csv` | subject id with refusal cause and trajectory length |
| `verify_pool.csv`, `verify_pool_large.csv` | subject id with per-unit audit outcome |
| `diag_completeness.csv` | subject id with per-unit search results |
| `levels_m*.pkl` | cached per-patient generalization levels |

Everything in `results/` is aggregate: one row per experimental condition, never per person.
`verify_pool_summary.csv` is the aggregate form of the withheld per-unit audit file, and is the
source of the certification-gap figure reported in the paper.
`.gitignore` blocks the raw data, the caches and those four files, so a careless `git add -A`
cannot leak them.

## Getting the data

1. Complete CITI "Data or Specimens Only Research" training.
2. Sign the data use agreement for [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/)
   and, for the record-level analysis only, [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/).
3. Place the downloaded files so that `raw/hosp/` contains `diagnoses_icd.csv.gz`,
   `procedures_icd.csv.gz`, `admissions.csv.gz` and `patients.csv.gz`.

The agreement requires local processing. Do not send these files, or anything derived from them at
the record level, to a hosted service of any kind.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONHASHSEED=0          # several scripts break ties through set iteration
```

| Result in the paper | Script | Runtime |
|---|---|---|
| Table 1, risk by adversary knowledge | `src/linkage_symmetric.py` | ~25 min |
| Supplementary S1, single attributes | `src/attributes_symmetric.py` | ~20 min |
| Enforcement cost, reached shares | `src/kmanon.py` | ~15 min |
| Table 2, fixed vs adaptive adversary | `src/adaptive_adversary.py` | ~10 min |
| Certification gap (17.1%) | `src/verify_pool.py` (120 units) or `src/verify_pool_large.py` (350) | ~20 / ~60 min |
| Supplementary S2, selection rule and K=20 | `src/kmanon_k20.py` | ~40 min |
| Supplementary S3, pool size, paired | `src/pool_paired.py` | ~35 min |
| Supplementary S4, who is refused | `src/who_fails.py` + `src/race_confound.py` | ~15 min |
| Table 3, utility with fixed test set | `src/utility_refusal.py`, then `src/utility_trainset.py` | ~25 min |
| Figure 1 | `fig/make_risk_curve.py` | seconds |

`utility_trainset.py` reuses the generalization levels cached by `utility_refusal.py`, so run that
one first. Times are for a single workstation without a GPU.

## Reproducibility caveats

- Several scripts break ties by iterating Python sets, so `PYTHONHASHSEED` must be fixed for
  bit-identical output. Without it, results move in the last reported digit, not in any conclusion.
- `scikit-learn` changed `LogisticRegressionCV` behaviour across versions; the pinned version in
  `requirements.txt` is the one used for the paper.
- The subset search is capped at 14 elements by default (`POOL`). The paper's central methodological
  finding is that this cap is not a sound audit of the guarantee — see `src/verify_pool.py`, which
  measures how often it certifies a unit that an unrestricted search shows to be violating.

## A note on what the code does wrong, deliberately

`src/unit_linkage.py` and `src/unit_protect.py` implement earlier, **incorrect** versions, kept
because the paper reports what they got wrong:

- `unit_linkage.py` grouped candidates by exact key equality instead of containment, which
  overstated risk by roughly an order of magnitude. `linkage_fixed.py` and `linkage_symmetric.py`
  are the corrected versions.
- `unit_protect.py` enforces the condition for one assumed knowledge set rather than every subset.
  `adaptive_adversary.py` measures what that costs against an adversary who chooses adaptively.

Do not use either as a reference implementation.

## Licence

Code is released under the MIT Licence (`LICENSE`). The paper's aggregate results in `results/` are
released under CC BY 4.0.

## Citation

This release is archived on Zenodo:

> Zamroz, P. (2026). *Code and aggregate results for "Refusal, Not Coarsening: What Enforcing
> k^m-Anonymity Costs on Longitudinal Patient Records"* (v1.0.1) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.23318662

The concept identifier `10.5281/zenodo.23318661` always resolves to the latest version.
Citation details for the accompanying paper will be added once it has a DOI.
