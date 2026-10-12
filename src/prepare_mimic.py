"""Preparation of chunks from MIMIC-IV-Note discharge summaries.

One chunk = one clinically meaningful section of a discharge summary. Administrative and
boilerplate sections (discharge instructions, follow-up visits, facility) are dropped: they are
identical across thousands of patients and would dominate the embedding space.

The patient subsample is set by N_PATIENTS, so that the volume is comparable with the Synthea run
(~100k chunks) and exact neighbour search stays feasible on a single machine.

The data does not leave this machine (PhysioNet DUA terms).

Output: data/chunks.parquet
"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
SEED = 20261005
N_PATIENTS = 20000
MIN_CHARS = 40           # shorter sections carry no clinical content
MAX_CHARS = 4000         # longer ones are truncated: the model window is 512 tokens

KEEP = ["Chief Complaint", "History of Present Illness", "Past Medical History",
        "Major Surgical or Invasive Procedure", "Brief Hospital Course",
        "Discharge Diagnosis", "Physical Exam", "Family History"]
DROP = ["Discharge Instructions", "Followup Instructions", "Discharge Disposition",
        "Discharge Condition", "Facility", "Discharge Medications",
        "Medications on Admission", "Allergies", "Social History", "Pertinent Results"]
# heading at the start of a line; the body may begin on the next line or right after the colon
HEAD = re.compile(r"^ *([A-Z][A-Za-z /&\-]{3,40}):", re.M)


def sections(text):
    """Split a discharge summary into (section name, body)."""
    marks = [(m.start(), m.end(), m.group(1).strip()) for m in HEAD.finditer(text)]
    out = []
    for i, (s, e, name) in enumerate(marks):
        body = text[e:marks[i + 1][0] if i + 1 < len(marks) else len(text)]
        out.append((name, body.strip()))
    return out


def main():
    rng = np.random.default_rng(SEED)
    src = RAW / "note" / "discharge.csv.gz"
    tmp = Path("/tmp/d.csv")
    path = tmp if tmp.exists() else src

    # first pass: which patients are present
    ids = set()
    for ch in pd.read_csv(path, usecols=["subject_id"], chunksize=200000):
        ids.update(ch.subject_id.tolist())
    ids = np.array(sorted(ids))
    keep_pat = set(rng.choice(ids, min(N_PATIENTS, len(ids)), replace=False).tolist())
    print(f"patients in total {len(ids)}, in subsample {len(keep_pat)}", flush=True)

    rows = []
    seen_notes = 0
    for ch in pd.read_csv(path, chunksize=50000):
        ch = ch[ch.subject_id.isin(keep_pat)]
        seen_notes += len(ch)
        for nid, sid, hid, ct, text in zip(ch.note_id, ch.subject_id, ch.hadm_id,
                                           ch.charttime, ch.text):
            for name, body in sections(str(text)):
                if name in DROP or name not in KEEP:
                    continue
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) < MIN_CHARS:
                    continue
                rows.append((f"{nid}::{name}", nid, int(sid),
                             int(hid) if pd.notna(hid) else -1, str(ct)[:10], name,
                             body[:MAX_CHARS]))
    df = pd.DataFrame(rows, columns=["chunk_id", "note_id", "subject_id", "hadm_id",
                                     "charttime", "section", "text"])
    df.to_parquet(OUT / "chunks.parquet", index=False)
    print(f"notes {seen_notes}, chunks {len(df)}, patients {df.subject_id.nunique()}")
    print(df.section.value_counts().to_string())
    print(f"text length: median {df.text.str.len().median():.0f}, "
          f"90% {df.text.str.len().quantile(.9):.0f}")


if __name__ == "__main__":
    main()
