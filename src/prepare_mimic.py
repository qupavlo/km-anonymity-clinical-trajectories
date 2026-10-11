"""Підготовка фрагментів з виписок MIMIC-IV-Note.

Один фрагмент = одна клінічно змістовна секція виписки. Адміністративні й шаблонні секції
(інструкції при виписці, подальші візити, заклад) відкидаємо: вони однакові в тисяч пацієнтів
і домінували б у просторі вкладень.

Підвибірка пацієнтів задається N_PATIENTS, щоб обсяг був порівнянний із прогоном на Synthea
(~100 тис. фрагментів) і щоб точний пошук сусідів лишався здійсненним на одній машині.

Дані не залишають цю машину (умови DUA PhysioNet).

Вихід: data/chunks.parquet
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
MIN_CHARS = 40           # коротші секції не несуть клінічного змісту
MAX_CHARS = 4000         # довші обрізаємо: вікно моделі 512 токенів

KEEP = ["Chief Complaint", "History of Present Illness", "Past Medical History",
        "Major Surgical or Invasive Procedure", "Brief Hospital Course",
        "Discharge Diagnosis", "Physical Exam", "Family History"]
DROP = ["Discharge Instructions", "Followup Instructions", "Discharge Disposition",
        "Discharge Condition", "Facility", "Discharge Medications",
        "Medications on Admission", "Allergies", "Social History", "Pertinent Results"]
# заголовок на початку рядка; текст може йти як з нового рядка, так і одразу після двокрапки
HEAD = re.compile(r"^ *([A-Z][A-Za-z /&\-]{3,40}):", re.M)


def sections(text):
    """Розбити виписку на (назва секції, тіло)."""
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

    # перший прохід: які пацієнти є
    ids = set()
    for ch in pd.read_csv(path, usecols=["subject_id"], chunksize=200000):
        ids.update(ch.subject_id.tolist())
    ids = np.array(sorted(ids))
    keep_pat = set(rng.choice(ids, min(N_PATIENTS, len(ids)), replace=False).tolist())
    print(f"пацієнтів усього {len(ids)}, у підвибірці {len(keep_pat)}", flush=True)

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
    print(f"нотаток {seen_notes}, фрагментів {len(df)}, пацієнтів {df.subject_id.nunique()}")
    print(df.section.value_counts().to_string())
    print(f"довжина тексту: медіана {df.text.str.len().median():.0f}, "
          f"90% {df.text.str.len().quantile(.9):.0f}")


if __name__ == "__main__":
    main()
