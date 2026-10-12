"""Embeddings of MIMIC chunks computed with local models (the data never leaves the machine)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"bge": "BAAI/bge-small-en-v1.5", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
MAX_LEN = 512


def main():
    df = pd.read_parquet(ROOT / "data" / "chunks.parquet", columns=["text"])
    texts = df.text.tolist()
    out = ROOT / "data" / "emb"
    out.mkdir(exist_ok=True)
    for key, name in MODELS.items():
        path = out / f"{key}.npy"
        if path.exists():
            print(f"{key}: already present")
            continue
        model = SentenceTransformer(name, device="mps")
        model.max_seq_length = MAX_LEN
        ntok = [len(x) for x in model.tokenizer(texts[:5000])["input_ids"]]
        t0 = time.time()
        order = np.argsort([len(t) for t in texts])          # sorting by length speeds up batching
        emb = model.encode([texts[i] for i in order], batch_size=128,
                           normalize_embeddings=True, convert_to_numpy=True)
        full = np.empty_like(emb)
        full[order] = emb
        np.save(path, full.astype(np.float32))
        print(f"{key}: {full.shape} by {time.time() - t0:.0f}s, "
              f"truncated above {MAX_LEN} tokens: {np.mean(np.array(ntok) > MAX_LEN):.3f}", flush=True)


if __name__ == "__main__":
    main()
