"""Вкладення медичними моделями (sentence-similarity). Моделі публічні, дані локальні."""
import time, sys
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"pubmedbert": "NeuML/pubmedbert-base-embeddings",
          "medembed": "abhinand/MedEmbed-small-v0.1"}

def main(keys):
    texts = pd.read_parquet(ROOT/"data"/"chunks.parquet", columns=["text"]).text.tolist()
    out = ROOT/"data"/"emb"; out.mkdir(exist_ok=True)
    for k in keys:
        p = out/f"{k}.npy"
        if p.exists(): print(f"{k}: вже є"); continue
        m = SentenceTransformer(MODELS[k], device="mps"); m.max_seq_length = 512
        t0 = time.time()
        order = np.argsort([len(t) for t in texts])
        e = m.encode([texts[i] for i in order], batch_size=96, normalize_embeddings=True, convert_to_numpy=True)
        full = np.empty_like(e); full[order] = e
        np.save(p, full.astype(np.float32))
        print(f"{k}: {full.shape} за {time.time()-t0:.0f}с", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
