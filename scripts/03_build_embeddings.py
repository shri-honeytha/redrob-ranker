#!/usr/bin/env python3
"""
03_build_embeddings.py
OFFLINE precompute step. Builds a dense semantic embedding for every
candidate's profile text and caches it to disk as a memory-mapped .npy
array, so the online ranking step only needs a single matrix multiply
against the JD vector (milliseconds, even for 100K rows).

TWO BACKENDS:

  --backend tfidf  (DEFAULT, network-free)
      TF-IDF (1-2 grams) -> TruncatedSVD to 256 dims -> L2-normalized.
      Runs fully offline with no model download, which matters because this
      project must be reproducible in network-restricted environments (this
      dev sandbox has no huggingface.co egress) AND because the competition's
      online ranking step explicitly forbids network access anyway. This is
      the backend actually used to produce the shipped submission.

  --backend bge    (OPTIONAL, requires internet ONCE during this offline
                    precompute step to download BAAI/bge-small-en-v1.5 —
                    this is explicitly allowed by the spec: "pre-computation
                    may exceed the 5-minute window... document this clearly".
                    The downloaded model is then cached locally; no network
                    is used at ranking time either way.)
      True dense sentence embeddings via sentence-transformers. Generally a
      stronger semantic signal than TF-IDF/SVD for paraphrase-style matches
      (e.g. a candidate describing "approximate nearest neighbour search"
      when the JD says "vector database"), at the cost of requiring the
      one-time model download. If you have internet access on your dev
      machine, switch to this backend and re-run; everything downstream
      (rank.py) reads whichever embeddings.npy is on disk and does not care
      which backend produced it.

Usage:
    python scripts/03_build_embeddings.py --backend tfidf
    python scripts/03_build_embeddings.py --backend bge      # needs internet once
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_embedding_texts(path):
    ids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["candidate_id"])
            texts.append(d["text"])
    return ids, texts


def build_tfidf_svd(texts, n_components=256, max_features=50000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2), min_df=2, max_df=0.6,
        sublinear_tf=True, stop_words="english",
    )
    tfidf = vectorizer.fit_transform(texts)
    print(f"  TF-IDF matrix: {tfidf.shape}, nnz={tfidf.nnz}", file=sys.stderr)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(tfidf)
    dense = normalize(dense)  # L2 normalize so cosine sim = dot product
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD explained variance ({n_components} dims): {explained:.3f}", file=sys.stderr)
    return dense.astype(np.float32), vectorizer, svd


def build_bge(texts, model_name="BAAI/bge-small-en-v1.5", batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return embeddings.astype(np.float32), model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-text", default="artifacts/embedding_text.jsonl")
    ap.add_argument("--out", default="artifacts/embeddings.npy")
    ap.add_argument("--out-ids", default="artifacts/embedding_ids.json")
    ap.add_argument("--backend", choices=["tfidf", "bge"], default="tfidf")
    ap.add_argument("--n-components", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    ids, texts = load_embedding_texts(args.embed_text)
    print(f"Loaded {len(ids)} embedding texts", file=sys.stderr)

    if args.backend == "tfidf":
        embeddings, vectorizer, svd = build_tfidf_svd(texts, n_components=args.n_components)
        # persist the fitted vectorizer+SVD so the JD can be embedded
        # identically at ranking time
        import pickle
        with open("artifacts/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)
        backend_meta = {"backend": "tfidf", "n_components": args.n_components}
    else:
        embeddings, model_name = build_bge(texts)
        backend_meta = {"backend": "bge", "model_name": model_name}

    np.save(args.out, embeddings)
    with open(args.out_ids, "w") as f:
        json.dump(ids, f)
    with open("artifacts/embedding_backend.json", "w") as f:
        json.dump(backend_meta, f)

    elapsed = time.time() - t0
    print(f"Saved embeddings {embeddings.shape} -> {args.out} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
