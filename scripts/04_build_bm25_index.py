#!/usr/bin/env python3
"""
04_build_bm25_index.py  — OFFLINE step
Builds a BM25 index over candidate embedding texts for hybrid retrieval.
BM25 captures exact keyword matches (e.g. "NDCG", "Pinecone", "BGE") that
dense embeddings sometimes miss, and runs in microseconds at query time.

Saves: artifacts/bm25_index.pkl (fitted BM25Okapi object)
"""
import json, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rank_bm25 import BM25Okapi

def tokenize(text):
    import re
    text = text.lower()
    return re.findall(r'[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]', text)

def main():
    t0 = time.time()
    Path("artifacts").mkdir(exist_ok=True)
    
    ids, corpus = [], []
    with open("artifacts/embedding_text.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["candidate_id"])
            corpus.append(tokenize(d["text"]))
    
    print(f"Building BM25 index over {len(corpus)} docs...", flush=True)
    bm25 = BM25Okapi(corpus)
    
    with open("artifacts/bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    
    print(f"BM25 index built in {time.time()-t0:.1f}s -> artifacts/bm25_index.pkl")
    print(f"Vocab size: {len(bm25.idf)} terms")
    
    # Quick sanity check
    query_tokens = tokenize("machine learning embedding retrieval NDCG ranking vector database pinecone")
    scores = bm25.get_scores(query_tokens)
    import numpy as np
    top5_idx = np.argsort(scores)[-5:][::-1]
    print("Top-5 BM25 hits for ML/embedding query:")
    for i in top5_idx:
        print(f"  {ids[i]}: score={scores[i]:.2f}")

if __name__ == "__main__":
    main()
