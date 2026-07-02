#!/usr/bin/env python3
"""
rank.py  v2 — Hybrid BM25 + Semantic + Full Redrob Signal Pipeline

ARCHITECTURE (2-stage, CPU-only, <5 min total):

  Offline (run once, any time budget):
    scripts/02_build_features.py    → artifacts/features.parquet  (45s)
    scripts/03_build_embeddings.py  → artifacts/embeddings.npy    (67s)
    scripts/04_build_bm25_index.py  → artifacts/bm25_index.pkl   (28s)

  Online (<5 min, CPU-only, no network):
    1. Load artifacts from disk                     (~2s)
    2. Embed JD with same TF-IDF vectorizer         (<1s)
    3. Cosine similarity (matrix multiply, 100K)    (<1s)
    4. BM25 query scores                            (<2s)
    5. Hybrid score = 0.35×BM25 + 0.65×cosine      (instant)
    6. compute_scores() vectorized — all 23 signals (~1s)
    7. Top 500 candidates selected
    8. [Optional] Cross-encoder reranking on top 500 (~10s if enabled)
    9. Generate reasoning for top 100
   10. Write CSV + self-validate

Usage:
    python rank.py --candidates data/candidates.jsonl --out submission.csv

To enable cross-encoder (better reranking, needs model download once):
    pip install sentence-transformers
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
    python rank.py --candidates data/candidates.jsonl --out submission.csv --cross-encoder
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jd_requirements import JD_TITLE_ANCHOR_TEXT
from scoring import compute_scores, score_breakdown, cross_encode_rerank, USE_CROSS_ENCODER, WEIGHTS
from reasoning import generate_reasoning, validate_rank_consistency, detect_template_repetition

TOP_N           = 100
RERANK_POOL     = 500    # cross-encoder candidates before final top-100

JD_FULL_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A talent intelligence platform.
Own the intelligence layer: ranking retrieval and matching systems that decide what 
recruiters see when they search for candidates. Deep technical depth in modern ML 
systems: embeddings retrieval ranking LLMs fine-tuning. Production experience with 
embeddings-based retrieval systems sentence-transformers OpenAI embeddings BGE E5 
deployed to real users handling embedding drift index refresh retrieval-quality 
regression in production. Production experience with vector databases or hybrid search 
infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS. 
Strong Python production code quality. Hands-on experience designing evaluation 
frameworks for ranking systems NDCG MRR MAP offline-to-online correlation A/B test 
interpretation. Ideal candidate 6-8 years total experience 4-5 years in applied ML AI 
roles at product companies has shipped at least one end-to-end ranking search or 
recommendation system to real users at meaningful scale has strong opinions about 
retrieval hybrid vs dense evaluation offline vs online and LLM integration when to 
fine-tune vs prompt. Learning to rank LTR approximate nearest neighbour ANN HNSW 
feature engineering model serving evaluation infrastructure recommendation system 
search relevance candidate matching feed ranking personalization.
"""


def load_artifacts(args):
    t0 = time.time()
    df = pd.read_parquet("artifacts/features.parquet")
    embeddings = np.load("artifacts/embeddings.npy")
    with open("artifacts/embedding_ids.json") as f:
        embedding_ids = json.load(f)
    with open("artifacts/embedding_backend.json") as f:
        backend_meta = json.load(f)
    with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer_pkg = pickle.load(f)

    # BM25 index (optional but strongly recommended)
    bm25_pkg = None
    bm25_path = Path("artifacts/bm25_index.pkl")
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_pkg = pickle.load(f)
    else:
        print("  WARNING: No BM25 index found. Run scripts/04_build_bm25_index.py for hybrid scoring.", 
              file=sys.stderr)

    print(f"  Artifacts loaded in {time.time()-t0:.1f}s — {len(df)} candidates, "
          f"emb={embeddings.shape}, backend={backend_meta['backend']}, "
          f"bm25={'yes' if bm25_pkg else 'no'}", file=sys.stderr)
    return df, embeddings, embedding_ids, backend_meta, vectorizer_pkg, bm25_pkg


def embed_jd(backend_meta, vectorizer_pkg):
    from sklearn.preprocessing import normalize
    vect = vectorizer_pkg["vectorizer"]
    svd  = vectorizer_pkg["svd"]
    jd_tfidf = vect.transform([JD_FULL_TEXT])
    jd_dense = normalize(svd.transform(jd_tfidf))
    return jd_dense[0].astype(np.float32)


def cosine_sims_for_df(embeddings, embedding_ids, df, jd_vec):
    id_to_row = {cid: i for i, cid in enumerate(embedding_ids)}
    rows = [id_to_row.get(cid, -1) for cid in df["candidate_id"]]
    emb_rows = np.array([
        embeddings[r] if r >= 0 else np.zeros(embeddings.shape[1], dtype=np.float32)
        for r in rows
    ], dtype=np.float32)
    return (emb_rows @ jd_vec).astype(np.float32)


def bm25_scores_for_df(bm25_pkg, df):
    """Query BM25 with JD keywords and align to df row order."""
    import re
    def tokenize(t):
        return re.findall(r'[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]', t.lower())

    jd_tokens = tokenize(JD_FULL_TEXT)
    bm25 = bm25_pkg["bm25"]
    bm25_ids = bm25_pkg["ids"]
    
    raw_scores = bm25.get_scores(jd_tokens).astype(np.float32)
    id_to_bm25 = dict(zip(bm25_ids, raw_scores))
    aligned = np.array([id_to_bm25.get(cid, 0.0) for cid in df["candidate_id"]], 
                        dtype=np.float32)
    return aligned


def load_assessment_scores(candidates_path):
    scores = {}
    with open(candidates_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            sigs = rec.get("redrob_signals") or {}
            ass = sigs.get("skill_assessment_scores")
            if ass:
                scores[rec["candidate_id"]] = ass
    return scores


def load_embedding_texts_for_top(candidates_path, top_ids_set):
    """Load raw embedding text for cross-encoder (only needed for top-500)."""
    texts = {}
    with open(candidates_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            cid = rec["candidate_id"]
            if cid in top_ids_set:
                from features import extract_features, dedupe_descriptions
                career = rec.get("career_history", []) or []
                descs = dedupe_descriptions(career)
                profile = rec.get("profile", {})
                skills = rec.get("skills", []) or []
                texts[cid] = " ".join(filter(None, [
                    profile.get("headline", ""), profile.get("summary", ""),
                    ", ".join((s.get("name","")).lower() for s in skills),
                    " ".join(descs),
                ]))
    return texts


def write_csv(rows, out_path):
    lines = ["candidate_id,rank,score,reasoning"]
    for row in rows:
        r = row["reasoning"].replace('"', "'").replace("\n", " ").strip()
        lines.append(f'{row["candidate_id"]},{row["rank"]},{row["score"]:.6f},"{r}"')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_validator(out_path):
    sys.path.insert(0, "scripts")
    try:
        from validate_submission import validate_submission
        errors = validate_submission(out_path)
        if errors:
            print(f"  ❌ Validation FAILED:", file=sys.stderr)
            for e in errors: print(f"    - {e}", file=sys.stderr)
            return False
        print("  ✅ Submission format valid", file=sys.stderr)
        return True
    except Exception as e:
        print(f"  WARNING: validator error: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cross-encoder", action="store_true",
                    help="Enable cross-encoder reranking (requires model download)")
    ap.add_argument("--rerank-pool", type=int, default=RERANK_POOL,
                    help=f"Candidates fed into cross-encoder (default {RERANK_POOL})")
    args = ap.parse_args()

    wall_t0 = time.time()
    print("=== Redrob Ranker v2 — Hybrid BM25 + Semantic + 23 Redrob Signals ===",
          file=sys.stderr)

    # 1. Load
    print("1. Loading artifacts...", file=sys.stderr)
    df, embeddings, embedding_ids, backend_meta, vectorizer_pkg, bm25_pkg = load_artifacts(args)

    # 2. JD embedding
    print("2. Embedding JD...", file=sys.stderr)
    t = time.time()
    jd_vec = embed_jd(backend_meta, vectorizer_pkg)
    print(f"   done in {time.time()-t:.2f}s", file=sys.stderr)

    # 3. Cosine similarities
    print("3. Cosine similarities (matrix multiply)...", file=sys.stderr)
    t = time.time()
    cosine_sims = cosine_sims_for_df(embeddings, embedding_ids, df, jd_vec)
    print(f"   done in {time.time()-t:.3f}s", file=sys.stderr)

    # 4. BM25 hybrid
    bm25_scores = None
    if bm25_pkg:
        print("4. BM25 query scores...", file=sys.stderr)
        t = time.time()
        bm25_scores = bm25_scores_for_df(bm25_pkg, df)
        print(f"   done in {time.time()-t:.2f}s", file=sys.stderr)
    else:
        print("4. BM25 skipped (index not found)", file=sys.stderr)

    # 5. Full scoring (vectorized)
    print("5. Vectorized scoring (all 23 redrob signals)...", file=sys.stderr)
    t = time.time()
    scores = compute_scores(df, cosine_sims, bm25_scores)
    df = df.copy()
    df["score"] = scores
    df["cosine_sim"] = cosine_sims
    if bm25_scores is not None:
        df["bm25_score"] = bm25_scores
    print(f"   done in {time.time()-t:.3f}s", file=sys.stderr)

    # 6. Select rerank pool
    print(f"6. Selecting top-{args.rerank_pool} for reranking...", file=sys.stderr)
    top_pool = df.nlargest(args.rerank_pool, "score").copy()
    print(f"   Score range in pool: {top_pool['score'].max():.4f} — {top_pool['score'].min():.4f}",
          file=sys.stderr)

    # 7. Cross-encoder reranking (optional)
    if args.cross_encoder or USE_CROSS_ENCODER:
        print("7. Cross-encoder reranking...", file=sys.stderr)
        t = time.time()
        top_ids_set = set(top_pool["candidate_id"])
        embedding_texts = load_embedding_texts_for_top(args.candidates, top_ids_set)
        top_pool["_embedding_text"] = top_pool["candidate_id"].map(embedding_texts)
        top_pool = cross_encode_rerank(top_pool, JD_FULL_TEXT, top_n=TOP_N)
        print(f"   done in {time.time()-t:.1f}s", file=sys.stderr)
    else:
        print("7. Cross-encoder: disabled (use --cross-encoder flag to activate)",
              file=sys.stderr)

    # 8. Final top-100 with tie-breaking
    top100 = top_pool.nlargest(TOP_N, "score") if len(top_pool) > TOP_N else top_pool
    top100 = top100.sort_values(["score","candidate_id"], ascending=[False,True]).reset_index(drop=True)
    top100["rank"] = range(1, len(top100)+1)

    print(f"\n--- Top-10 ranked candidates ---", file=sys.stderr)
    for _, row in top100.head(10).iterrows():
        print(f"  Rank {int(row['rank']):2d} | {row['candidate_id']} | "
              f"{str(row.get('current_title',''))[:35]:35s} | "
              f"score={row['score']:.4f} | "
              f"coverage={row.get('must_have_coverage_total',0):.1f}/5 | "
              f"company={str(row.get('current_company',''))[:20]}",
              file=sys.stderr)

    # 9. Load assessment scores and generate reasoning
    print("\n8. Loading assessment scores + generating reasoning...", file=sys.stderr)
    t = time.time()
    assessment_scores_map = load_assessment_scores(args.candidates)

    bm25_id_to_score = {}
    if bm25_scores is not None and bm25_pkg:
        bm25_max = bm25_scores.max() + 1e-9
        bm25_id_to_score = dict(zip(df["candidate_id"],
                                     (bm25_scores / bm25_max).tolist()))

    top100_rows = []
    for _, row in top100.iterrows():
        cid = row["candidate_id"]
        row_dict = row.to_dict()
        row_dict["_assessment_scores"] = assessment_scores_map.get(cid)
        bdown = score_breakdown(row_dict, row["cosine_sim"],
                                 bm25_norm=bm25_id_to_score.get(cid, 0.0))
        reasoning = generate_reasoning(row_dict, int(row["rank"]), bdown)
        top100_rows.append({
            "candidate_id": cid,
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "reasoning": reasoning,
        })
    print(f"   done in {time.time()-t:.2f}s", file=sys.stderr)

    # Validate reasoning quality
    issues = validate_rank_consistency(top100_rows)
    if issues:
        print(f"  ⚠ {len(issues)} rank-consistency issues (first 3):", file=sys.stderr)
        for rank, issue in issues[:3]:
            print(f"    rank {rank}: {issue}", file=sys.stderr)
    warn = detect_template_repetition(top100_rows)
    if warn: print(f"  ⚠ {warn}", file=sys.stderr)

    # 10. Write + validate
    print("9. Writing CSV...", file=sys.stderr)
    write_csv(top100_rows, args.out)
    print(f"   Written: {args.out}", file=sys.stderr)
    print("10. Running format validation...", file=sys.stderr)
    run_validator(args.out)

    # Resource report
    wall_time = time.time() - wall_t0
    try:
        import resource
        mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
        print(f"\n⏱  Wall time: {wall_time:.1f}s  |  Peak RAM: {mem_gb:.2f} GB",
              file=sys.stderr)
    except:
        print(f"\n⏱  Wall time: {wall_time:.1f}s", file=sys.stderr)

    print(f"✅ Done — {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
