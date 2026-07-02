#!/usr/bin/env python3
"""
02_build_features.py
OFFLINE precompute step (not part of the 5-minute online ranking budget —
see submission_spec.md Section 3: "pre-computation may exceed the 5-minute
window... document this clearly").

Streams candidates.jsonl once, memory-safely (never holds all 100K parsed
dicts in memory at once — flushes to parquet in chunks), and writes:
  artifacts/features.parquet      -- one row per candidate, all engineered
                                      features EXCEPT the raw embedding text
  artifacts/embedding_text.jsonl  -- candidate_id -> embedding_text, kept
                                      separate so features.parquet stays small

Usage:
    python scripts/02_build_features.py --candidates data/candidates.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd
from features import extract_features

CHUNK_SIZE = 5000


def stream_candidates(path):
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--out-features", default="artifacts/features.parquet")
    ap.add_argument("--out-embed-text", default="artifacts/embedding_text.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)

    chunk_rows = []
    all_chunks = []
    n = 0

    with open(args.out_embed_text, "w", encoding="utf-8") as embed_f:
        for rec in stream_candidates(args.candidates):
            n += 1
            if args.limit and n > args.limit:
                break
            try:
                feat = extract_features(rec)
            except Exception as e:
                print(f"WARN: failed on {rec.get('candidate_id')}: {e}", file=sys.stderr)
                continue

            embed_text = feat.pop("_embedding_text")
            embed_f.write(json.dumps({"candidate_id": feat["candidate_id"], "text": embed_text}) + "\n")

            # JSON-encode dict-valued columns so parquet stays flat/simple
            feat["must_have_coverage_detail"] = json.dumps(feat["must_have_coverage_detail"])
            feat["latent_concepts_matched"] = json.dumps(feat["latent_concepts_matched"])

            chunk_rows.append(feat)
            if len(chunk_rows) >= CHUNK_SIZE:
                all_chunks.append(pd.DataFrame(chunk_rows))
                chunk_rows = []
                if n % 20000 == 0:
                    elapsed = time.time() - t0
                    print(f"  ...{n} candidates processed ({elapsed:.1f}s elapsed)", file=sys.stderr)

    if chunk_rows:
        all_chunks.append(pd.DataFrame(chunk_rows))

    df = pd.concat(all_chunks, ignore_index=True)
    df.to_parquet(args.out_features, index=False)

    elapsed = time.time() - t0
    print(f"Done. {len(df)} candidates -> {args.out_features} ({df.memory_usage(deep=True).sum()/1e6:.1f} MB in memory)")
    print(f"Embedding text -> {args.out_embed_text}")
    print(f"Total time: {elapsed:.1f}s")

    # quick gate-rate sanity printout
    print("\n--- Gate trigger rates ---")
    for col in ["gate_non_tech_title", "gate_consulting_only", "gate_research_only",
                "gate_honeypot", "gate_closed_source"]:
        rate = df[col].mean()
        print(f"  {col}: {rate*100:.2f}%  ({df[col].sum()} candidates)")


if __name__ == "__main__":
    main()
