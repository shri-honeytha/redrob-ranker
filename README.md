# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Submission by:** [YOUR TEAM NAME]  
**Track:** Intelligent Candidate Discovery & Ranking Challenge

---

## What this system does

Ranks 100,000 candidates against a Senior AI Engineer job description by combining:

1. **Hard gates** (non-technical title, consulting-only career, honeypot/internal inconsistency)  
2. **Semantic embedding similarity** (TF-IDF + SVD, 256-dim, pre-computed offline)  
3. **Engineered structural features** (5 JD must-have coverage scores, skill trust, career trajectory, recency-weighted skill overlap, latent concept graph)  
4. **Behavioral multiplier** (Redrob platform signals: last-active date, recruiter response rate, reachability composite, interview/offer rates)

No LLM calls are made at ranking time — all computation is vectorized numpy/pandas over pre-cached arrays.

---

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

### Step 1 — Offline precompute (run once; may exceed 5 min; fine per spec)

```bash
python scripts/02_build_features.py --candidates data/candidates.jsonl
python scripts/03_build_embeddings.py --backend tfidf
```

This produces `artifacts/features.parquet`, `artifacts/embeddings.npy`, and supporting files.

> **Optional / recommended:** if your machine has internet access, swap the second command for:  
> `python scripts/03_build_embeddings.py --backend bge`  
> This uses `BAAI/bge-small-en-v1.5` (one-time ~33MB download) for stronger semantic matching.

### Step 2 — Online ranking (≤5 min, ≤16 GB, CPU only, no network)

```bash
python rank.py --candidates data/candidates.jsonl --out submission.csv
```

Outputs `submission.csv` — 100 rows, format-validated automatically.

**Measured runtime (this machine):** 22.1s wall time, 0.54 GB peak RAM.

---

## Reproduce from scratch

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
# Place candidates.jsonl in data/ (not committed — 487 MB)
python scripts/02_build_features.py --candidates data/candidates.jsonl
python scripts/03_build_embeddings.py --backend tfidf
python rank.py --candidates data/candidates.jsonl --out submission.csv
```

---

## Architecture

```
candidates.jsonl (100K)
     │
     ├─► [OFFLINE] scripts/02_build_features.py
     │     • parse + normalize → engineered features
     │     • 5 gates, 10 scoring features, behavioral signals
     │     → artifacts/features.parquet
     │
     ├─► [OFFLINE] scripts/03_build_embeddings.py
     │     • deduplicate description text (quirk: ~36% of candidates
     │       have repeated paragraphs across jobs)
     │     • TF-IDF (1-2grams, 50K vocab) → TruncatedSVD (256 dims)
     │     → artifacts/embeddings.npy  [100K × 256 float32]
     │
     └─► [ONLINE ≤5min] rank.py
           • load artifacts (file reads only)
           • embed JD with same vectorizer
           • cosine similarity: single matrix multiply
           • compute_scores(): vectorized gate-caps + weighted features
           • top-100 candidates
           • generate_reasoning(): slot-based, 3 templates/slot (varied
             by candidate_id hash for determinism + Stage-4 variation)
           • rank-consistency + template-repeat validation
           → submission.csv
```

### Key design decisions

| Decision | Rationale |
|---|---|
| TF-IDF+SVD embedding (default) | Network-free, reproducible in any sandbox; BGE-small is a drop-in swap via `--backend bge` if internet available once |
| Gate caps (not zeros) | Spec flags "all identical scores" as a rejection; caps keep score distribution differentiable |
| Honeypot gate: 2 signals, calibrated vs full 100K pool | Tested 4 candidate signals; 2 had >1,000 false-positive triggers on the full pool (dropped). Kept signals fire on 63/100K candidates (~80 stated in spec) |
| `offer_acceptance_rate=-1` and `github_activity_score=-1` treated as neutral | Spec schema defines -1 as "no prior data" not "zero" — treating as bad score would silently penalize new-to-platform candidates |
| Behavioral multiplier applied last | JD's own framing: "down-weight [unavailable candidates] appropriately" — it's a modifier on technical fit, not a competing signal for weight budget |
| Reasoning: template + candidate_id hash (not random) | Same run = identical output, required for Stage 3 code reproduction; hash also ensures structural variation across rows for Stage 4 manual review |

---

## Methodology summary (≤200 words, for portal metadata)

Our system combines structured feature engineering with semantic similarity to rank candidates the way a great recruiter would — not by keyword count, but by understanding who actually fits.

**Offline:** We extract 15 engineered features per candidate: career coherence (title relevance, trajectory slope, product-vs-consulting ratio), 5 JD must-have coverage scores (each checked in both the structured skills list AND the free-text description, to catch candidates who describe vector search without labeling it as "Pinecone"), Eightfold-inspired recency-weighted skill overlap (ever-used vs. currently-using), a concept graph for latent skill inference, and a reachability composite (last-active + response rate + open-to-work flag, explicitly named in the JD). Candidate text is embedded via TF-IDF + SVD (network-free; BGE-small also supported).

**Online (< 30s):** We embed the JD identically, run a single cosine-similarity matrix multiply, apply gate caps (non-technical titles, consulting-only careers, honeypot profiles — each calibrated against the full 100K pool), combine features by weighted sum, apply the behavioral multiplier last, and take the top 100.

**Every component is traceable to a named feature — no black-box step.** Reasoning strings are populated from actual extracted profile fields, never templated filler or hallucinated claims.

---

## Files

| File | Purpose |
|---|---|
| `rank.py` | Main ranking script (online step) |
| `src/features.py` | Feature extraction: gates + 15 scoring features |
| `src/jd_requirements.py` | Structured JD facts (single source of truth) |
| `src/scoring.py` | Vectorized score combiner |
| `src/reasoning.py` | Slot-based reasoning generator + validators |
| `src/metrics.py` | NDCG@k / MAP / P@k implementation (unit-tested) |
| `scripts/02_build_features.py` | Offline feature precompute |
| `scripts/03_build_embeddings.py` | Offline embedding build (TF-IDF or BGE) |
| `scripts/validate_submission.py` | Official format validator (unchanged from bundle) |
| `scripts/01_explore_data.py` | Data exploration — full 100K stats |
| `docs/BUILD_SPEC.md` | Team design spec |
| `docs/HANDOVER.md` | Build log and next-steps tracker |
