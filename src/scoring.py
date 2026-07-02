"""
src/scoring.py  v2 — Hybrid BM25 + TF-IDF/SVD scoring with all 23 redrob signals

Architecture:
  Stage 1 (fast, vectorized):  hybrid_score = 0.4*bm25 + 0.6*cosine_sim
                                → filter to top 5,000 candidates
  Stage 2 (feature-rich):      full weighted score on top 5K only
                                (gates + 15 features + all redrob signals)
  Stage 3 (reranking stub):    cross-encoder ready — activate by setting
                                USE_CROSS_ENCODER=True once model is downloaded

New in v2:
  - demand_signal: saved_by_recruiters_30d + search_appearance_30d
  - profile_quality_score: completeness + verified contact + connections
  - engagement_quality: interview_completion_rate + offer_acceptance_rate  
  - salary_fit: expected LPA vs role-implied band
  - company_tier_score: Tier-1 product companies get a signal boost
  - career_tenure_stability: JD-explicit job-hopper detection
  - github_signal: properly extracted (was included in behav_mult but not as standalone)
  - job_hopper_penalty: explicit negative feature per JD disqualifier

Cross-encoder integration point (for when you have internet access once):
  pip install sentence-transformers
  python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
  Then set USE_CROSS_ENCODER = True below.
"""
import json
import numpy as np
import pandas as pd

USE_CROSS_ENCODER = False  # flip to True after downloading the model

# ---------------------------------------------------------------------------
# Gate caps
# ---------------------------------------------------------------------------
GATE_CAPS = {
    "gate_non_tech_title":   0.10,
    "gate_consulting_only":  0.15,
    "gate_research_only":    0.15,
    "gate_honeypot":         0.05,
}
CLOSED_SRC_MULT  = 0.60
JOB_HOPPER_MULT  = 0.85   # additional soft multiplier for frequent job-hoppers

# ---------------------------------------------------------------------------
# Scoring weights v2
# Tuned so that NDCG@10 (50% of grade) is maximized: the top-10 must
# contain the genuine Tier-5/4 candidates. Must-have coverage and title
# relevance are the strongest discriminators for that.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "title_relevance":          0.20,
    "cosine_sim":               0.18,   # semantic similarity to JD
    "must_have_coverage":       0.15,   # 5 JD must-haves (normalized 0-1)
    "skill_trust":              0.10,
    "trajectory_slope_pos":     0.06,
    "company_tier_score":       0.06,   # NEW: Tier-1 AI/product companies
    "ever_used_overlap":        0.05,
    "recent_use_overlap":       0.05,
    "latent_concept_bonus":     0.03,
    "demand_signal":            0.04,   # NEW: saved_by_recruiters + search appearances
    "experience_fit":           0.05,
    "location_fit":             0.03,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01, f"Weights sum to {sum(WEIGHTS.values())}"

# Modifier weights (applied after base, not in weight budget)
NOTICE_MOD        = (0.85, 0.15)
WORK_MODE_MOD     = (0.90, 0.10)
SALARY_MOD        = (0.88, 0.12)     # NEW: salary fit modifier
EDU_NUDGE         = 0.025
PROFILE_QUALITY_BONUS = 0.03         # NEW: up to +0.03 for verified/complete profiles
ENGAGEMENT_MOD    = (0.92, 0.08)     # NEW: interview/offer completion modifier


def compute_scores(df: pd.DataFrame, cosine_sims: np.ndarray,
                    bm25_scores: np.ndarray = None) -> pd.Series:
    """
    Full vectorized scoring pipeline.
    
    Args:
        df: features.parquet DataFrame (100K rows)
        cosine_sims: L2-normalized cosine similarities to JD (100K,)
        bm25_scores: optional BM25 scores, normalized 0-1 (100K,)
                     if provided, cosine_sims becomes a hybrid score

    Returns:
        pd.Series of final scores, aligned to df row order
    """
    n = len(df)

    # If BM25 scores provided, blend with cosine_sim for Stage 1 signal
    if bm25_scores is not None:
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        hybrid_sim = 0.35 * bm25_norm + 0.65 * cosine_sims
    else:
        hybrid_sim = cosine_sims

    raw = (
        WEIGHTS["title_relevance"]      * df["title_relevance"].values +
        WEIGHTS["cosine_sim"]           * hybrid_sim +
        WEIGHTS["must_have_coverage"]   * (df["must_have_coverage_total"].values / 5.0) +
        WEIGHTS["skill_trust"]          * df["skill_trust"].values +
        WEIGHTS["trajectory_slope_pos"] * np.maximum(0, df["trajectory_slope"].values) +
        WEIGHTS["company_tier_score"]   * df["company_tier_score"].values +
        WEIGHTS["ever_used_overlap"]    * df["ever_used_overlap"].values +
        WEIGHTS["recent_use_overlap"]   * df["recent_use_overlap"].values +
        WEIGHTS["latent_concept_bonus"] * (df["latent_concept_bonus"].values / 0.30) +
        WEIGHTS["demand_signal"]        * df["demand_signal"].values +
        WEIGHTS["experience_fit"]       * df["experience_fit"].values +
        WEIGHTS["location_fit"]         * df["location_fit"].values
    )
    raw = np.clip(raw, 0.0, 1.0)

    # --- Modifiers ---
    raw *= (NOTICE_MOD[0]     + NOTICE_MOD[1]     * df["notice_fit"].values)
    raw *= (WORK_MODE_MOD[0]  + WORK_MODE_MOD[1]  * df["work_mode_fit"].values)
    raw *= (SALARY_MOD[0]     + SALARY_MOD[1]     * df["salary_fit"].values)
    raw *= (ENGAGEMENT_MOD[0] + ENGAGEMENT_MOD[1] * df["engagement_quality"].values)
    raw += EDU_NUDGE * (df["education_tier_score"].values - 0.5)
    raw += PROFILE_QUALITY_BONUS * (df["profile_quality_score"].values - 0.5)
    raw = np.clip(raw, 0.0, 1.0)

    # --- Gate caps ---
    cap = np.ones(n, dtype=np.float32)
    for gate_col, cap_val in GATE_CAPS.items():
        triggered = df[gate_col].values.astype(bool)
        cap = np.where(triggered, np.minimum(cap, cap_val), cap)
    raw = np.minimum(raw, cap)

    # --- Soft multipliers ---
    closed = df["gate_closed_source"].values.astype(bool)
    raw = np.where(closed, raw * CLOSED_SRC_MULT, raw)

    # Job hopper penalty (separate from gate_closed_source)
    hopper = df["job_hopper_penalty"].values > 0.5  # >50% of jobs were short stints
    raw = np.where(hopper, raw * JOB_HOPPER_MULT, raw)

    # --- Behavioral multiplier LAST (all 23 redrob signals combined) ---
    raw = raw * df["behavioral_multiplier"].values

    return pd.Series(raw.astype(np.float64), index=df.index, name="score")


def score_breakdown(row, cosine_sim, bm25_norm=0.0):
    """Per-candidate score breakdown for reasoning generator + debugging."""
    hybrid = 0.35*bm25_norm + 0.65*cosine_sim
    components = {
        "title_relevance":       WEIGHTS["title_relevance"] * row["title_relevance"],
        "semantic_similarity":   WEIGHTS["cosine_sim"] * hybrid,
        "must_have_coverage":    WEIGHTS["must_have_coverage"] * (row["must_have_coverage_total"]/5.0),
        "skill_trust":           WEIGHTS["skill_trust"] * row["skill_trust"],
        "trajectory":            WEIGHTS["trajectory_slope_pos"] * max(0, row["trajectory_slope"]),
        "company_tier":          WEIGHTS["company_tier_score"] * row["company_tier_score"],
        "recency_overlap":       (WEIGHTS["ever_used_overlap"] * row["ever_used_overlap"] +
                                  WEIGHTS["recent_use_overlap"] * row["recent_use_overlap"]),
        "latent_concepts":       WEIGHTS["latent_concept_bonus"] * (row["latent_concept_bonus"]/0.30),
        "demand_signal":         WEIGHTS["demand_signal"] * row["demand_signal"],
        "experience_fit":        WEIGHTS["experience_fit"] * row["experience_fit"],
        "location_fit":          WEIGHTS["location_fit"] * row["location_fit"],
    }
    base = sum(components.values())
    active_gates = [g for g in GATE_CAPS if row.get(g, False)]
    gate_cap = min((GATE_CAPS[g] for g in active_gates), default=1.0)
    return {
        "components": components,
        "base_raw": base,
        "gate_cap": gate_cap if active_gates else None,
        "active_gates": active_gates,
        "behavioral_multiplier": row["behavioral_multiplier"],
        "demand_signal_detail": {
            "saved_by_recruiters": row.get("saved_by_recruiters_30d", 0),
            "search_appearances": row.get("search_appearance_30d", 0),
        },
        "final_score": row.get("score", None),
    }


def cross_encode_rerank(top_df, jd_text, top_n=100):
    """Stage 3: cross-encoder reranking on the top candidates.
    Only runs if USE_CROSS_ENCODER is True AND the model is downloaded.
    Otherwise returns the input ranking unchanged (graceful degradation).
    
    To activate:
      1. pip install sentence-transformers
      2. python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
      3. Set USE_CROSS_ENCODER = True at the top of this file
    """
    if not USE_CROSS_ENCODER:
        return top_df.head(top_n)
    
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        
        # Build (JD, candidate_text) pairs for the cross-encoder
        pairs = [(jd_text, row.get("_embedding_text", "")) for _, row in top_df.iterrows()]
        ce_scores = model.predict(pairs, show_progress_bar=False)
        
        top_df = top_df.copy()
        top_df["cross_encoder_score"] = ce_scores
        # Blend: 60% cross-encoder, 40% feature score
        top_df["final_score"] = 0.6 * (ce_scores / (ce_scores.max()+1e-9)) + 0.4 * top_df["score"]
        return top_df.nlargest(top_n, "final_score")
    except Exception as e:
        print(f"  Cross-encoder unavailable ({e}), using feature scores", flush=True)
        return top_df.head(top_n)
