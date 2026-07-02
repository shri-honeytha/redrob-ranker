#!/usr/bin/env python3
"""
app.py — Streamlit sandbox demo for the Redrob "India Runs" Track 1 submission.

This IS the working sandbox the submission spec asks for (README / BUILD_SPEC
Section 12: "working HF Spaces / Colab / Streamlit / Docker sandbox on a
small sample"). It runs the REAL pipeline (src/features.py, src/scoring.py,
src/reasoning.py — the exact code used to produce submission.csv), just
against a small synthetic sample (data/candidates_sample.jsonl, 125
candidates) instead of the real 100K/487MB candidates.jsonl, which is too
large to ship in a demo sandbox.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy (pick one, then put the URL in submission_metadata.yaml -> sandbox_link):
    - Streamlit Community Cloud (streamlit.io/cloud) — point it at your GitHub
      repo, entry point app.py. Free, one click.
    - Hugging Face Spaces — create a Space, SDK = Streamlit, push this repo.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scoring import compute_scores, score_breakdown
from reasoning import generate_reasoning, validate_rank_consistency, detect_template_repetition

st.set_page_config(page_title="Redrob Ranker — Sandbox Demo", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden !important;}
    header [data-testid="stToolbar"] {display: none !important;}
    footer {visibility: hidden !important;}
    </style>
""", unsafe_allow_html=True)

theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        </style>
    """, unsafe_allow_html=True)

DEFAULT_JD = """Senior AI Engineer — Founding Team, Redrob AI (Series A talent intelligence platform).
Own the intelligence layer: ranking, retrieval and matching systems that decide what
recruiters see when they search for candidates. Deep technical depth in modern ML
systems: embeddings, retrieval, ranking, LLMs, fine-tuning. Production experience with
embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5)
deployed to real users, handling embedding drift, index refresh, retrieval-quality
regression in production. Production experience with vector databases or hybrid search
infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python production code quality. Hands-on experience designing evaluation
frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test
interpretation). Ideal candidate: 6-8 years total experience, 4-5 years in applied ML/AI
roles at product companies, has shipped at least one end-to-end ranking/search/
recommendation system to real users at meaningful scale, has strong opinions about
retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration
(when to fine-tune vs prompt)."""

ARTIFACT_DIR = Path("artifacts")
SAMPLE_CANDIDATES = Path("data/candidates_sample.jsonl")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    import numpy as np
    import pickle
    df = pd.read_parquet(ARTIFACT_DIR / "features.parquet")
    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
    embedding_ids = json.loads((ARTIFACT_DIR / "embedding_ids.json").read_text())
    backend_meta = json.loads((ARTIFACT_DIR / "embedding_backend.json").read_text())
    with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer_pkg = pickle.load(f)
    bm25_pkg = None
    bm25_path = ARTIFACT_DIR / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_pkg = pickle.load(f)
    return df, embeddings, embedding_ids, backend_meta, vectorizer_pkg, bm25_pkg


@st.cache_resource(show_spinner=False)
def load_assessment_scores():
    scores = {}
    with open(SAMPLE_CANDIDATES) as f:
        for line in f:
            rec = json.loads(line)
            sigs = rec.get("redrob_signals") or {}
            ass = sigs.get("skill_assessment_scores")
            if ass:
                scores[rec["candidate_id"]] = ass
    return scores


def embed_jd(jd_text, vectorizer_pkg):
    from sklearn.preprocessing import normalize
    vect, svd = vectorizer_pkg["vectorizer"], vectorizer_pkg["svd"]
    jd_tfidf = vect.transform([jd_text])
    return normalize(svd.transform(jd_tfidf))[0]


def bm25_scores_for_df(bm25_pkg, df, jd_text):
    import re
    import numpy as np

    def tokenize(t):
        return re.findall(r"[a-z0-9][a-z0-9\-.]*[a-z0-9]|[a-z0-9]", t.lower())

    bm25, bm25_ids = bm25_pkg["bm25"], bm25_pkg["ids"]
    raw = bm25.get_scores(tokenize(jd_text)).astype("float32")
    id_to_score = dict(zip(bm25_ids, raw))
    return np.array([id_to_score.get(cid, 0.0) for cid in df["candidate_id"]], dtype="float32")


def run_ranking(jd_text, top_n):
    import numpy as np

    df, embeddings, embedding_ids, backend_meta, vectorizer_pkg, bm25_pkg = load_artifacts()
    jd_vec = embed_jd(jd_text, vectorizer_pkg).astype("float32")

    id_to_row = {cid: i for i, cid in enumerate(embedding_ids)}
    rows = [id_to_row.get(cid, -1) for cid in df["candidate_id"]]
    emb_rows = np.array(
        [embeddings[r] if r >= 0 else np.zeros(embeddings.shape[1], dtype="float32") for r in rows],
        dtype="float32",
    )
    cosine_sims = (emb_rows @ jd_vec).astype("float32")

    bm25_scores = bm25_scores_for_df(bm25_pkg, df, jd_text) if bm25_pkg else None

    scores = compute_scores(df, cosine_sims, bm25_scores)
    df = df.copy()
    df["score"] = scores
    df["cosine_sim"] = cosine_sims
    if bm25_scores is not None:
        df["bm25_score"] = bm25_scores

    n = min(top_n, len(df))
    top = df.nlargest(n, "score").sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    top["rank"] = range(1, len(top) + 1)

    assessment_scores_map = load_assessment_scores()
    bm25_id_to_score = {}
    if bm25_scores is not None:
        bm25_max = bm25_scores.max() + 1e-9
        bm25_id_to_score = dict(zip(df["candidate_id"], (bm25_scores / bm25_max).tolist()))

    results = []
    for _, row in top.iterrows():
        cid = row["candidate_id"]
        row_dict = row.to_dict()
        row_dict["_assessment_scores"] = assessment_scores_map.get(cid)
        bdown = score_breakdown(row_dict, row["cosine_sim"], bm25_norm=bm25_id_to_score.get(cid, 0.0))
        reasoning = generate_reasoning(row_dict, int(row["rank"]), bdown)
        results.append({
            "rank": int(row["rank"]),
            "candidate_id": cid,
            "score": round(float(row["score"]), 4),
            "title": row.get("current_title", ""),
            "company": row.get("current_company", ""),
            "years_exp": row.get("years_of_experience", ""),
            "coverage": row.get("must_have_coverage_total", 0),
            "reasoning": reasoning,
        })
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🔎 Redrob Candidate Ranker — Sandbox Demo")
st.caption(
    "Hybrid BM25 + semantic-embedding + 23-signal candidate ranker · "
    "running the **real scoring pipeline** (`src/features.py`, `src/scoring.py`, "
    "`src/reasoning.py`) against a 125-candidate synthetic sample. "
    "The full submission runs the same code against the real 100K-candidate pool."
)

with st.expander("ℹ️ About this sandbox", expanded=False):
    st.markdown(
        "- **No LLM calls at ranking time** — pure vectorized numpy/pandas over pre-cached features.\n"
        "- **Gates first**: non-technical titles, consulting-only careers, honeypots, and closed-source-only "
        "profiles are capped/down-weighted before scoring, not zeroed out.\n"
        "- **Hybrid retrieval**: BM25 (exact keyword match) blended with TF-IDF+SVD cosine similarity.\n"
        "- **Reasoning is grounded**: every sentence is filled from real extracted profile fields — no free-typed claims.\n"
        "- This sample dataset is synthetic (see `scripts/gen_sample_data.py`) and includes intentional "
        "keyword-stuffed traps and honeypots so you can see the gates fire."
    )

col1, col2 = st.columns([3, 1])
with col1:
    jd_text = st.text_area("Job description", value=DEFAULT_JD, height=220)
with col2:
    top_n = st.slider("Candidates to show", min_value=5, max_value=100, value=10, step=5)
    st.metric("Sample pool size", "125 candidates")
    run = st.button("▶ Run ranking", type="primary", use_container_width=True)

if run:
    if not jd_text.strip():
        st.error("Please enter a job description.")
    else:
        t0 = time.time()
        with st.spinner("Scoring candidates..."):
            results_df = run_ranking(jd_text, top_n)
        elapsed = time.time() - t0

        st.success(f"Ranked 125 candidates in {elapsed*1000:.0f} ms — top {len(results_df)} shown below.")

        for _, r in results_df.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([1, 5])
                with c1:
                    st.metric(f"Rank {r['rank']}", f"{r['score']:.3f}")
                with c2:
                    st.markdown(f"**{r['candidate_id']}** — {r['title']} at {r['company']} · {r['years_exp']} yrs exp · JD coverage {r['coverage']:.1f}/5")
                    st.write(r["reasoning"])

        st.divider()
        st.subheader("Raw table")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download as CSV (submission format)",
            data=results_df.rename(columns={"rank": "rank", "score": "score"})[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False),
            file_name="sandbox_ranking.csv",
            mime="text/csv",
        )
else:
    st.info("Edit the job description (or keep the default Redrob JD) and click **Run ranking**.")
