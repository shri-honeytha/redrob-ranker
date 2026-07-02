"""
jd_requirements.py
A single structured, hand-curated representation of the Senior AI Engineer —
Founding Team JD. Every gate, scoring component, and reasoning string in this
project reads from this file instead of hardcoding JD facts inline, so the
whole system stays defensible and easy to re-tune in one place.

Source: docs/job_description.docx (full text reviewed and extracted; see
docs/exploration_output.txt and the original BUILD_SPEC.md for the line-by-line
JD reasoning that produced these structures).
"""

# ---------------------------------------------------------------------------
# 1. Non-technical title trap list
# ---------------------------------------------------------------------------
# Confirmed against the full 100K pool (see docs/exploration_output.txt):
# these 12 buckets alone account for ~65,700 of the 100,000 candidates
# (business analyst, hr manager, mechanical engineer, accountant, project
# manager, customer support, operations manager, content writer, sales
# executive, civil engineer, graphic designer, marketing manager).
NON_TECH_TITLE_KEYWORDS = [
    "hr manager", "human resources", "recruiter", "talent acquisition",
    "accountant", "accounting", "finance manager", "content writer",
    "graphic designer", "sales executive", "sales manager",
    "operations manager", "office manager", "marketing manager",
    "mechanical engineer", "civil engineer", "customer support",
    "administrative", "office admin", "business analyst", "project manager",
]

# Vocabulary that, if found in a job's description text, proves the role was
# genuinely technical/ML-adjacent even if the title string looks generic
# (e.g. "Engineer" titles that are secretly accounting roles, and vice versa).
TECH_PROOF_PHRASES = [
    "machine learning", "embeddings", "embedding", "retrieval", "ranking",
    "vector database", "vector search", "recommendation system",
    "recommender system", "nlp", "natural language processing",
    "llm", "large language model", "fine-tun", "rag ", "retrieval-augmented",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "opensearch", "ndcg", "mrr", "map@", "a/b test", "ab test",
    "search relevance", "semantic search", "bert", "transformer",
    "pytorch", "tensorflow", "scikit-learn", "production ml",
    "data pipeline", "spark", "kafka", "airflow", "model serving",
    "feature store", "mlops",
]

NON_TECH_PROOF_PHRASES = [
    "gaap", "ind-as", "accounts payable", "fixed-asset", "fixed asset",
    "payroll", "month-end close", "statutory compliance", "tax filing",
    "general ledger", "p&l", "invoice processing", "recruitment cycle",
    "candidate sourcing", "onboarding new hires", "employee relations",
    "social media calendar", "ad copy", "blog post", "seo content",
    "photoshop", "illustrator", "figma mockup", "brand guidelines",
    "cold calling", "sales pipeline", "quota attainment", "crm records",
    "dfm", "dfma", "solidworks", "ansys", "cad design", "manufacturing line",
]

# ---------------------------------------------------------------------------
# 2. Consulting-only firms (services companies the JD explicitly warns about)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "hcl", "tech mahindra",
    "ltimindtree", "ltimindtee", "l&t infotech",
]

# ---------------------------------------------------------------------------
# 3. Pure-research-only disqualifier
# ---------------------------------------------------------------------------
RESEARCH_ONLY_INDUSTRY_KEYWORDS = ["research", "academia", "academic"]
PRODUCTION_PROOF_PHRASES = [
    "production", "deployed", "real users", "shipped", "scale", "live system",
    "serving traffic", "in prod", "rolled out", "launched",
]

# ---------------------------------------------------------------------------
# 4. The five JD "must-haves" (Section 5.4 of BUILD_SPEC.md)
# ---------------------------------------------------------------------------
MUST_HAVES = {
    "embedding_retrieval": {
        "label": "Production embedding/retrieval experience",
        "skill_terms": ["embeddings", "sentence-transformers", "bge", "e5",
                         "openai embeddings", "dense retrieval", "retrieval"],
        "desc_phrases": ["embedding", "retrieval", "embedding drift",
                          "index refresh", "dense vector", "semantic search"],
    },
    "vector_db": {
        "label": "Vector database / hybrid search infra",
        "skill_terms": ["pinecone", "weaviate", "qdrant", "milvus",
                         "opensearch", "elasticsearch", "faiss", "vector database"],
        "desc_phrases": ["pinecone", "weaviate", "qdrant", "milvus", "faiss",
                          "opensearch", "elasticsearch", "vector database",
                          "hybrid search", "ann index", "approximate nearest neighbour",
                          "approximate nearest neighbor"],
    },
    "ranking_eval": {
        "label": "Ranking evaluation rigor",
        "skill_terms": ["ndcg", "mrr", "map", "a/b testing", "learning to rank",
                         "learning-to-rank", "ltr"],
        "desc_phrases": ["ndcg", "mrr", "mean average precision", "offline-to-online",
                          "offline to online", "a/b test", "ab test", "learning-to-rank",
                          "learning to rank", "click-through", "ctr "],
    },
    "production_python": {
        "label": "Strong Python in production",
        "skill_terms": ["python"],
        "desc_phrases": ["production python", "python service", "python pipeline",
                          "deployed in python", "python codebase", "wrote most of",
                          "owned the", "on-call"],
    },
    "shipped_ranking_system": {
        "label": "Shipped an end-to-end ranking/search/recsys system",
        "skill_terms": ["recommendation system", "recommender system", "search engine",
                         "ranking system"],
        "desc_phrases": ["recommendation system", "recommender system",
                          "search system", "ranking system", "search relevance",
                          "candidate matching", "feed ranking", "personalization"],
    },
}

# ---------------------------------------------------------------------------
# 5. Title relevance map (continuous 0-1, used post-gate as a feature)
# ---------------------------------------------------------------------------
TITLE_RELEVANCE_MAP = [
    # (substring match on lowercased title, relevance score) — first match wins,
    # ordered most-specific first.
    ("machine learning engineer", 1.0),
    ("ml engineer", 1.0),
    ("ai engineer", 1.0),
    ("research scientist", 0.85),
    ("recommendation", 0.95),
    ("recommender", 0.95),
    ("search engineer", 0.95),
    ("search relevance", 0.95),
    ("ranking engineer", 0.95),
    ("nlp engineer", 0.9),
    ("applied scientist", 0.9),
    ("data scientist", 0.75),
    ("mlops", 0.8),
    ("backend engineer", 0.55),
    ("data engineer", 0.55),
    ("software engineer", 0.5),
    ("full stack", 0.45),
    ("platform engineer", 0.5),
    ("devops", 0.35),
    ("cloud engineer", 0.35),
    ("frontend", 0.2),
    ("mobile", 0.2),
    ("qa", 0.15),
    ("test engineer", 0.15),
]
DEFAULT_TITLE_RELEVANCE = 0.1  # anything unmatched and not gated falls here

# ---------------------------------------------------------------------------
# 6. Location tiers (confirmed against JD text directly)
# ---------------------------------------------------------------------------
LOCATION_TIERS = {
    "pune": 1.0, "noida": 1.0,
    "hyderabad": 0.8, "mumbai": 0.8, "bangalore": 0.8, "bengaluru": 0.8,
    "delhi": 0.8, "gurgaon": 0.8, "gurugram": 0.8, "new delhi": 0.8, "ncr": 0.8,
}
OTHER_INDIA_TIER = 0.55
NON_INDIA_RELOCATE_TIER = 0.35
NON_INDIA_NO_RELOCATE_TIER = 0.1

# ---------------------------------------------------------------------------
# 7. Experience fit curve (years_of_experience)
# ---------------------------------------------------------------------------
EXPERIENCE_IDEAL_MIN = 6
EXPERIENCE_IDEAL_MAX = 8
EXPERIENCE_BAND_MIN = 5    # JD's stated band
EXPERIENCE_BAND_MAX = 9

# ---------------------------------------------------------------------------
# 8. Notice period thresholds (days)
# ---------------------------------------------------------------------------
NOTICE_FULL_MARKS_MAX = 30
NOTICE_GOOD_MAX = 60
NOTICE_FAIR_MAX = 90

# ---------------------------------------------------------------------------
# 9. Concept graph for latent skill inference (Section 5.5)
# ---------------------------------------------------------------------------
CONCEPT_GRAPH = {
    "learning_to_rank": ["learning to rank", "learning-to-rank", "ltr model",
                          "pointwise ranking", "pairwise ranking", "listwise ranking"],
    "ann_search": ["approximate nearest neighbour", "approximate nearest neighbor",
                   "ann index", "hnsw", "ivf index"],
    "ab_testing": ["a/b test", "ab test", "online experiment", "experimentation platform"],
    "embedding_drift": ["embedding drift", "index refresh", "retrieval-quality regression",
                        "retrieval quality regression"],
    "rag": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "feature_engineering": ["feature engineering", "feature store", "feature pipeline"],
    "model_serving": ["model serving", "inference service", "online inference",
                      "low-latency serving"],
    "eval_infra": ["offline benchmark", "offline evaluation", "eval harness",
                   "evaluation framework", "golden set", "regression suite"],
}

# Confidence weight applied when a concept is found ONLY in description text
# and NOT already present in the candidate's structured skills list (so we
# never double count what skill-trust already rewarded).
LATENT_CONCEPT_BONUS_PER_HIT = 0.06
LATENT_CONCEPT_MAX_BONUS = 0.30

JD_TITLE_ANCHOR_TEXT = (
    "Senior AI Engineer Founding Team Redrob AI Series A talent intelligence "
    "platform. Own the intelligence layer: ranking, retrieval, and matching "
    "systems that decide what recruiters see when they search for candidates. "
    "Deep technical depth in modern ML systems: embeddings, retrieval, ranking, "
    "LLMs, fine-tuning. Production experience with embeddings-based retrieval "
    "systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to "
    "real users, handling embedding drift, index refresh, retrieval-quality "
    "regression in production. Production experience with vector databases or "
    "hybrid search infrastructure: Pinecone, Weaviate, Qdrant, Milvus, "
    "OpenSearch, Elasticsearch, FAISS. Strong Python, production code quality. "
    "Hands-on experience designing evaluation frameworks for ranking systems: "
    "NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. "
    "Ideal candidate: 6-8 years total experience, 4-5 years in applied ML/AI "
    "roles at product companies, has shipped at least one end-to-end ranking, "
    "search, or recommendation system to real users at meaningful scale, has "
    "strong opinions about retrieval (hybrid vs dense), evaluation (offline vs "
    "online), and LLM integration (when to fine-tune vs prompt)."
)
