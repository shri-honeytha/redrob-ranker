#!/usr/bin/env python3
"""
gen_sample_data.py — generates a small synthetic candidates.jsonl that
matches data/candidate_schema.json exactly, for use as the demo/sandbox
sample (the real 100K candidates.jsonl is 487MB and not shipped).

Includes a mix of: strong fits, adjacent fits, keyword-stuffed traps
(HR/Sales/Accounting with "AI" sprinkled in), consulting-only careers,
and a couple of honeypots — so the demo actually shows the gates working.
"""
import json, random, hashlib
from datetime import date

random.seed(42)

TIER1 = ["Google", "Microsoft", "Amazon", "Razorpay", "Zomato", "Swiggy", "Flipkart", "Meesho", "Cred", "Groww"]
CONSULTING = ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]
OTHER = ["Initech", "Globex", "Umbrella Softworks", "Initrode", "Soylent Corp", "Hooli"]

GOOD_DESC = [
    "Owned the retrieval and ranking layer for our search product; built a hybrid BM25 + dense embedding pipeline using FAISS and OpenSearch, deployed to production serving 2M+ queries/day. Designed the NDCG/MRR offline eval harness and ran A/B tests to validate offline-online correlation.",
    "Shipped an end-to-end recommendation system (candidate generation + learning-to-rank re-ranker, LightGBM-based) to real users. Fine-tuned sentence-transformer embeddings for our domain, managed embedding drift and periodic index refresh in a Pinecone-backed vector store.",
    "Built and maintained a production RAG system on top of Qdrant, integrating OpenAI and BGE embeddings; owned prompt design, fine-tuning vs prompting tradeoffs, and retrieval-quality regression monitoring.",
]
ADJACENT_DESC = [
    "Built backend services and data pipelines supporting search infrastructure; started contributing to relevance tuning in the last year and is ramping up on embeddings and vector search.",
    "Data engineer maintaining ETL pipelines that feed a downstream ranking model; some exposure to feature engineering for ML but not the modeling itself.",
]
TRAP_DESC = [
    "Managed end-to-end recruitment lifecycle, used AI-powered ATS tools to screen resumes and coordinated with hiring managers to close roles faster.",
    "Handled monthly book-closing, GST reconciliation, and vendor payments; used an AI-based expense categorization tool to speed up reporting.",
    "Ran outbound sales campaigns using an AI sales-enablement platform to prioritize leads; consistently exceeded quota.",
]
CONSULTING_DESC = "Delivered client engagements as part of a large consulting bench; rotated across banking and telecom accounts on staff-augmentation projects."

SKILLS_ML = ["Python", "PyTorch", "Embeddings", "Vector Databases", "Pinecone", "FAISS", "LLM Fine-tuning",
             "Learning to Rank", "NDCG Evaluation", "Sentence Transformers", "OpenSearch", "Elasticsearch"]
SKILLS_TRAP = ["Recruitment", "ATS Tools", "GST", "Excel", "Sales Enablement", "Client Communication"]

TITLES_GOOD = ["Senior AI Engineer", "Lead AI Engineer", "Senior Machine Learning Engineer", "AI Engineer",
               "Applied Scientist", "Search & Ranking Engineer", "ML Engineer - Recommendations"]
TITLES_ADJ = ["Data Engineer", "Backend Engineer", "Software Engineer II"]
TITLES_TRAP = ["HR Manager", "Senior Accountant", "Sales Manager", "Content Writer", "Graphic Designer"]

LOCS = [("Pune", "India"), ("Noida", "India"), ("Bangalore", "India"), ("Hyderabad", "India"),
        ("Mumbai", "India"), ("Toronto", "Canada"), ("Berlin", "Germany"), ("Dubai", "UAE")]


def mk_career(kind, yoe):
    n_jobs = max(1, min(4, round(yoe / 2.5)))
    hist = []
    remaining = yoe
    for i in range(n_jobs):
        dur = max(6, round((remaining / (n_jobs - i)) * 12))
        remaining -= dur / 12
        if kind == "good":
            company = random.choice(TIER1 if i == 0 else TIER1 + OTHER)
            title = random.choice(TITLES_GOOD)
            desc = random.choice(GOOD_DESC)
            industry = "Technology"
        elif kind == "adjacent":
            company = random.choice(OTHER + TIER1)
            title = random.choice(TITLES_ADJ)
            desc = random.choice(ADJACENT_DESC)
            industry = "Technology"
        elif kind == "consulting":
            company = random.choice(CONSULTING)
            title = "Consultant"
            desc = CONSULTING_DESC
            industry = "IT Services"
        else:  # trap
            company = random.choice(OTHER)
            title = random.choice(TITLES_TRAP)
            desc = random.choice(TRAP_DESC) + " Leveraged AI tools daily to boost productivity."
            industry = random.choice(["Human Resources", "Finance", "Sales", "Marketing"])
        hist.append({
            "company": company, "title": title,
            "start_date": f"{2026 - int(remaining) - dur//12}-01-01",
            "end_date": None if i == 0 else f"{2026 - int(remaining)}-01-01",
            "duration_months": int(dur), "is_current": i == 0,
            "industry": industry, "company_size": random.choice(["201-500", "1001-5000", "10001+"]),
            "description": desc,
        })
    return hist


def mk_candidate(idx, kind, honeypot=False):
    cid = f"CAND_{idx:07d}"
    yoe = round(random.uniform(2, 10), 1)
    loc, country = random.choice(LOCS)
    if kind == "good":
        title = random.choice(TITLES_GOOD)
        skills_pool = SKILLS_ML
        headline = f"{title} | Retrieval, Ranking & LLM systems"
        summary = "Applied ML engineer focused on search, ranking and retrieval systems shipped to production."
    elif kind == "adjacent":
        title = random.choice(TITLES_ADJ)
        skills_pool = SKILLS_ML[:4] + ["SQL", "Airflow"]
        headline = f"{title} ramping into ML/search"
        summary = "Backend/data engineer with growing exposure to ranking and retrieval work."
    elif kind == "consulting":
        title = "Consultant"
        skills_pool = ["Java", "SAP", "Client Delivery"]
        headline = "IT Consultant — staff augmentation"
        summary = "Consulting-only career across banking and telecom client engagements."
    else:
        title = random.choice(TITLES_TRAP)
        skills_pool = SKILLS_TRAP
        headline = f"{title} | AI-savvy professional"
        summary = "Experienced professional leveraging AI tools to work faster."

    skills = []
    for s in random.sample(skills_pool, k=min(len(skills_pool), random.randint(3, 6))):
        skills.append({
            "name": s,
            "proficiency": "expert" if (honeypot and kind != "good") else random.choice(["intermediate", "advanced", "expert"]),
            "endorsements": 0 if honeypot else random.randint(0, 40),
            "duration_months": random.randint(1, 3) if honeypot else random.randint(6, 60),
        })

    rec = {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": f"Candidate {idx}",
            "headline": headline,
            "summary": summary,
            "location": loc, "country": country,
            "years_of_experience": yoe,
            "current_title": title,
            "current_company": random.choice(TIER1 if kind == "good" else OTHER + CONSULTING),
            "current_company_size": random.choice(["201-500", "1001-5000", "10001+"]),
            "current_industry": "Technology" if kind in ("good", "adjacent") else random.choice(["Human Resources", "Finance", "IT Services"]),
        },
        "career_history": mk_career(kind, yoe),
        "education": [{
            "institution": random.choice(["IIT Bombay", "BITS Pilani", "Mahindra University", "State University", "NIT Trichy"]),
            "degree": "B.Tech", "field_of_study": "Computer Science",
            "start_year": 2026 - int(yoe) - 4, "end_year": 2026 - int(yoe),
            "grade": None, "tier": random.choice(["tier_1", "tier_2", "tier_3"]),
        }],
        "skills": skills,
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "professional"}],
        "redrob_signals": {
            "profile_completeness_score": random.uniform(60, 100),
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-25" if not honeypot else "2026-06-25",
            "open_to_work_flag": True,
            "profile_views_received_30d": random.randint(0, 50),
            "applications_submitted_30d": random.randint(0, 10),
            "recruiter_response_rate": round(random.uniform(0.1, 0.95), 2),
            "avg_response_time_hours": round(random.uniform(1, 72), 1),
            "skill_assessment_scores": {s["name"]: round(random.uniform(50, 99), 1) for s in skills[:2]} if kind == "good" else {},
            "connection_count": random.randint(50, 2000),
            "endorsements_received": random.randint(0, 200),
            "notice_period_days": random.choice([0, 15, 30, 60, 90]),
            "expected_salary_range_inr_lpa": {"min": 20, "max": 60},
            "preferred_work_mode": random.choice(["remote", "hybrid", "onsite", "flexible"]),
            "willing_to_relocate": random.choice([True, False]),
            "github_activity_score": round(random.uniform(20, 95), 1) if kind == "good" else -1,
            "search_appearance_30d": random.randint(0, 40),
            "saved_by_recruiters_30d": random.randint(0, 10),
            "interview_completion_rate": round(random.uniform(0.5, 1.0), 2),
            "offer_acceptance_rate": round(random.uniform(0.2, 1.0), 2) if random.random() > 0.3 else -1,
            "verified_email": True, "verified_phone": random.choice([True, False]),
            "linkedin_connected": True,
        },
    }
    return rec


def main():
    out = []
    idx = 1
    # 40 strong/good fits, 25 adjacent, 15 consulting-only, 40 trap/keyword-stuffed, 5 honeypots
    for _ in range(40):
        out.append(mk_candidate(idx, "good")); idx += 1
    for _ in range(25):
        out.append(mk_candidate(idx, "adjacent")); idx += 1
    for _ in range(15):
        out.append(mk_candidate(idx, "consulting")); idx += 1
    for _ in range(40):
        out.append(mk_candidate(idx, "trap")); idx += 1
    for _ in range(5):
        out.append(mk_candidate(idx, "trap", honeypot=True)); idx += 1

    random.shuffle(out)
    with open("data/candidates_sample.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(out)} synthetic candidates -> data/candidates_sample.jsonl")


if __name__ == "__main__":
    main()
