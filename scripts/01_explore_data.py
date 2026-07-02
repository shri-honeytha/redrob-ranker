#!/usr/bin/env python3
"""
01_explore_data.py
Streams candidates.jsonl once (memory-safe — never holds all 100K parsed
records in memory at once) and computes summary statistics used to sanity
check the assumptions in docs/BUILD_SPEC.md against the FULL pool, not just
the 50-candidate sample.

Usage:
    python scripts/01_explore_data.py --candidates data/candidates.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

NON_TECH_TITLE_KEYWORDS = [
    "hr ", "human resources", "recruiter", "talent acquisition",
    "accountant", "accounting", "finance manager", "content writer",
    "graphic designer", "sales executive", "sales manager",
    "operations manager", "office manager", "marketing manager",
    "mechanical engineer", "civil engineer", "customer support",
    "administrative", "office admin",
]
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "hcl", "tech mahindra",
    "ltimindtree",
]


def stream_candidates(path):
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap rows for a quick smoke run")
    args = ap.parse_args()

    n = 0
    title_counter = Counter()
    country_counter = Counter()
    location_counter = Counter()
    industry_counter = Counter()
    skill_counter = Counter()
    edu_tier_counter = Counter()

    non_tech_title_hits = 0
    consulting_only_count = 0
    honeypot_candidates = []  # candidate_ids that look internally inconsistent
    title_desc_mismatch_examples = []
    duplicate_desc_examples = 0
    yoe_vs_history_gap = []  # (candidate_id, years_of_experience, sum_history_years)
    missing_required_field_errors = []

    skill_dur_exceeds_yoe = 0
    expert_low_duration = 0

    for rec in stream_candidates(args.candidates):
        n += 1
        if args.limit and n > args.limit:
            break

        cid = rec.get("candidate_id", f"ROW_{n}")
        profile = rec.get("profile", {})
        career = rec.get("career_history", []) or []
        skills = rec.get("skills", []) or []
        education = rec.get("education", []) or []

        title = (profile.get("current_title") or "").lower()
        title_counter[title] += 1
        country_counter[profile.get("country", "UNKNOWN")] += 1
        location_counter[profile.get("location", "UNKNOWN")] += 1
        industry_counter[profile.get("current_industry", "UNKNOWN")] += 1

        for s in skills:
            skill_counter[s.get("name", "").lower()] += 1

        for e in education:
            edu_tier_counter[e.get("tier", "unknown")] += 1

        # --- non-technical title trap check ---
        if any(kw in title for kw in NON_TECH_TITLE_KEYWORDS):
            non_tech_title_hits += 1

        # --- consulting-only check ---
        if career:
            months_at_consulting = 0
            total_months = 0
            for job in career:
                dm = job.get("duration_months", 0) or 0
                total_months += dm
                comp = (job.get("company") or "").lower()
                if any(cf in comp for cf in CONSULTING_FIRMS):
                    months_at_consulting += dm
            if total_months > 0 and months_at_consulting == total_months:
                consulting_only_count += 1

        # --- duplicated description text across jobs (quirk #2) ---
        descs = [j.get("description", "") for j in career if j.get("description")]
        if len(descs) != len(set(descs)) and len(descs) > 1:
            duplicate_desc_examples += 1

        # --- title vs description contradiction sample (quirk #3) — cheap heuristic ---
        # flag if current_title looks technical but description of the matching
        # career entry contains strong non-tech vocabulary, or vice versa.
        for job in career[:1]:  # just check most recent for speed
            desc = (job.get("description") or "").lower()
            jt = (job.get("title") or "").lower()
            if any(kw in jt for kw in ["engineer", "scientist", "developer"]) and \
               any(kw in desc for kw in ["gaap", "accounts payable", "fixed-asset", "payroll", "ind-as"]):
                if len(title_desc_mismatch_examples) < 10:
                    title_desc_mismatch_examples.append((cid, jt, desc[:160]))

        # --- years_of_experience vs sum(career_history durations) ---
        yoe = profile.get("years_of_experience")
        sum_months = sum((j.get("duration_months") or 0) for j in career)
        if yoe is not None and career:
            gap_years = abs((sum_months / 12.0) - yoe)
            if gap_years >= 3:
                yoe_vs_history_gap.append((cid, yoe, round(sum_months / 12.0, 1)))

        # --- honeypot heuristic: expert proficiency + near-zero duration ---
        expert_low_dur_count = 0
        for s in skills:
            prof = s.get("proficiency")
            dm = s.get("duration_months", None)
            if prof == "expert" and dm is not None and dm < 6:
                expert_low_dur_count += 1
            if dm is not None and yoe is not None and dm > (yoe * 12 + 24):
                skill_dur_exceeds_yoe += 1
        if expert_low_dur_count >= 5:
            expert_low_duration += 1
            if len(honeypot_candidates) < 30:
                honeypot_candidates.append((cid, expert_low_dur_count, yoe))

    print(f"Total candidates scanned: {n}\n")

    print("=== Top 20 current_title (lowercased) ===")
    for t, c in title_counter.most_common(20):
        print(f"  {c:6d}  {t}")

    print("\n=== Top 15 country ===")
    for t, c in country_counter.most_common(15):
        print(f"  {c:6d}  {t}")

    print("\n=== Top 15 current_industry ===")
    for t, c in industry_counter.most_common(15):
        print(f"  {c:6d}  {t}")

    print("\n=== education.tier distribution ===")
    for t, c in edu_tier_counter.most_common():
        print(f"  {c:6d}  {t}")

    print("\n=== Top 25 skills ===")
    for t, c in skill_counter.most_common(25):
        print(f"  {c:6d}  {t}")

    print(f"\n=== Trap / quirk confirmation against full pool ===")
    print(f"Non-technical-title hits (current_title keyword match): {non_tech_title_hits} ({non_tech_title_hits/n*100:.2f}%)")
    print(f"Pure consulting-only candidates (100% career months at named consulting firms): {consulting_only_count} ({consulting_only_count/n*100:.2f}%)")
    print(f"Candidates with duplicated description text across >=2 jobs: {duplicate_desc_examples} ({duplicate_desc_examples/n*100:.2f}%)")
    print(f"Candidates with |years_of_experience - sum(history)/12| >= 3 years: {len(yoe_vs_history_gap)} ({len(yoe_vs_history_gap)/n*100:.2f}%)")
    print(f"Candidates with >=5 'expert' skills at <6 months duration each (honeypot heuristic): {expert_low_duration} ({expert_low_duration/n*100:.3f}%)")
    print(f"Skill duration_months entries wildly exceeding plausible bound vs YOE: {skill_dur_exceeds_yoe}")

    print(f"\n--- Sample title/description mismatch candidates (quirk #3) ---")
    for cid, jt, desc in title_desc_mismatch_examples:
        print(f"  {cid}: title='{jt}' desc='{desc}...'")

    print(f"\n--- Sample honeypot-heuristic candidates (expert@low-duration) ---")
    for cid, cnt, yoe in honeypot_candidates[:15]:
        print(f"  {cid}: {cnt} expert-skills-at-<6mo, years_of_experience={yoe}")

    print(f"\n--- Sample YOE-vs-history-gap candidates ---")
    for cid, yoe, hist in yoe_vs_history_gap[:15]:
        print(f"  {cid}: stated_yoe={yoe}, sum(career_history)={hist}")


if __name__ == "__main__":
    main()
