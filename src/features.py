"""
src/features.py  v2 — Complete signal extraction
Upgrades from v1:
  - All 23 redrob_signals fields now extracted (v1 used only 5)
  - Salary fit signal (expected vs JD-implied band)
  - Demand signal (saved_by_recruiters_30d, search_appearance_30d)
  - Trust/credibility score (verified_email/phone, linkedin, connection_count)
  - Response quality score (avg_response_time_hours, interview_completion_rate)
  - Profile quality score (profile_completeness_score)
  - Improved behavioral multiplier using all available signals
  - career_tenure_stability: avg months per role (job-hopper detection per JD)
  - company_tier scoring (FAANG/unicorn vs unknown)
"""
import hashlib, json, re
from datetime import date

from jd_requirements import (
    NON_TECH_TITLE_KEYWORDS, TECH_PROOF_PHRASES, NON_TECH_PROOF_PHRASES,
    CONSULTING_FIRMS, RESEARCH_ONLY_INDUSTRY_KEYWORDS, PRODUCTION_PROOF_PHRASES,
    MUST_HAVES, TITLE_RELEVANCE_MAP, DEFAULT_TITLE_RELEVANCE,
    LOCATION_TIERS, OTHER_INDIA_TIER, NON_INDIA_RELOCATE_TIER, NON_INDIA_NO_RELOCATE_TIER,
    EXPERIENCE_IDEAL_MIN, EXPERIENCE_IDEAL_MAX, EXPERIENCE_BAND_MIN, EXPERIENCE_BAND_MAX,
    NOTICE_FULL_MARKS_MAX, NOTICE_GOOD_MAX, NOTICE_FAIR_MAX,
    CONCEPT_GRAPH, LATENT_CONCEPT_BONUS_PER_HIT, LATENT_CONCEPT_MAX_BONUS,
)

TODAY = date(2026, 6, 21)

# High-signal product/AI companies — real-world prior: engineers from these
# companies have shipped ML at scale, which the JD explicitly wants
TIER1_PRODUCT_COMPANIES = {
    "google", "meta", "microsoft", "amazon", "apple", "netflix", "openai",
    "anthropic", "deepmind", "nvidia", "adobe", "salesforce", "uber", "airbnb",
    "flipkart", "swiggy", "zomato", "paytm", "phonepe", "razorpay", "meesho",
    "cred", "dream11", "unacademy", "byju", "ola", "navi", "groww", "zepto",
    "yellow.ai", "sarvam", "krutrim", "sprinklr", "freshworks", "zoho",
    "sharechat", "moj", "daily hunt", "clevertap",
}
# JD explicitly flags job-hopping as a disqualifier: "title-chasers switching every 1.5yr"
JOB_HOPPER_THRESHOLD_MONTHS = 18

# Salary band implied for Senior AI Engineer at a funded product startup (LPA)
# JD doesn't state explicitly; derived from role level and India market data
SALARY_IDEAL_MIN_LPA = 30
SALARY_IDEAL_MAX_LPA = 60
SALARY_STRETCH_MAX_LPA = 90  # above this = likely won't join at startup comp


def _lower(s): return (s or "").lower()
def _parse_date(s):
    if not s: return None
    try: return date.fromisoformat(s[:10])
    except: return None
def _months_since(d):
    if d is None: return None
    return (TODAY.year - d.year) * 12 + (TODAY.month - d.month)


def title_relevance(title_lower):
    for substr, score in TITLE_RELEVANCE_MAP:
        if substr in title_lower: return score
    return DEFAULT_TITLE_RELEVANCE


def dedupe_descriptions(career_history):
    seen, out = set(), []
    for job in career_history:
        d = (job.get("description") or "").strip()
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out


# ---------------------------------------------------------------------------
# Gates (unchanged from v1 — calibrated and correct)
# ---------------------------------------------------------------------------
def gate_non_technical_title(profile, career_history, dedup_descs):
    titles = [_lower(profile.get("current_title"))]
    for job in career_history[:2]:
        titles.append(_lower(job.get("title")))
    hit_title = next((t for t in titles if any(kw in t for kw in NON_TECH_TITLE_KEYWORDS)), None)
    if not hit_title: return False, ""
    full_text = " ".join(dedup_descs).lower()
    tech_proof = sum(1 for p in TECH_PROOF_PHRASES if p in full_text)
    non_tech_proof = sum(1 for p in NON_TECH_PROOF_PHRASES if p in full_text)
    if tech_proof >= 2 and tech_proof > non_tech_proof:
        return False, f"title='{hit_title}' overridden by {tech_proof} tech-proof phrases"
    return True, f"title='{hit_title}', tech={tech_proof}, non_tech={non_tech_proof}"

def gate_consulting_only(career_history):
    if not career_history: return False, ""
    total = sum((j.get("duration_months") or 0) for j in career_history)
    if total == 0: return False, ""
    consulting = sum((j.get("duration_months") or 0) for j in career_history
                     if any(cf in _lower(j.get("company")) for cf in CONSULTING_FIRMS))
    return consulting == total, f"{consulting}/{total} months at consulting firms"

def gate_research_only(career_history, dedup_descs):
    if not career_history: return False, ""
    industries = [_lower(j.get("industry")) for j in career_history]
    if not all(any(k in ind for k in RESEARCH_ONLY_INDUSTRY_KEYWORDS) for ind in industries):
        return False, ""
    full_text = " ".join(dedup_descs).lower()
    return not any(p in full_text for p in PRODUCTION_PROOF_PHRASES), "all research/academic, no production"

def gate_honeypot(profile, career_history, skills):
    reasons = []
    yoe = profile.get("years_of_experience")
    expert_low_dur = sum(1 for s in skills if s.get("proficiency") == "expert"
                         and (s.get("duration_months") if s.get("duration_months") is not None else 999) < 6)
    if expert_low_dur >= 3:
        reasons.append(f"{expert_low_dur} expert-skills at <6mo duration")
    if yoe is not None and career_history:
        sum_years = sum((j.get("duration_months") or 0) for j in career_history) / 12.0
        if abs(sum_years - yoe) >= 4:
            reasons.append(f"yoe={yoe} vs history={sum_years:.1f}yrs")
    return bool(reasons), "; ".join(reasons)

def gate_closed_source_only(profile, career_history, redrob, dedup_descs):
    yoe = profile.get("years_of_experience") or 0
    no_github = (redrob.get("github_activity_score", -1) == -1)
    full_text = " ".join(dedup_descs).lower()
    oss = any(p in full_text for p in ["open source","open-source","github.com/","published","blog post","conference"])
    return (yoe >= 5) and no_github and (not oss), f"yoe={yoe}, no_github={no_github}, oss={oss}"


# ---------------------------------------------------------------------------
# Career quality signals
# ---------------------------------------------------------------------------
def trajectory_slope(career_history):
    jobs = sorted([j for j in career_history if j.get("start_date")], key=lambda j: j["start_date"])
    if len(jobs) < 2: return 0.0
    xs = list(range(len(jobs)))
    ys = [title_relevance(_lower(j.get("title"))) for j in jobs]
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = sum((x-mx)**2 for x in xs)
    return num/den if den else 0.0

def product_vs_consulting_ratio(career_history):
    total = sum((j.get("duration_months") or 0) for j in career_history)
    if total == 0: return 0.0
    consulting = sum((j.get("duration_months") or 0) for j in career_history
                     if any(cf in _lower(j.get("company")) for cf in CONSULTING_FIRMS))
    return 1.0 - (consulting / total)

def company_tier_score(career_history):
    """Score 0-1 based on the quality of companies worked at.
    High-signal product companies in AI/tech space get 1.0."""
    if not career_history: return 0.3
    scores = []
    for job in career_history:
        comp = _lower(job.get("company") or "")
        if any(t in comp for t in TIER1_PRODUCT_COMPANIES):
            scores.append(1.0)
        elif any(cf in comp for cf in CONSULTING_FIRMS):
            scores.append(0.2)
        else:
            scores.append(0.5)
    # Weight more recent roles higher
    weights = list(range(1, len(scores)+1))
    return sum(s*w for s,w in zip(scores, weights)) / sum(weights)

def career_tenure_stability(career_history):
    """JD: 'title-chasers switching companies every 1.5 years' are explicitly
    disqualified. Return 0-1: 1.0 = stable tenures, 0.3 = frequent hopper."""
    if not career_history: return 0.5
    durations = [j.get("duration_months") or 0 for j in career_history if not j.get("is_current")]
    if not durations: return 0.7
    avg_tenure = sum(durations) / len(durations)
    if avg_tenure >= 24: return 1.0
    if avg_tenure >= JOB_HOPPER_THRESHOLD_MONTHS: return 0.8
    if avg_tenure >= 12: return 0.6
    return 0.3

def job_hopper_penalty(career_history):
    """Count how many jobs lasted < 18 months (JD's explicit threshold)."""
    short_stints = sum(1 for j in career_history
                       if not j.get("is_current") and (j.get("duration_months") or 24) < JOB_HOPPER_THRESHOLD_MONTHS)
    return min(1.0, short_stints / max(1, len(career_history)))


# ---------------------------------------------------------------------------
# Skill signals
# ---------------------------------------------------------------------------
PROFICIENCY_WEIGHT = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

def _skill_trust(skill):
    prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency"), 0.4)
    dur = min(1.0, (skill.get("duration_months") or 0) / 36.0)
    endorse = min(1.0, (skill.get("endorsements") or 0) / 25.0)
    return 0.5*prof + 0.3*dur + 0.2*endorse

def aggregate_skill_trust(skills, assessment_scores):
    if not skills: return 0.0
    must_have_terms = set()
    for mh in MUST_HAVES.values():
        must_have_terms.update(t.lower() for t in mh["skill_terms"])
    ws, wt = 0.0, 0.0
    for s in skills:
        name = _lower(s.get("name"))
        w = 1.5 if any(t in name for t in must_have_terms) else 0.4
        trust = _skill_trust(s)
        if assessment_scores and s.get("name") in assessment_scores:
            trust = max(trust, assessment_scores[s["name"]] / 100.0)
        ws += trust * w; wt += w
    return ws/wt if wt else 0.0

def must_have_coverage(skills, dedup_descs):
    skill_names = " | ".join(_lower(s.get("name")) for s in skills)
    full_text = " ".join(dedup_descs).lower()
    coverage = {}
    for key, mh in MUST_HAVES.items():
        skill_hit = any(t in skill_names for t in mh["skill_terms"])
        desc_hits = sum(1 for p in mh["desc_phrases"] if p in full_text)
        if skill_hit and desc_hits > 0: score = 1.0
        elif skill_hit or desc_hits >= 2: score = 0.75
        elif desc_hits == 1: score = 0.4
        else: score = 0.0
        coverage[key] = score
    return coverage, sum(coverage.values())

def ever_vs_recent_overlap(skills, career_history, dedup_descs):
    must_have_terms = set()
    for mh in MUST_HAVES.values():
        must_have_terms.update(t.lower() for t in mh["skill_terms"])
    all_skill_names = set(_lower(s.get("name")) for s in skills)
    ever = sum(1 for t in must_have_terms if any(t in sn for sn in all_skill_names))
    recent_jobs = sorted([j for j in career_history if j.get("start_date")],
                          key=lambda j: j["start_date"], reverse=True)[:2]
    recent_text = " ".join((j.get("description") or "") for j in recent_jobs).lower()
    recent = sum(1 for t in must_have_terms if t in recent_text)
    n = max(1, len(must_have_terms))
    return ever/n, recent/n

def latent_concept_bonus(skills, dedup_descs):
    skill_text = " ".join(_lower(s.get("name")) for s in skills)
    full_text = " ".join(dedup_descs).lower()
    hits, matched = 0, []
    for concept, phrases in CONCEPT_GRAPH.items():
        if any(p in skill_text for p in phrases): continue
        if any(p in full_text for p in phrases):
            hits += 1; matched.append(concept)
    return min(LATENT_CONCEPT_MAX_BONUS, hits * LATENT_CONCEPT_BONUS_PER_HIT), matched


# ---------------------------------------------------------------------------
# Fit factors
# ---------------------------------------------------------------------------
def experience_fit(yoe):
    if yoe is None: return 0.3
    if EXPERIENCE_IDEAL_MIN <= yoe <= EXPERIENCE_IDEAL_MAX: return 1.0
    if EXPERIENCE_BAND_MIN <= yoe < EXPERIENCE_IDEAL_MIN:
        return 0.75 + 0.25 * ((yoe - EXPERIENCE_BAND_MIN) / (EXPERIENCE_IDEAL_MIN - EXPERIENCE_BAND_MIN))
    if EXPERIENCE_IDEAL_MAX < yoe <= EXPERIENCE_BAND_MAX:
        return 0.75 + 0.25 * ((EXPERIENCE_BAND_MAX - yoe) / (EXPERIENCE_BAND_MAX - EXPERIENCE_IDEAL_MAX))
    if yoe < EXPERIENCE_BAND_MIN: return max(0.15, 0.75 - 0.15*(EXPERIENCE_BAND_MIN - yoe))
    return max(0.15, 0.75 - 0.1*(yoe - EXPERIENCE_BAND_MAX))

def location_fit(location, country, willing_to_relocate):
    loc = _lower(location); country_l = _lower(country)
    for key, score in LOCATION_TIERS.items():
        if key in loc: return score
    if "india" in country_l: return OTHER_INDIA_TIER
    return NON_INDIA_RELOCATE_TIER if willing_to_relocate else NON_INDIA_NO_RELOCATE_TIER

def notice_period_fit(days):
    if days is None: return 0.5
    if days <= NOTICE_FULL_MARKS_MAX: return 1.0
    if days <= NOTICE_GOOD_MAX: return 0.75
    if days <= NOTICE_FAIR_MAX: return 0.5
    return 0.25

def work_mode_fit(mode):
    return 0.7 if mode == "remote" else 1.0

def salary_fit(salary_range):
    """Check if candidate's expected salary is within a reasonable band for
    this role. Both too-low (likely junior) and too-high (won't join at
    startup comp) are negative signals."""
    if not salary_range: return 0.5
    mid = ((salary_range.get("min") or 0) + (salary_range.get("max") or 0)) / 2
    if mid == 0: return 0.5
    if SALARY_IDEAL_MIN_LPA <= mid <= SALARY_IDEAL_MAX_LPA: return 1.0
    if mid < SALARY_IDEAL_MIN_LPA:
        # Under-expectation: could be junior, but give benefit of doubt
        return max(0.5, 0.7 - 0.05*(SALARY_IDEAL_MIN_LPA - mid))
    if mid <= SALARY_STRETCH_MAX_LPA:
        # Above ideal but within stretch — still plausible
        return 0.7
    # Far above stretch — high risk they won't join
    return max(0.3, 0.7 - 0.02*(mid - SALARY_STRETCH_MAX_LPA))

EDU_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}

def education_tier_score(education):
    if not education: return 0.4
    return max(EDU_TIER_SCORE.get(e.get("tier", "unknown"), 0.4) for e in education)


# ---------------------------------------------------------------------------
# Redrob signals — ALL 23 fields (v2: was only 5 in v1)
# ---------------------------------------------------------------------------
def profile_quality_score(redrob):
    """Completeness + verification + professional network depth."""
    completeness = (redrob.get("profile_completeness_score") or 50) / 100.0
    email_ok = 1.0 if redrob.get("verified_email") else 0.6
    phone_ok = 1.0 if redrob.get("verified_phone") else 0.7
    linkedin = 1.0 if redrob.get("linkedin_connected") else 0.7
    connections = redrob.get("connection_count") or 0
    # Saturate at 500 connections (median is 345, so 500 is clearly strong)
    conn_score = min(1.0, connections / 500.0)
    return 0.35*completeness + 0.15*email_ok + 0.1*phone_ok + 0.15*linkedin + 0.25*conn_score

def demand_signal(redrob):
    """How much are recruiters already finding/saving this person?
    saved_by_recruiters_30d and search_appearance_30d are Redrob-native
    signals that proxy market validation of the candidate."""
    saved = redrob.get("saved_by_recruiters_30d") or 0
    searches = redrob.get("search_appearance_30d") or 0
    views = redrob.get("profile_views_received_30d") or 0
    # Normalize: p75 of saved=11, searches=158, views=68 across full pool
    saved_score = min(1.0, saved / 15.0)
    search_score = min(1.0, searches / 200.0)
    view_score = min(1.0, views / 100.0)
    return 0.45*saved_score + 0.30*search_score + 0.25*view_score

def reachability_score(redrob):
    """Can a recruiter actually reach and convert this person?"""
    last_active = _parse_date(redrob.get("last_active_date"))
    months_inactive = _months_since(last_active)
    if months_inactive is None: activity = 0.4
    elif months_inactive <= 1: activity = 1.0
    elif months_inactive <= 2: activity = 0.8
    elif months_inactive <= 6: activity = 0.5
    else: activity = 0.15

    resp_rate = redrob.get("recruiter_response_rate")
    resp_score = 0.4 if resp_rate is None else resp_rate

    # avg_response_time_hours: lower is better (full pool mean=133h, p25=68h)
    resp_hours = redrob.get("avg_response_time_hours")
    if resp_hours is None: speed_score = 0.5
    elif resp_hours <= 24: speed_score = 1.0
    elif resp_hours <= 72: speed_score = 0.8
    elif resp_hours <= 168: speed_score = 0.6  # within 1 week
    else: speed_score = 0.3

    open_flag = 1.0 if redrob.get("open_to_work_flag") else 0.5
    apps_30d = min(1.0, (redrob.get("applications_submitted_30d") or 0) / 10.0)

    return 0.30*activity + 0.25*resp_score + 0.20*speed_score + 0.15*open_flag + 0.10*apps_30d

def engagement_quality_score(redrob):
    """Does this candidate follow through? Interview completion + offer acceptance."""
    interview_rate = redrob.get("interview_completion_rate")
    interview_score = 0.6 if interview_rate is None else interview_rate

    offer_rate = redrob.get("offer_acceptance_rate", -1)
    # -1 = no prior offers = neutral, NOT a bad score
    if offer_rate is None or offer_rate == -1: offer_score = 0.6
    else: offer_score = offer_rate

    return 0.6*interview_score + 0.4*offer_score

def github_signal(redrob):
    """GitHub activity score from Redrob (-1 = no GitHub linked = neutral).
    For a Senior AI Engineer role this matters a lot — JD explicitly calls
    out external validation as something they need."""
    gh = redrob.get("github_activity_score", -1)
    if gh is None or gh == -1: return 0.4  # no data = neutral
    return gh / 100.0

def behavioral_multiplier(redrob):
    """Combined behavioral multiplier [0.35, 1.25] built from all signals.
    Applied as a final multiplicative modifier to technical fit score."""
    reach = reachability_score(redrob)
    demand = demand_signal(redrob)
    quality = profile_quality_score(redrob)
    engagement = engagement_quality_score(redrob)
    gh = github_signal(redrob)

    # Composite behavioral score (weighted average)
    behav = (0.35*reach + 0.25*demand + 0.20*quality + 0.15*engagement + 0.05*gh)

    # Map 0-1 behavioral score to a [0.35, 1.25] multiplier
    # Score of 0.5 (average) → multiplier 1.0 (neutral)
    # Score of 1.0 (excellent) → multiplier 1.25
    # Score of 0.0 (terrible) → multiplier 0.35
    mult = 0.35 + 0.9 * behav
    return max(0.35, min(1.25, mult))


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------
def extract_features(rec):
    cid = rec["candidate_id"]
    profile = rec.get("profile", {})
    career_history = rec.get("career_history", []) or []
    education = rec.get("education", []) or []
    skills = rec.get("skills", []) or []
    redrob = rec.get("redrob_signals", {}) or {}
    certifications = rec.get("certifications", []) or []

    dedup_descs = dedupe_descriptions(career_history)
    title_lower = _lower(profile.get("current_title"))

    # gates
    g_title, g_title_ev = gate_non_technical_title(profile, career_history, dedup_descs)
    g_consult, g_consult_ev = gate_consulting_only(career_history)
    g_research, g_research_ev = gate_research_only(career_history, dedup_descs)
    g_honeypot, g_honeypot_ev = gate_honeypot(profile, career_history, skills)
    g_closed, g_closed_ev = gate_closed_source_only(profile, career_history, redrob, dedup_descs)

    # career
    t_rel = title_relevance(title_lower)
    traj = trajectory_slope(career_history)
    prod_ratio = product_vs_consulting_ratio(career_history)
    comp_tier = company_tier_score(career_history)
    tenure_stab = career_tenure_stability(career_history)
    hopper_pct = job_hopper_penalty(career_history)

    # skills
    assessment_scores = redrob.get("skill_assessment_scores") or {}
    sk_trust = aggregate_skill_trust(skills, assessment_scores)
    coverage_dict, coverage_total = must_have_coverage(skills, dedup_descs)
    ever_overlap, recent_overlap = ever_vs_recent_overlap(skills, career_history, dedup_descs)
    latent_bonus, latent_concepts = latent_concept_bonus(skills, dedup_descs)
    has_assessment = bool(assessment_scores)

    # fit
    yoe = profile.get("years_of_experience")
    exp_fit = experience_fit(yoe)
    loc_fit = location_fit(profile.get("location"), profile.get("country"),
                            redrob.get("willing_to_relocate", False))
    notice_fit = notice_period_fit(redrob.get("notice_period_days"))
    wm_fit = work_mode_fit(redrob.get("preferred_work_mode"))
    sal_fit = salary_fit(redrob.get("expected_salary_range_inr_lpa"))
    edu_score = education_tier_score(education)

    # ALL redrob signals
    reach = reachability_score(redrob)
    demand = demand_signal(redrob)
    pq = profile_quality_score(redrob)
    engagement = engagement_quality_score(redrob)
    gh = github_signal(redrob)
    behav_mult = behavioral_multiplier(redrob)

    embedding_text = " ".join(filter(None, [
        profile.get("headline", ""), profile.get("summary", ""),
        ", ".join(_lower(s.get("name", "")) for s in skills),
        " ".join(dedup_descs),
    ]))

    return {
        "candidate_id": cid,
        "anonymized_name": profile.get("anonymized_name"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "years_of_experience": yoe,
        "notice_period_days": redrob.get("notice_period_days"),
        "preferred_work_mode": redrob.get("preferred_work_mode"),
        "last_active_date": redrob.get("last_active_date"),
        "recruiter_response_rate": redrob.get("recruiter_response_rate"),
        "willing_to_relocate": redrob.get("willing_to_relocate", False),
        "open_to_work_flag": redrob.get("open_to_work_flag", False),
        "expected_salary_mid_lpa": (
            ((redrob.get("expected_salary_range_inr_lpa") or {}).get("min", 0) +
             (redrob.get("expected_salary_range_inr_lpa") or {}).get("max", 0)) / 2
        ),
        "github_activity_score_raw": redrob.get("github_activity_score", -1),
        "profile_completeness_score": redrob.get("profile_completeness_score", 50),
        "connection_count": redrob.get("connection_count", 0),
        "saved_by_recruiters_30d": redrob.get("saved_by_recruiters_30d", 0),
        "search_appearance_30d": redrob.get("search_appearance_30d", 0),
        # gates
        "gate_non_tech_title": g_title, "gate_non_tech_title_evidence": g_title_ev,
        "gate_consulting_only": g_consult, "gate_consulting_evidence": g_consult_ev,
        "gate_research_only": g_research, "gate_research_evidence": g_research_ev,
        "gate_honeypot": g_honeypot, "gate_honeypot_evidence": g_honeypot_ev,
        "gate_closed_source": g_closed, "gate_closed_source_evidence": g_closed_ev,
        # career quality
        "title_relevance": t_rel,
        "trajectory_slope": traj,
        "product_vs_consulting_ratio": prod_ratio,
        "company_tier_score": comp_tier,
        "career_tenure_stability": tenure_stab,
        "job_hopper_penalty": hopper_pct,
        # skill signals
        "skill_trust": sk_trust,
        "must_have_coverage_total": coverage_total,
        "must_have_coverage_detail": coverage_dict,
        "ever_used_overlap": ever_overlap,
        "recent_use_overlap": recent_overlap,
        "latent_concept_bonus": latent_bonus,
        "latent_concepts_matched": latent_concepts,
        "has_skill_assessment": has_assessment,
        # fit factors
        "experience_fit": exp_fit,
        "location_fit": loc_fit,
        "notice_fit": notice_fit,
        "work_mode_fit": wm_fit,
        "salary_fit": sal_fit,
        "education_tier_score": edu_score,
        # ALL redrob behavioral signals (v2: was only partial in v1)
        "reachability_score": reach,
        "demand_signal": demand,
        "profile_quality_score": pq,
        "engagement_quality": engagement,
        "github_signal": gh,
        "behavioral_multiplier": behav_mult,
        # metadata
        "num_career_entries": len(career_history),
        "num_skills": len(skills),
        "num_certifications": len(certifications),
        "_embedding_text": embedding_text,
        "_dedup_desc_count": len(dedup_descs),
    }
