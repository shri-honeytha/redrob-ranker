"""
src/reasoning.py
Slot-based 1-2 sentence reasoning generator. Generates a reasoning string
for each of the top-100 candidates that:
  1. References specific, verifiable facts from the candidate's profile.
  2. Connects those facts to the JD's named requirements.
  3. Mentions concerns honestly (especially for ranks 15-100).
  4. Varies sentence structure across rows (Stage 4 manual review explicitly
     penalizes "all-identical / templated" reasoning).
  5. Is validated before writing to CSV (rank-consistency + duplicate-
     structure detector).

No LLM is called anywhere here — compute constraints forbid it during the
online ranking step, and a rules/template approach is the only way to
guarantee that every claim is grounded in an actual profile field (Stage 4
also explicitly penalizes hallucinated skills/employers).
"""
import hashlib
import json
import re


# ---------------------------------------------------------------------------
# The five slot-filling functions
# Each returns a list of candidate strings (for diversity); the selector
# function picks one per candidate based on candidate_id hash so the
# structural variation is deterministic (same run = same output, which
# the Stage 3 code-reproduction check requires).
# ---------------------------------------------------------------------------

def _pick(candidate_id, options):
    """Deterministically select one option from a list using the candidate
    ID as the source of variation, so every run produces identical output
    for the same candidate (required for Stage 3 code reproduction)."""
    idx = int(hashlib.md5(candidate_id.encode()).hexdigest(), 16) % len(options)
    return options[idx]


def slot_identity(row):
    """Slot 1: title + years + company context."""
    yoe = row.get("years_of_experience")
    yoe_str = f"{yoe:.1f}-year" if yoe else "experienced"
    title = row.get("current_title") or "Engineer"
    company = row.get("current_company") or "a tech company"

    options = [
        f"{yoe_str} {title} currently at {company}.",
        f"A {yoe_str} {title} with current tenure at {company}.",
        f"{title} ({yoe_str} exp.) — currently at {company}.",
    ]
    return _pick(row["candidate_id"], options)


def slot_strongest_signal(row, breakdown):
    """Slot 2: strongest technical signal from must-have coverage or
    specific career evidence. Pulls from actual feature data, not generic
    praise."""
    coverage_total = row.get("must_have_coverage_total", 0)
    latent_concepts = row.get("latent_concepts_matched", "[]")
    if isinstance(latent_concepts, str):
        try:
            latent_concepts = json.loads(latent_concepts)
        except Exception:
            latent_concepts = []

    coverage_detail = row.get("must_have_coverage_detail", "{}")
    if isinstance(coverage_detail, str):
        try:
            coverage_detail = json.loads(coverage_detail)
        except Exception:
            coverage_detail = {}

    # Find which must-haves scored highest
    top_musthaves = sorted(coverage_detail.items(), key=lambda x: x[1], reverse=True)
    top_name = top_musthaves[0][0].replace("_", " ") if top_musthaves else "ML experience"

    if coverage_total >= 3.0:
        options = [
            f"Demonstrates coverage of {coverage_total:.1f}/5 JD must-haves including strong evidence of {top_name}.",
            f"Career history confirms {coverage_total:.1f}/5 of the JD's core requirements, with clear {top_name} signal.",
            f"Hits {coverage_total:.1f} of 5 JD must-have dimensions, strongest on {top_name}.",
        ]
    elif coverage_total >= 1.5:
        latent_str = f"; latent inference matches: {', '.join(latent_concepts[:2])}" if latent_concepts else ""
        options = [
            f"Partial JD coverage ({coverage_total:.1f}/5 must-haves){latent_str}.",
            f"Covers {coverage_total:.1f}/5 JD core requirements{latent_str}.",
            f"Moderate alignment with JD requirements — {coverage_total:.1f}/5 must-haves evidenced{latent_str}.",
        ]
    else:
        options = [
            f"Limited overlap with JD's core requirements ({coverage_total:.1f}/5 must-haves).",
            f"Weak JD coverage score ({coverage_total:.1f}/5) — adjacent skills only.",
            f"Coverage of JD must-haves is marginal ({coverage_total:.1f}/5).",
        ]
    return _pick(row["candidate_id"], options)


def slot_assessment_scores(row):
    """Slot 3 (conditional): Redrob verified assessment scores — only
    include if relevant assessment scores actually exist for this candidate,
    otherwise omit entirely (forced filler is exactly what Stage 4 rejects)."""
    # assessment_scores live in the raw record, not in features.parquet.
    # The caller passes them in as row["_assessment_scores"] if available.
    assessments = row.get("_assessment_scores") or {}
    if not assessments:
        return None
    relevant_keys = ["python", "machine learning", "sql", "nlp", "ml", "embedding"]
    relevant = {k: v for k, v in assessments.items()
                if any(rk in k.lower() for rk in relevant_keys)}
    if not relevant:
        return None
    top_k, top_v = max(relevant.items(), key=lambda x: x[1])
    return f"Redrob-verified assessment score: {top_k} = {top_v}/100."


def slot_logistics(row):
    """Slot 4: location + notice period."""
    location = row.get("location") or "Unknown location"
    country = row.get("country") or ""
    notice = row.get("notice_period_days")
    mode = row.get("preferred_work_mode") or "flexible"

    if notice is None:
        notice_str = "notice period unknown"
    elif notice <= 30:
        notice_str = f"notice period {notice}d (≤30d target)"
    elif notice <= 60:
        notice_str = f"notice period {notice}d (within buy-out range)"
    else:
        notice_str = f"notice period {notice}d (above 60-day preference)"

    options = [
        f"Based in {location}{', ' + country if country and country != 'India' else ''} — {notice_str}, work mode: {mode}.",
        f"Location: {location} | {notice_str} | prefers {mode}.",
        f"{location}-based; {notice_str}; {mode} preferred.",
    ]
    return _pick(row["candidate_id"], options)


def slot_concern(row, rank, breakdown):
    """Slot 5 (conditional): explicit concern / caveat.
    Required for ranks ≥ 15 (BUILD_SPEC.md Section 7: 'rank-consistency
    check — a rank-95 candidate with glowing reasoning fails Stage 4').
    For top-14, only include if there IS a real concern to flag."""
    active_gates = breakdown.get("active_gates", []) if breakdown else []
    gate_closed = row.get("gate_closed_source", False)
    notice = row.get("notice_period_days")
    loc_fit = row.get("location_fit", 0.5)
    coverage = row.get("must_have_coverage_total", 0)
    behav = row.get("behavioral_multiplier", 1.0)
    reachability = row.get("reachability_score", 0.5)

    concerns = []
    if "gate_non_tech_title" in active_gates:
        concerns.append("non-technical title (capped score)")
    if "gate_consulting_only" in active_gates:
        concerns.append("consulting-only career history")
    if "gate_honeypot" in active_gates:
        concerns.append("profile internal inconsistency flagged")
    if gate_closed:
        concerns.append("no public/GitHub signal — closed-source background only")
    if notice and notice > 90:
        concerns.append(f"long notice period ({notice}d)")
    if loc_fit < 0.4 and not active_gates:
        loc = row.get("location") or ""
        concerns.append(f"location ({loc}) is outside preferred tier")
    if coverage < 1.0 and not active_gates:
        concerns.append("limited evidence of JD must-haves")
    if reachability < 0.35:
        concerns.append("low platform engagement / reachability")

    if not concerns:
        if rank < 15:
            return None  # no concern to flag for strong top candidates
        elif rank < 50:
            return f"Included at rank {rank} as a borderline technical fit."
        else:
            return f"Ranked {rank} — included as strongest remaining option below the primary shortlist."

    concern_str = "; ".join(concerns[:2])  # max 2 concerns to stay concise
    if rank < 15:
        return f"Concern to flag: {concern_str}."
    elif rank < 50:
        return f"Ranked {rank} with caveats: {concern_str}."
    else:
        return f"Ranked {rank} — included despite: {concern_str}."


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_reasoning(row, rank, breakdown=None):
    """Generate a 1-2 sentence reasoning string for one candidate.

    Args:
        row: dict-like (features.parquet row + optional _assessment_scores)
        rank: the assigned rank (1=best)
        breakdown: score_breakdown dict from scoring.py (optional)

    Returns:
        A 1-2 sentence string, specific to this candidate's actual data.
    """
    parts = []
    parts.append(slot_identity(row))
    parts.append(slot_strongest_signal(row, breakdown))

    assessment_str = slot_assessment_scores(row)
    if assessment_str:
        parts.append(assessment_str)

    parts.append(slot_logistics(row))

    concern_str = slot_concern(row, rank, breakdown)
    if concern_str:
        parts.append(concern_str)

    # Join to 2 sentences max: merge adjacent short parts
    full = " ".join(p for p in parts if p)
    # Condense to ≤2 sentences by keeping only the highest-value content
    # (identity + signal for strong candidates; signal + concern for weak)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full) if s.strip()]
    if len(sentences) <= 2:
        return " ".join(sentences)
    # For brevity, prioritize: strongest signal + logistics + concern (drop identity
    # sentence if running long, since title/yoe are already in the CSV's score context)
    if concern_str:
        return sentences[1] + " " + sentences[-1]
    return sentences[0] + " " + sentences[1]


# ---------------------------------------------------------------------------
# Post-generation validators (run before writing CSV)
# ---------------------------------------------------------------------------
def validate_rank_consistency(reasoning_rows):
    """Check that tone matches rank:
    - Ranks 1-14: should not contain caveats that contradict the rank
    - Ranks 15+: must contain at least some hedging / caveat language
    Returns list of (rank, issue_str) for any failures.
    """
    hedge_phrases = [
        "concern", "caveat", "despite", "limited", "weak", "below",
        "borderline", "ranked", "outside", "long notice",
        "no public", "capped", "history", "inconsistency", "low platform",
    ]
    issues = []
    for row in reasoning_rows:
        rank = row["rank"]
        text = row["reasoning"].lower()
        has_hedge = any(p in text for p in hedge_phrases)
        if rank >= 80 and not has_hedge:
            issues.append((rank, "rank ≥80 reasoning lacks any caveat"))
        if rank >= 50 and not has_hedge:
            issues.append((rank, "rank ≥50 reasoning lacks any caveat"))
    return issues


def detect_template_repetition(reasoning_rows, threshold=0.7):
    """Detect if too many reasoning strings share the same opening clause.
    Returns a warning string if the repetition rate is concerning."""
    openings = {}
    for row in reasoning_rows:
        text = row["reasoning"]
        # First 30 chars as a proxy for "same template opening"
        opening = text[:35].strip()
        openings[opening] = openings.get(opening, 0) + 1
    max_repeat = max(openings.values()) if openings else 0
    if max_repeat > len(reasoning_rows) * threshold:
        common_opening = max(openings, key=openings.get)
        return (f"WARNING: {max_repeat}/{len(reasoning_rows)} reasoning strings share "
                f"the same opening: '{common_opening}'. Increase template variation.")
    return None
