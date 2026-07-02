"""
metrics.py
Our own offline copy of the competition's exact scoring metrics, so we can
tune gate thresholds and scoring weights against something real instead of
intuition. There is no public leaderboard and only 3 submissions total —
this is the instrument panel.

Final composite = 0.50 x NDCG@10 + 0.30 x NDCG@50 + 0.15 x MAP + 0.05 x P@10
(submission_spec.md Section 4)
"""
import math


def dcg_at_k(relevances, k):
    """Standard DCG with the common (2^rel - 1)/log2(rank+1) gain function."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        rank = i + 1
        dcg += (2 ** rel - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(ranked_relevances, k):
    """ranked_relevances: list of true relevance tiers (0-5) in the order
    OUR system ranked them (index 0 = our rank 1)."""
    actual_dcg = dcg_at_k(ranked_relevances, k)
    ideal = sorted(ranked_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def map_score(ranked_relevances, relevance_threshold=3):
    """Mean Average Precision treating relevance >= threshold as 'relevant'."""
    relevant_flags = [1 if r >= relevance_threshold else 0 for r in ranked_relevances]
    total_relevant = sum(relevant_flags)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, flag in enumerate(relevant_flags):
        if flag:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / total_relevant


def precision_at_k(ranked_relevances, k, relevance_threshold=3):
    top_k = ranked_relevances[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if r >= relevance_threshold)
    return relevant / len(top_k)


def composite(ranked_relevances):
    return (
        0.50 * ndcg_at_k(ranked_relevances, 10) +
        0.30 * ndcg_at_k(ranked_relevances, 50) +
        0.15 * map_score(ranked_relevances) +
        0.05 * precision_at_k(ranked_relevances, 10)
    )


def composite_breakdown(ranked_relevances):
    return {
        "ndcg@10": ndcg_at_k(ranked_relevances, 10),
        "ndcg@50": ndcg_at_k(ranked_relevances, 50),
        "map": map_score(ranked_relevances),
        "p@10": precision_at_k(ranked_relevances, 10),
        "composite": composite(ranked_relevances),
    }


# ---------------------------------------------------------------------------
# Unit tests against hand-computed toy examples (BUILD_SPEC.md Section 3
# Step 2 explicitly requires this before trusting the implementation).
# Run directly: `python src/metrics.py`
# ---------------------------------------------------------------------------
def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def _run_self_tests():
    # Test 1: perfect ranking (already sorted descending) -> NDCG = 1.0
    perfect = [5, 5, 4, 3, 2, 1, 0, 0, 0, 0]
    assert _approx(ndcg_at_k(perfect, 10), 1.0), f"perfect NDCG@10 should be 1.0, got {ndcg_at_k(perfect, 10)}"

    # Test 2: worst ranking (ascending instead of descending) -> NDCG < 1.0
    worst = list(reversed(perfect))
    assert ndcg_at_k(worst, 10) < 1.0, "worst-case ordering should score below 1.0"
    assert ndcg_at_k(worst, 10) > 0.0, "worst-case ordering should still be > 0 (some relevant items present)"

    # Test 3: all-zero relevances -> NDCG defined as 0 (ideal_dcg is 0, guard divides safely)
    all_zero = [0] * 10
    assert ndcg_at_k(all_zero, 10) == 0.0

    # Test 4: hand-computed DCG check.
    # relevances = [3, 2, 0], k=3
    # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^0-1)/log2(4)
    #     = 7/1 + 3/1.58496 + 0/2
    #     = 7.0 + 1.892789... + 0 = 8.892789...
    hand = [3, 2, 0]
    expected_dcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**0 - 1) / math.log2(4)
    assert _approx(dcg_at_k(hand, 3), expected_dcg), f"DCG hand-check failed: {dcg_at_k(hand, 3)} vs {expected_dcg}"

    # Test 5: MAP hand check.
    # relevances (threshold=3): [5, 1, 4, 0, 3] -> relevant flags [1,0,1,0,1]
    # precisions at hit positions: pos1 (1/1=1.0), pos3 (2/3=0.667), pos5 (3/5=0.6)
    # MAP = (1.0 + 0.667 + 0.6) / 3 = 0.7556
    map_test = [5, 1, 4, 0, 3]
    expected_map = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert _approx(map_score(map_test, 3), expected_map), f"MAP hand-check failed: {map_score(map_test,3)} vs {expected_map}"

    # Test 6: MAP with zero relevant items -> 0.0, not a crash
    assert map_score([0, 0, 0], 3) == 0.0

    # Test 7: precision@k hand check.
    # [5,4,3,2,1,0,0,0,0,0], threshold=3, k=5 -> 3 of top 5 are >=3 -> 0.6
    p_test = [5, 4, 3, 2, 1, 0, 0, 0, 0, 0]
    assert _approx(precision_at_k(p_test, 5, 3), 0.6), f"P@5 hand-check failed: {precision_at_k(p_test,5,3)}"

    # Test 8: precision@k with k larger than list -> uses available items only
    short = [5, 5]
    assert _approx(precision_at_k(short, 10, 3), 1.0)

    # Test 9: composite weighting sanity -- perfect ranking should give
    # composite very close to 1.0 (MAP and P@10 are well-defined and =1 too
    # when everything above threshold is front-loaded correctly)
    perfect_long = [5] * 10 + [4] * 40 + [0] * 50  # 50 "relevant" (>=3) items, all front-loaded
    comp = composite(perfect_long)
    assert comp > 0.99, f"near-perfect front-loaded ranking should score near 1.0, got {comp}"

    print("All metric self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
