"""Basic aggregator to combine evidence evaluations into a final verdict."""
from collections import Counter
from config import get_logger

logger = get_logger(__name__)

def aggregate_evaluations(evaluations):
    """
    Basic aggregation:
    - Count SUPPORTS vs REFUTES across evidence.
    - Optionally weight by source type (factcheck > faiss > web).
    """
    logger.info("Aggregating %d evaluations", len(evaluations))
    weight_map = {"faiss": 1.0, "web": 0.8}
    score = 0.0
    total_weight = 0.0

    for ev in evaluations:
        label = ev.get("label")
        src = ev.get("evidence", {}).get("_source", "web")
        w = weight_map.get(src, 1.0)
        if label == "SUPPORTS":
            score += 1.0 * w
        elif label == "REFUTES":
            score -= 1.0 * w
        total_weight += w

    # Simple normalized score
    if total_weight == 0:
        normalized = 0
    else:
        normalized = score / total_weight

    # thresholds are heuristic; tweak as needed
    if normalized > 0.2:
        verdict = "Likely True"
    elif normalized < -0.2:
        verdict = "Likely False"
    else:
        verdict = "Unverified / Mixed Evidence"

    logger.info("Aggregation result verdict=%s score=%s total_weight=%s", verdict, normalized, total_weight)
    return {"verdict": verdict, "score": normalized, "raw_score": score, "total_weight": total_weight}