# backend/agents/aggregator_and_verdict.py
"""Aggregator to combine evidence evaluations into a final verdict."""
from collections import Counter
from config import get_logger

logger = get_logger(__name__)

def aggregate_evaluations(evaluations):
    """Aggregate evaluations with intelligent, robust scoring."""
    logger.warning("📈 Aggregating %d evaluations", len(evaluations))
    
    if not evaluations:
        logger.warning("⚠️  No evaluations to aggregate")
        return {
            "verdict": "Unverified / No Evidence Found",
            "score": 0.0,
            "raw_score": 0,
            "total_weight": 0,
            "breakdown": {"SUPPORTS": 0, "REFUTES": 0, "NOT_ENOUGH_INFO": 0}
        }
    
    # Separate evaluations by type
    supports_evals = []
    refutes_evals = []
    neutral_evals = []
    
    for ev in evaluations:
        label = ev.get("label", "NOT_ENOUGH_INFO")
        evidence = ev.get("evidence", {})
        src = evidence.get("_source", "web") if isinstance(evidence, dict) else "web"
        
        # Weight map: web search (current info) > FAISS (knowledge base)
        weight = 1.5 if src in ["web", "google_search", "google_search_fallback"] else 1.0
        
        if label == "SUPPORTS":
            supports_evals.append((weight, src))
        elif label == "REFUTES":
            refutes_evals.append((weight, src))
        else:  # NOT_ENOUGH_INFO
            neutral_evals.append((weight, src))
    
    logger.warning("📊 Detailed evaluation breakdown:")
    logger.warning("   SUPPORTS: %d items", len(supports_evals))
    for i, (w, src) in enumerate(supports_evals, 1):
        logger.warning("      %d. weight=%.1f src=%s", i, w, src)
    
    logger.warning("   REFUTES: %d items", len(refutes_evals))
    for i, (w, src) in enumerate(refutes_evals, 1):
        logger.warning("      %d. weight=%.1f src=%s", i, w, src)
    
    logger.warning("   NOT_ENOUGH_INFO: %d items (neutral, not scored)", len(neutral_evals))
    
    # Calculate scores - ONLY from SUPPORTS and REFUTES
    # NOT_ENOUGH_INFO does NOT contribute to the score
    support_score = sum(w for w, _ in supports_evals)
    refute_score = sum(w for w, _ in refutes_evals)
    
    # Raw score: positive for supports, negative for refutes
    raw_score = support_score - refute_score
    total_weighted_evidence = support_score + refute_score
    
    # Normalized score: -1.0 to +1.0
    if total_weighted_evidence == 0:
        # No actual evidence (only NOT_ENOUGH_INFO) - return neutral
        normalized = 0.0
        confidence = 0.0
        verdict = "Unverified / Insufficient Evidence"
        logger.warning("📊 No actual evidence found (only NOT_ENOUGH_INFO)")
        return {
            "verdict": verdict,
            "score": 0.0,
            "raw_score": 0,
            "total_weight": 0,
            "confidence": 0.0,
            "breakdown": {
                "SUPPORTS": len(supports_evals),
                "REFUTES": len(refutes_evals),
                "NOT_ENOUGH_INFO": len(neutral_evals)
            }
        }
    else:
        normalized = raw_score / total_weighted_evidence
    
    logger.warning("📊 Evidence calculation:")
    logger.warning("   Support Score: %.1f", support_score)
    logger.warning("   Refute Score: %.1f", refute_score)
    logger.warning("   Raw Score: %.1f", raw_score)
    logger.warning("   Total Weighted Evidence: %.1f", total_weighted_evidence)
    logger.warning("   Normalized Score: %.2f", normalized)
    
    # Generate verdict based on normalized score
    verdict, confidence = _generate_verdict_and_confidence(
        normalized,
        len(supports_evals),
        len(refutes_evals),
        len(neutral_evals)
    )
    
    logger.warning("   → Verdict: %s (confidence: %.1f%%)", verdict, confidence * 100)
    
    return {
        "verdict": verdict,
        "score": normalized,
        "raw_score": raw_score,
        "total_weight": total_weighted_evidence,
        "confidence": confidence,
        "breakdown": {
            "SUPPORTS": len(supports_evals),
            "REFUTES": len(refutes_evals),
            "NOT_ENOUGH_INFO": len(neutral_evals)
        }
    }


def _generate_verdict_and_confidence(normalized_score, support_count, refute_count, neutral_count):
    """Generate verdict and confidence based on evidence."""
    
    # Check if evidence is overwhelming in one direction
    total_evidence = support_count + refute_count
    
    # If mostly refuting evidence
    if refute_count > support_count * 1.5:  # At least 1.5x more refutes
        if normalized_score < -0.8:
            return "False", 0.95
        elif normalized_score < -0.5:
            return "Likely False", 0.75
        elif normalized_score < -0.2:
            return "Mostly False", 0.60
        else:
            return "Likely False", 0.60
    
    # If mostly supporting evidence
    elif support_count > refute_count * 1.5:  # At least 1.5x more supports
        if normalized_score > 0.8:
            return "True", 0.95
        elif normalized_score > 0.5:
            return "Likely True", 0.75
        elif normalized_score > 0.2:
            return "Mostly True", 0.60
        else:
            return "Likely True", 0.60
    
    # Balanced evidence (mixed)
    else:
        if normalized_score > 0.3:
            return "Likely True", 0.50
        elif normalized_score < -0.3:
            return "Likely False", 0.50
        else:
            return "Mixed Evidence / Inconclusive", 0.40
    
    # Fallback
    return "Unverified", 0.30


def calculate_confidence_from_evidence_quality(support_count, refute_count, neutral_count, normalized_score):
    """Calculate confidence based on evidence quality and consensus."""
    
    total_evidence = support_count + refute_count
    
    if total_evidence == 0:
        return 0.0
    
    # Base confidence from normalized score magnitude
    base_confidence = min(abs(normalized_score), 1.0)
    
    # Boost if there's strong consensus (one direction dominates)
    if support_count > refute_count * 2:
        base_confidence = min(base_confidence + 0.25, 1.0)
    elif refute_count > support_count * 2:
        base_confidence = min(base_confidence + 0.25, 1.0)
    
    # Reduce if evidence is sparse
    if total_evidence < 3:
        base_confidence = max(base_confidence * 0.6, 0.3)
    
    # Reduce if high amount of NOT_ENOUGH_INFO (suggests weak evidence base)
    if neutral_count > total_evidence:
        base_confidence = max(base_confidence * 0.5, 0.2)
    
    return min(base_confidence, 1.0)