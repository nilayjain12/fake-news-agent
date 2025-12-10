# backend/tools/custom_tools.py (SIMPLIFIED - Remove LLM dependencies)
"""
Simplified tools that don't make LLM API calls.
All LLM-heavy operations moved to root_orchestrator.
"""

import requests
from bs4 import BeautifulSoup
from tools.faiss_tool import faiss_search_tool
from tools.google_search_tool import google_search
from config import get_logger

logger = get_logger(__name__)


# ===== SIMPLE UTILITY FUNCTIONS (NO API CALLS) =====

def extract_url_content(url: str) -> str:
    """Extract text from URL - NO LLM CALL"""
    logger.warning("📥 Fetching URL: %s", url[:60])
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs)
        
        logger.warning("✅ Extracted %d paragraphs", len(paragraphs))
        return text[:5000]  # Limit length
    except Exception as e:
        logger.warning("❌ Failed to extract URL: %s", str(e)[:50])
        return ""


def validate_and_clean_text(text: str) -> str:
    """Clean input text - NO LLM CALL"""
    logger.warning("🧹 Cleaning text")
    if not text or len(text.strip()) == 0:
        return ""
    
    # Remove extra whitespace
    cleaned = " ".join(text.split())
    result = cleaned[:5000]
    
    logger.warning("✅ Cleaned to %d chars", len(result))
    return result


def extract_main_claim(text: str) -> str:
    """
    Extract main claim using simple heuristics - NO LLM CALL
    This is now handled by orchestrator's combined LLM call
    """
    logger.warning("🔍 Extracting claim (local heuristic)")
    
    # Heuristic: Often the first or second sentence is the main claim
    sentences = text.split(".")
    
    if len(sentences) > 0:
        main_sentence = sentences[0].strip()
        if len(main_sentence) < 20 and len(sentences) > 1:
            main_sentence = sentences[1].strip()
        
        logger.warning("✅ Claim extracted: %s", main_sentence[:60])
        return main_sentence
    
    return text[:100]


def search_knowledge_base(claim: str) -> list:
    """Search FAISS - Delegates to tool - NO ADDITIONAL API CALLS"""
    logger.warning("🔎 Searching FAISS (k=5)")
    try:
        results = faiss_search_tool(claim, k=5)
        logger.warning("✅ Found %d FAISS results", len(results))
        return results
    except Exception as e:
        logger.warning("❌ FAISS error: %s", str(e)[:50])
        return []


def search_web(claim: str) -> list:
    """Search Google - Delegates to tool - NO ADDITIONAL API CALLS"""
    logger.warning("🌐 Searching Google (k=10)")
    try:
        results = google_search(claim, top_k=10)
        logger.warning("✅ Found %d web results", len(results))
        return results
    except Exception as e:
        logger.warning("❌ Google search error: %s", str(e)[:50])
        return []


def evaluate_evidence(claim: str, evidence_list: list) -> dict:
    """
    Evaluate evidence using local heuristics - NO LLM CALL
    This is now handled by orchestrator's local evaluation
    """
    logger.warning("📊 Evaluating %d evidence items (local heuristic)", len(evidence_list))
    
    support_keywords = ["true", "confirm", "verify", "correct", "accurate", "proven"]
    refute_keywords = ["false", "incorrect", "wrong", "deny", "refute", "debunk"]
    
    supports = 0
    refutes = 0
    
    for item in evidence_list[:10]:
        content = item.get('content', '').lower() if isinstance(item, dict) else str(item).lower()
        
        support_score = sum(1 for kw in support_keywords if kw in content)
        refute_score = sum(1 for kw in refute_keywords if kw in content)
        
        if support_score > refute_score:
            supports += 1
        elif refute_score > support_score:
            refutes += 1
    
    logger.warning("✅ Evaluated: %d support, %d refute", supports, refutes)
    return {"supports": supports, "refutes": refutes, "total": len(evidence_list[:10])}


def count_evidence(evaluations: list) -> dict:
    """Count evidence - NO LLM CALL"""
    logger.warning("📈 Counting evidence")
    
    if isinstance(evaluations, dict):
        # Already in dict format
        return evaluations
    
    supports = sum(1 for e in evaluations if isinstance(e, dict) and e.get("label") == "SUPPORTS")
    refutes = sum(1 for e in evaluations if isinstance(e, dict) and e.get("label") == "REFUTES")
    
    logger.warning("✅ Supports: %d, Refutes: %d", supports, refutes)
    return {"supports": supports, "refutes": refutes}


def generate_verdict(supports: int, refutes: int) -> dict:
    """Generate verdict using heuristic - NO LLM CALL"""
    logger.warning("📋 Generating verdict (S:%d R:%d)", supports, refutes)
    
    total = supports + refutes
    
    if total == 0:
        verdict = "INCONCLUSIVE"
        confidence = 0.3
    elif supports > refutes * 2:  # Strong support
        verdict = "TRUE"
        confidence = min(0.95, 0.6 + (supports / max(total, 1)) * 0.35)
    elif refutes > supports * 2:  # Strong refutation
        verdict = "FALSE"
        confidence = min(0.95, 0.6 + (refutes / max(total, 1)) * 0.35)
    else:  # Mixed or insufficient
        verdict = "INCONCLUSIVE"
        confidence = 0.5
    
    logger.warning("✅ Verdict: %s (confidence: %.1f%%)", verdict, confidence * 100)
    return {"verdict": verdict, "confidence": confidence}


def format_report(claim: str, verdict: str, confidence: float, evidence_count: int) -> str:
    """Format report - NO LLM CALL"""
    logger.warning("📄 Formatting report")
    
    report = f"""# 📋 Fact-Check Report

**Claim:** {claim}

**Verdict:** **{verdict}** ({confidence:.1%} confidence)

**Evidence Evaluated:** {evidence_count}

---
*Generated by Optimized ADK Fact Check Agent*"""
    
    return report


# ===== UTILITY HELPERS =====

def is_url(text: str) -> bool:
    """Check if text is a URL"""
    return text.strip().startswith(("http://", "https://"))


def extract_claim_summary(text: str, max_length: int = 200) -> str:
    """Extract summary of claim"""
    text = text.strip()
    if len(text) <= max_length:
        return text
    
    # Find natural break point
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.7:
        return text[:last_period + 1]
    
    return truncated + "..."