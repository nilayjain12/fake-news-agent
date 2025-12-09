# ==============================================================================
# FILE 2: backend/tools/custom_tools.py (NEW FILE)
# ==============================================================================

import requests
from bs4 import BeautifulSoup
from tools.faiss_tool import faiss_search_tool
from tools.google_search_tool import google_search_tool
import json
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
from google import genai
import os

logger = get_logger(__name__)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


def extract_url_content(url: str) -> str:
    """Extracts text content from a URL"""
    logger.warning("📥 Fetching URL: %s", url[:60])
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs)
        logger.warning("✅ Extracted %d paragraphs", len(paragraphs))
        return text
    except Exception as e:
        logger.warning("❌ Failed to extract URL: %s", str(e)[:50])
        return ""


def validate_and_clean_text(text: str) -> str:
    """Validates and cleans input text"""
    logger.warning("🧹 Cleaning text (length: %d)", len(text))
    if not text or len(text.strip()) == 0:
        return ""
    cleaned = " ".join(text.split())
    result = cleaned[:5000]
    logger.warning("✅ Cleaned to %d chars", len(result))
    return result


def extract_main_claim(text: str) -> dict:
    """Extracts main claim using LLM"""
    logger.warning("🔍 Extracting main claim from text")
    
    prompt = f"""Extract the PRIMARY claim from this text:

TEXT: {text[:2000]}

Return ONLY JSON:
{{"main_claim": "claim", "claim_type": "type"}}"""
    
    try:
        response = client.models.generate_content(model=ADK_MODEL_NAME, contents=prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        
        parsed = json.loads(json_str.strip())
        logger.warning("✅ Claim extracted: %s", parsed.get('main_claim', '')[:60])
        return parsed
    except Exception as e:
        logger.warning("❌ Error: %s", str(e)[:100])
        return {"main_claim": text[:100], "claim_type": "unknown"}


def search_knowledge_base(claim: str) -> list:
    """Searches FAISS knowledge base"""
    logger.warning("🔎 Searching FAISS (k=5)")
    try:
        results = faiss_search_tool(claim, k=5)
        logger.warning("✅ Found %d FAISS results", len(results))
        return results
    except Exception as e:
        logger.warning("❌ FAISS error: %s", str(e)[:50])
        return []


def search_web(claim: str) -> list:
    """Searches web using Google"""
    logger.warning("🔍 Searching Google (k=10)")
    try:
        results = google_search_tool(claim, top_k=10)
        logger.warning("✅ Found %d web results", len(results))
        return results
    except Exception as e:
        logger.warning("❌ Google search error: %s", str(e)[:50])
        return []


def evaluate_evidence(claim: str, evidence_list: list) -> list:
    """Evaluates evidence as SUPPORTS or REFUTES"""
    logger.warning("📊 Evaluating %d evidence items", len(evidence_list))
    
    if not evidence_list:
        return []
    
    evidence_items = evidence_list[:10]
    
    prompt = f"""Evaluate each evidence against: "{claim}"

For each evidence respond with ONLY: SUPPORTS or REFUTES
One label per line.

Evidence:"""
    
    for i, item in enumerate(evidence_items, 1):
        content = item.get('content', '')[:100] if isinstance(item, dict) else str(item)[:100]
        prompt += f"\n{i}. {content}"
    
    try:
        response = client.models.generate_content(model=ADK_MODEL_NAME, contents=prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        lines = text.strip().split('\n')
        
        evaluations = []
        for i, (line, item) in enumerate(zip(lines, evidence_items), 1):
            label = "SUPPORTS" if "SUPPORT" in line.upper() else "REFUTES"
            evaluations.append({"evidence": item, "label": label})
        
        logger.warning("✅ Evaluated %d items", len(evaluations))
        return evaluations
    except Exception as e:
        logger.warning("❌ Evaluation error: %s", str(e)[:100])
        return []


def count_evidence(evaluations: list) -> dict:
    """Counts SUPPORTS vs REFUTES"""
    logger.warning("📈 Counting evidence")
    supports = sum(1 for e in evaluations if e.get("label") == "SUPPORTS")
    refutes = sum(1 for e in evaluations if e.get("label") == "REFUTES")
    logger.warning("✅ Supports: %d, Refutes: %d", supports, refutes)
    return {"supports": supports, "refutes": refutes}


def generate_verdict(supports: int, refutes: int) -> dict:
    """Generates verdict based on evidence"""
    logger.warning("📋 Generating verdict (S:%d R:%d)", supports, refutes)
    
    total = supports + refutes
    
    if supports > refutes:
        verdict = "TRUE"
        confidence = min(0.95, 0.5 + (supports / max(total, 1)) * 0.45)
    elif refutes > supports:
        verdict = "FALSE"
        confidence = min(0.95, 0.5 + (refutes / max(total, 1)) * 0.45)
    else:
        verdict = "INCONCLUSIVE"
        confidence = 0.5
    
    logger.warning("✅ Verdict: %s (confidence: %.1f%%)", verdict, confidence * 100)
    return {"verdict": verdict, "confidence": confidence}


def format_report(claim: str, verdict: str, confidence: float, evidence_count: int, execution_time: float) -> str:
    """Formats final report"""
    logger.warning("📄 Formatting report")
    
    report = f"""# 📋 Fact-Check Report

**Claim:** {claim}

**Verdict:** **{verdict}** ({confidence:.1%} confidence)

**Evidence Evaluated:** {evidence_count}

**Execution Time:** {execution_time:.0f}ms

---
*Generated by ADK-based Fact Check Agent*"""
    
    return report