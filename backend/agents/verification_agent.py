# backend/agents/verification_agent.py
"""Evidence retrieval and evaluation with minimal API calls."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.faiss_tool import faiss_search
from tools.google_search_tool import google_search_tool
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import os
import time

logger = get_logger(__name__)

# Set up the Gemini client
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


class VerificationAgent:
    """Runs evidence retrieval and batches evaluations to minimize API calls."""
    
    def __init__(self, top_k: int = 3):  # Reduced from 5 to 3
        self.top_k = top_k
        self.model = ADK_MODEL_NAME
        logger.warning("🔍 VerificationAgent initialized (top_k=%d)", top_k)

    def _run_web_search(self, claim):
        """Run Google Search for the claim."""
        try:
            logger.warning("🔎 Google Search: %s", claim[:60])
            result = google_search_tool(query=claim, top_k=3)  # Reduced from 5
            logger.warning("   → Found %d web results", len(result))
            return result
        except Exception as e:
            logger.warning("⚠️  Web search failed: %s", str(e)[:50])
            return []

    def _run_faiss(self, claim):
        """Run FAISS search for the claim."""
        try:
            logger.warning("🔎 FAISS search: %s", claim[:60])
            result = faiss_search(claim, k=self.top_k)
            logger.warning("   → Found %d FAISS results", len(result))
            return result
        except Exception as e:
            logger.warning("⚠️  FAISS search failed: %s", str(e)[:50])
            return []

    def retrieve_all(self, claim):
        """Run retrievals in parallel (local ops, no API calls)."""
        logger.warning("🌐 Retrieving evidence from FAISS + Google...")
        results = {"faiss": [], "web": []}
        
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {
                ex.submit(self._run_faiss, claim): "faiss",
                ex.submit(self._run_web_search, claim): "web"
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    logger.warning("⚠️  Error in %s: %s", key, str(e)[:50])
                    results[key] = []
        
        return results

    def batch_evaluate_evidence(self, claim, evidence_items):
        """
        Batch evaluate multiple evidence items in ONE API call (not one per item).
        This reduces API calls from N to 1!
        """
        if not evidence_items:
            return []
        
        # Limit to first 5 items to save tokens
        evidence_items = evidence_items[:5]
        
        try:
            # Create a batch prompt that evaluates all evidence at once
            evidence_text = "\n\n".join([
                f"Evidence {i}: {item.get('content', str(item))[:300]}"
                for i, item in enumerate(evidence_items, 1)
            ])
            
            prompt = f"""You are a strict verification model. For the claim and evidence items, evaluate each:

Claim:
{claim}

Evidence Items:
{evidence_text}

For EACH evidence item, respond with:
Evidence 1: SUPPORTS / REFUTES / NOT_ENOUGH_INFO
Evidence 2: SUPPORTS / REFUTES / NOT_ENOUGH_INFO
[etc]

Respond ONLY with the format above, nothing else."""
            
            logger.warning("🔄 Batch evaluating %d evidence items (1 API call)", len(evidence_items))
            
            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            text = response.text if hasattr(response, 'text') else str(response)
            lines = text.strip().split('\n')
            
            evaluations = []
            for i, (line, item) in enumerate(zip(lines, evidence_items), 1):
                label = "NOT_ENOUGH_INFO"
                if "SUPPORT" in line.upper():
                    label = "SUPPORTS"
                elif "REFUTE" in line.upper() or "FALSE" in line.upper():
                    label = "REFUTES"
                
                evaluations.append({
                    "evidence": item,
                    "label": label,
                    "raw": line
                })
            
            logger.warning("✅ Batch evaluation complete (%d evaluations from 1 API call)", len(evaluations))
            return evaluations
            
        except Exception as e:
            logger.warning("⚠️  Batch evaluation failed: %s", str(e)[:50])
            # Fallback: Mark all as NOT_ENOUGH_INFO
            return [{
                "evidence": item,
                "label": "NOT_ENOUGH_INFO",
                "raw": "Error during evaluation"
            } for item in evidence_items]

    def run(self, claim):
        """Run the complete verification process for a claim."""
        logger.warning("📊 Verifying claim: %s", claim[:60])
        
        # Retrieve evidence (NO API calls - these are local/cached ops)
        retrieved = self.retrieve_all(claim)
        
        # Flatten evidence items
        flat = []
        for src, items in retrieved.items():
            for it in items:
                if isinstance(it, dict):
                    it["_source"] = src
                else:
                    it = {"content": str(it), "_source": src}
                flat.append(it)
        
        logger.warning("   → Retrieved %d evidence items", len(flat))
        
        # Batch evaluate (1 API call instead of N!)
        evaluations = self.batch_evaluate_evidence(claim, flat)
        
        logger.warning("✅ Verification complete")
        return {"retrieved": retrieved, "evaluations": evaluations}