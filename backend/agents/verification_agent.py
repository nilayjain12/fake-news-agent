# backend/agents/verification_agent.py
"""Evidence retrieval and evaluation with improved API integration."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.faiss_tool import faiss_search
from tools.google_search_tool import google_search_tool
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import os
import time
import json

logger = get_logger(__name__)

# Set up the Gemini client
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


class VerificationAgent:
    """Runs evidence retrieval and batches evaluations to minimize API calls."""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.model = ADK_MODEL_NAME
        logger.warning("🔍 VerificationAgent initialized (top_k=%d)", top_k)

    def _run_web_search(self, claim):
        """Run Google Search for the claim with improved query handling."""
        try:
            logger.warning("🔎 Google Search: %s", claim[:60])
            # Extract key terms from claim for better search
            result = google_search_tool(query=claim, top_k=5)
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

    def _build_evaluation_prompt(self, claim, evidence_items):
        """Build a robust evaluation prompt with detailed instructions."""
        evidence_text = "\n\n".join([
            f"[Evidence {i}]\n{item.get('content', str(item))[:500]}"
            for i, item in enumerate(evidence_items, 1)
        ])
        
        prompt = f"""You are a fact-checking expert. Analyze each piece of evidence and determine if it SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO about the claim.

CLAIM TO VERIFY:
"{claim}"

EVIDENCE ITEMS:
{evidence_text}

EVALUATION CRITERIA:
- SUPPORTS: Evidence clearly confirms the claim is true
- REFUTES: Evidence clearly shows the claim is false or contradicts it
- NOT_ENOUGH_INFO: Evidence is irrelevant or doesn't address the claim

Provide ONLY the evaluation labels in this format, one per line:
Evidence 1: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 2: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
[etc...]

IMPORTANT: Do NOT include any explanation, just the labels."""
        
        return prompt

    def batch_evaluate_evidence(self, claim, evidence_items):
        """
        Batch evaluate multiple evidence items in ONE API call with improved logic.
        This reduces API calls from N to 1!
        """
        if not evidence_items:
            logger.warning("❌ No evidence items to evaluate")
            return []
        
        # Limit to first 10 items to save tokens but be more thorough
        evidence_items = evidence_items[:10]
        
        try:
            prompt = self._build_evaluation_prompt(claim, evidence_items)
            
            logger.warning("🔄 Batch evaluating %d evidence items (1 API call)", len(evidence_items))
            
            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            text = response.text if hasattr(response, 'text') else str(response)
            logger.warning("📝 Raw evaluation response: %s", text[:200])
            
            lines = text.strip().split('\n')
            
            evaluations = []
            label_count = {"SUPPORTS": 0, "REFUTES": 0, "NOT_ENOUGH_INFO": 0}
            
            for i, (line, item) in enumerate(zip(lines, evidence_items), 1):
                label = "NOT_ENOUGH_INFO"
                line_upper = line.upper()
                
                # More robust label extraction
                if "SUPPORT" in line_upper and "NOT" not in line_upper:
                    label = "SUPPORTS"
                elif "REFUTE" in line_upper or "FALSE" in line_upper and "NOT" not in line_upper:
                    label = "REFUTES"
                elif "NOT_ENOUGH" in line_upper or "NOT ENOUGH" in line_upper:
                    label = "NOT_ENOUGH_INFO"
                
                label_count[label] += 1
                
                evaluations.append({
                    "evidence": item,
                    "label": label,
                    "raw": line.strip()
                })
                
                logger.warning("   Evidence %d: %s", i, label)
            
            logger.warning("✅ Batch evaluation complete - SUPPORTS: %d, REFUTES: %d, NOT_ENOUGH: %d", 
                          label_count["SUPPORTS"], label_count["REFUTES"], label_count["NOT_ENOUGH_INFO"])
            
            return evaluations
            
        except Exception as e:
            logger.warning("⚠️  Batch evaluation failed: %s", str(e)[:100])
            # Fallback: Try to evaluate manually
            return self._fallback_evaluate(claim, evidence_items)

    def _fallback_evaluate(self, claim, evidence_items):
        """Fallback evaluation using keyword matching when API fails."""
        logger.warning("⚠️  Using fallback keyword-based evaluation")
        
        evaluations = []
        claim_lower = claim.lower()
        
        for item in evidence_items:
            content = str(item.get('content', '')).lower()
            label = "NOT_ENOUGH_INFO"
            
            # Extract key terms from claim
            key_terms = [term.strip() for term in claim_lower.split() if len(term) > 3]
            matching_terms = sum(1 for term in key_terms if term in content)
            
            if matching_terms > 0:
                # Check for negation words that might indicate refutation
                negation_words = ["not", "false", "deny", "refute", "wrong", "incorrect", "contradicts"]
                has_negation = any(neg in content for neg in negation_words)
                
                if has_negation and matching_terms < 3:
                    label = "REFUTES"
                else:
                    label = "SUPPORTS"
            
            evaluations.append({
                "evidence": item,
                "label": label,
                "raw": "Fallback evaluation"
            })
        
        return evaluations

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
        
        logger.warning("   → Retrieved %d evidence items total", len(flat))
        
        if not flat:
            logger.warning("⚠️  No evidence retrieved, using fallback")
            # Return a neutral evaluation when no evidence is found
            return {
                "retrieved": retrieved, 
                "evaluations": [{
                    "evidence": {"content": "No evidence found in knowledge base"},
                    "label": "NOT_ENOUGH_INFO",
                    "raw": "No evidence"
                }]
            }
        
        # Batch evaluate (1 API call instead of N!)
        evaluations = self.batch_evaluate_evidence(claim, flat)
        
        logger.warning("✅ Verification complete with %d evaluations", len(evaluations))
        return {"retrieved": retrieved, "evaluations": evaluations}