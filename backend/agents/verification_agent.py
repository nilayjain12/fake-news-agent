# backend/agents/verification_agent.py
"""Evidence retrieval and evaluation agent."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.faiss_tool import faiss_search
from tools.google_search_tool import google_search_agent_tool
from google.adk.models.google_llm import Gemini
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

_llm = Gemini(model=ADK_MODEL_NAME)

class VerificationAgent:
    """Runs evidence retrieval in parallel (FAISS + Google)."""
    
    def __init__(self, top_k=5, model=_llm):
        self.top_k = top_k
        self.model = model

    def _run_web_search(self, claim):
        try:
            return google_search_agent_tool.tool(query=claim, top_k=5)
        except Exception as e:
            logger.warning("❌ Web search failed: %s", str(e)[:50])
            return []

    def _run_faiss(self, claim):
        try:
            return faiss_search(claim, k=self.top_k)
        except Exception as e:
            logger.warning("❌ FAISS search failed: %s", str(e)[:50])
            return []

    def retrieve_all(self, claim):
        """Run retrievals in parallel."""
        logger.warning("🔍 Retrieving evidence for claim")
        results = {"faiss": [], "web": []}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(self._run_faiss, claim): "faiss",
                ex.submit(self._run_web_search, claim): "web"
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception:
                    results[key] = []
        return results

    def evaluate_evidence(self, claim, evidence_item):
        """Evaluate if evidence supports/refutes the claim."""
        evidence_text = evidence_item.get("content") if isinstance(evidence_item, dict) else str(evidence_item)
        prompt = f"""You are a strict verification model. Given the claim and one piece of evidence,
decide if the evidence SUPPORTS, REFUTES, or gives NOT_ENOUGH_INFO about the claim.
Return a single word: SUPPORTS / REFUTES / NOT_ENOUGH_INFO and a one-line justification.

Claim:
{claim}

Evidence:
{evidence_text}"""
        
        try:
            resp = self.model.generate(prompt)
            text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
            first_line = (text.strip().splitlines() or [""])[0].strip().upper()
            
            label = "NOT_ENOUGH_INFO"
            if "SUPPORT" in first_line:
                label = "SUPPORTS"
            elif "REFUTE" in first_line or "FALSE" in first_line:
                label = "REFUTES"
            
            return {"evidence": evidence_item, "label": label, "raw": text}
        except Exception as e:
            logger.warning("❌ Evaluation failed: %s", str(e)[:50])
            return {"evidence": evidence_item, "label": "NOT_ENOUGH_INFO", "raw": ""}

    def run(self, claim):
        logger.warning("📊 Verifying claim: %s", claim[:60])
        retrieved = self.retrieve_all(claim)
        
        flat = []
        for src, items in retrieved.items():
            for it in items:
                if isinstance(it, dict):
                    it["_source"] = src
                else:
                    it = {"content": str(it), "_source": src}
                flat.append(it)
        
        evaluations = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(self.evaluate_evidence, claim, e) for e in flat]
            for fut in as_completed(futures):
                try:
                    evaluations.append(fut.result())
                except Exception:
                    pass
        
        logger.warning("   → Evaluated %d evidence items", len(evaluations))
        return {"retrieved": retrieved, "evaluations": evaluations}