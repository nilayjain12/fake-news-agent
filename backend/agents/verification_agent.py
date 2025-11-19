"""This agent orchestrates parallel evidence searches (FAISS, Google Search) and evaluates evidence with Gemini."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.faiss_tool import faiss_search
from tools.google_search_tool import google_search_agent_tool
from google.adk.models.google_llm import Gemini
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

_llm = Gemini(model=ADK_MODEL_NAME)

class VerificationAgent:
    """
    Runs evidence retrieval in parallel (FAISS + Google),
    then evaluates retrieved evidence using the LLM.
    """
    def __init__(self, top_k=5, model=_llm):
        self.top_k = top_k
        self.model = model
        logger.debug("VerificationAgent initialized top_k=%d", top_k)

    def _run_web_search(self, claim):
        # Use ADK's google_search via the AgentTool wrapper.
        try:
            logger.debug("Running web search for claim: %s", claim)
            return google_search_agent_tool.tool(query=claim, top_k=5)
        except Exception as e:
            logger.exception("Web search failed: %s", e)
            return []

    def _run_faiss(self, claim):
        try:
            logger.debug("Running FAISS search for claim: %s", claim)
            return faiss_search(claim, k=self.top_k)
        except Exception as e:
            logger.exception("FAISS search failed: %s", e)
            return []

    def retrieve_all(self, claim):
        # Run retrievals in parallel threads for speed
        logger.info("Retrieving evidence for claim")
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
                    logger.debug("Retrieved %d items from %s", len(results[key]) if results[key] else 0, key)
                except Exception as e:
                    logger.exception("Retrieval future failed for %s: %s", key, e)
                    results[key] = []
        return results

    def evaluate_evidence(self, claim, evidence_item):
        """
        Ask the LLM whether a single evidence item Supports/Refutes/NotEnoughInfo the claim.
        """
        evidence_text = evidence_item.get("content") if isinstance(evidence_item, dict) else str(evidence_item)
        prompt = f"""
You are a strict verification model. Given the claim and one piece of evidence,
decide if the evidence SUPPORTS, REFUTES, or gives NOT_ENOUGH_INFO about the claim.
Return a single word: SUPPORTS / REFUTES / NOT_ENOUGH_INFO and a one-line justification.

Claim:
{claim}

Evidence:
{evidence_text}
"""
        try:
            resp = self.model.generate(prompt)
            text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
            logger.debug("Evaluation model output (truncated): %s", (text or "")[:300])
            # naive parsing:
            first_line = (text.strip().splitlines() or [""])[0].strip().upper()
            label = "NOT_ENOUGH_INFO"
            if "SUPPORT" in first_line:
                label = "SUPPORTS"
            elif "REFUTE" in first_line or "FALSE" in first_line:
                label = "REFUTES"
            return {"evidence": evidence_item, "label": label, "raw": text}
        except Exception as e:
            logger.exception("Failed to evaluate evidence: %s", e)
            return {"evidence": evidence_item, "label": "NOT_ENOUGH_INFO", "raw": ""}

    def run(self, claim):
        logger.info("VerificationAgent.run claim=%s", claim)
        # 1) Retrieve evidence
        retrieved = self.retrieve_all(claim)

        # 2) Flatten evidence items to a single list with provenance
        flat = []
        for src, items in retrieved.items():
            for it in items:
                # annotate provenance
                if isinstance(it, dict):
                    it["_source"] = src
                else:
                    it = {"content": str(it), "_source": src}
                flat.append(it)

        logger.debug("Total flattened evidence items=%d", len(flat))

        # 3) Evaluate each piece of evidence (we can parallelize this too)
        evaluations = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(self.evaluate_evidence, claim, e) for e in flat]
            for fut in as_completed(futures):
                try:
                    evaluations.append(fut.result())
                except Exception as e:
                    logger.exception("Evaluation future failed: %s", e)

        logger.info("Verification complete evaluations=%d", len(evaluations))
        return {"retrieved": retrieved, "evaluations": evaluations}