# backend/agents/verification_agent.py
"""Evidence retrieval and evaluation with SEMANTIC RANKING."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tools.faiss_tool import faiss_search
from tools.google_search_tool import google_search_tool
from tools.semantic_ranker import SemanticRanker
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import os

logger = get_logger(__name__)
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


class VerificationAgent:
    """Runs evidence retrieval with semantic ranking for better accuracy."""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.model = ADK_MODEL_NAME
        self.ranker = SemanticRanker()  # Initialize ranker
        logger.warning("🔍 VerificationAgent initialized (top_k=%d) with semantic ranking", top_k)

    def _run_web_search(self, claim):
            """Run Google Search (now with semantic filtering via ADK Agent)."""
            try:
                logger.warning("🔍 Google Search (ADK Agent): %s", claim[:60])
                
                # Now uses ADK Agent with semantic filtering
                result = google_search_tool(query=claim, top_k=5)
                
                # Ensure result is a list
                if not isinstance(result, list):
                    logger.warning("âš ï¸  Web search returned non-list type: %s", type(result).__name__)
                    return []
                
                logger.warning("   ✅ Found %d semantically relevant results", len(result))
                
                # Validate and log relevance scores
                validated_results = []
                for i, res in enumerate(result, 1):
                    # Ensure res is a dict with required fields
                    if not isinstance(res, dict):
                        logger.warning("   [%d] Skipping non-dict result: %s", i, type(res).__name__)
                        continue
                    
                    relevance_score = res.get('relevance_score', 0)
                    title = res.get('title', 'N/A')
                    
                    logger.warning("   [%d] Relevance: %.2f | %s", i, relevance_score, title[:60])
                    validated_results.append(res)
                
                return validated_results
                
            except Exception as e:
                logger.warning("âš ï¸  Web search failed: %s", str(e)[:50])
                import traceback
                logger.warning("   Traceback: %s", traceback.format_exc()[:200])
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
        """Run retrievals in parallel and rank results."""
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

    def retrieve_and_rank(self, claim):
        """
        NEW METHOD: Retrieve all evidence and rank by semantic relevance.
        
        Returns: (ranked_evidence, retrieval_metadata)
        """
        logger.warning("📊 Starting retrieve-and-rank process...")
        
        # Step 1: Retrieve from both sources
        retrieved = self.retrieve_all(claim)
        
        # Step 2: Combine all results
        all_results = []
        for src, items in retrieved.items():
            for item in items:
                if isinstance(item, dict):
                    item["_source"] = src
                else:
                    item = {"content": str(item), "_source": src}
                all_results.append(item)
        
        logger.warning("📦 Combined results: %d total", len(all_results))
        logger.warning("   FAISS: %d, Google: %d", len(retrieved['faiss']), len(retrieved['web']))
        
        if not all_results:
            logger.warning("⚠️  No results retrieved")
            return [], {"total_retrieved": 0, "top_k_used": 0}
        
        # Step 3: Rank by semantic relevance
        ranked_evidence, similarities = self.ranker.rank_evidence_by_relevance(
            query=claim,
            evidence_items=all_results,
            top_k=self.top_k,
            min_similarity=0.5  # Only keep results with 50%+ similarity
        )
        
        # Metadata for logging
        metadata = {
            "total_retrieved": len(all_results),
            "top_k_used": len(ranked_evidence),
            "retrieval_sources": {
                "faiss": len(retrieved['faiss']),
                "google": len(retrieved['web'])
            },
            "similarities": similarities,
            "ranking_applied": True
        }
        
        logger.warning("✅ Ranking complete: %d → %d (selected top %d)", 
                      len(all_results), len(ranked_evidence), len(ranked_evidence))
        
        return ranked_evidence, metadata

    def batch_evaluate_evidence(self, claim, evidence_items):
        """Batch evaluate ranked evidence."""
        if not evidence_items:
            logger.warning("❌ No evidence items to evaluate")
            return []
        
        evidence_items = evidence_items[:10]  # Limit to first 10
        
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
                
                # Extract label
                if "REFUTE" in line_upper:
                    label = "REFUTES"
                elif "FALSE" in line_upper and "NOT" not in line_upper:
                    label = "REFUTES"
                elif "CONTRADICTS" in line_upper or "DENIES" in line_upper or "DEBUNK" in line_upper:
                    label = "REFUTES"
                elif "SUPPORT" in line_upper and "NOT" not in line_upper:
                    label = "SUPPORTS"
                elif "CONFIRMS" in line_upper or "AFFIRMS" in line_upper:
                    label = "SUPPORTS"
                
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
            return self._fallback_evaluate(claim, evidence_items)

    def _build_evaluation_prompt(self, claim, evidence_items):
        """Build evaluation prompt."""
        evidence_text = "\n\n".join([
            f"[Evidence {i}]\n"
            f"Source: {item.get('_source', 'unknown')}\n"
            f"Content: {item.get('content', str(item))[:300]}"
            for i, item in enumerate(evidence_items, 1)
        ])
        
        prompt = f"""You are an expert fact-checker. Evaluate each evidence against the claim.

CLAIM: "{claim}"

EVIDENCE ITEMS:
{evidence_text}

RULES:
- SUPPORTS: Evidence confirms the claim is TRUE
- REFUTES: Evidence shows the claim is FALSE or contradicts it
- NOT_ENOUGH_INFO: Evidence is vague or doesn't address the claim

Response format (labels only, no explanations):
Evidence 1: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 2: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
..."""
        
        return prompt

    def _fallback_evaluate(self, claim, evidence_items):
        """Fallback evaluation using keyword matching."""
        logger.warning("⚠️  Using fallback keyword-based evaluation")
        
        evaluations = []
        claim_lower = claim.lower()
        
        for item in evidence_items:
            content = str(item.get('content', '')).lower()
            label = "NOT_ENOUGH_INFO"
            
            if "alive" in content and "dead" in claim_lower:
                label = "REFUTES"
            elif "dead" in content and "dead" in claim_lower:
                label = "SUPPORTS"
            elif "died" in content and "death" in claim_lower or "died" in claim_lower:
                label = "SUPPORTS"
            
            evaluations.append({
                "evidence": item,
                "label": label,
                "raw": "Fallback evaluation"
            })
        
        return evaluations

    def run(self, claim):
        """Run complete verification with semantic ranking."""
        logger.warning("📊 Verifying claim: %s", claim[:60])
        
        # NEW: Use retrieve_and_rank instead of retrieve_all
        ranked_evidence, metadata = self.retrieve_and_rank(claim)
        
        logger.warning("   → Using %d ranked evidence items", len(ranked_evidence))
        
        if not ranked_evidence:
            logger.warning("⚠️  No evidence after ranking")
            return {
                "retrieved": {"faiss": [], "web": []},
                "evaluations": [],
                "ranking_metadata": metadata
            }
        
        # Batch evaluate ranked evidence
        evaluations = self.batch_evaluate_evidence(claim, ranked_evidence)
        
        logger.warning("✅ Verification complete with %d evaluations", len(evaluations))
        
        return {
            "retrieved": {"ranked": ranked_evidence},
            "evaluations": evaluations,
            "ranking_metadata": metadata
        }