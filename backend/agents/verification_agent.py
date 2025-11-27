# backend/agents/verification_agent.py - UPDATED
"""
UPDATED: Binary evidence classification based on relevance (SUPPORTS/REFUTES only)
No NOT_ENOUGH_INFO - all evidence must be classified as relevant (SUPPORTS) or irrelevant (REFUTES)
"""
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
    """
    UPDATED: Binary evidence classification based on relevance
    - SUPPORTS: Evidence is highly relevant to the claim
    - REFUTES: Evidence is not relevant or contradicts the claim
    - NO NOT_ENOUGH_INFO category
    """
    
    def __init__(self, google_top_k: int = 10, faiss_top_k: int = 5, final_top_k: int = 10):
        self.google_top_k = google_top_k
        self.faiss_top_k = faiss_top_k
        self.final_top_k = final_top_k
        self.model = ADK_MODEL_NAME
        self.ranker = SemanticRanker()
        logger.warning("🔍 VerificationAgent initialized (BINARY CLASSIFICATION)")
        logger.warning("   Google Search: %d evidences", google_top_k)
        logger.warning("   FAISS: %d evidences", faiss_top_k)
        logger.warning("   Final Selection: top %d from combined", final_top_k)

    def _run_web_search(self, claim):
        """Run Google Search with increased evidence collection (10 results)."""
        try:
            logger.warning("🔍 Google Search (ADK Agent): %s", claim[:60])
            result = google_search_tool(query=claim, top_k=self.google_top_k)
            
            if result is None:
                logger.warning("   ⚠️  Google search returned None")
                return []
            
            if not isinstance(result, list):
                logger.warning("   ⚠️  Google search returned non-list type: %s", type(result).__name__)
                if isinstance(result, dict):
                    result = [result]
                else:
                    return []
            
            logger.warning("   ✅ Found %d web search results (target: %d)", len(result), self.google_top_k)
            
            validated_results = []
            for i, res in enumerate(result, 1):
                if not isinstance(res, dict):
                    logger.warning("   ⚠️  [%d] Result is not dict (type: %s), skipping", i, type(res).__name__)
                    continue
                
                title = res.get('title', 'N/A')
                relevance_score = res.get('relevance_score', 0)
                content = res.get('content', '')
                url = res.get('url', '')
                
                if not (title or content or url):
                    logger.warning("   ⚠️  [%d] Result missing all fields, skipping", i)
                    continue
                
                logger.warning("   [%d] Title: %s | Relevance: %.2f | URL: %s", 
                             i, title[:50], relevance_score, url[:40] if url else "N/A")
                
                if '_source' not in res:
                    res['_source'] = 'web'
                
                validated_results.append(res)
            
            logger.warning("   ✅ Validated %d web results", len(validated_results))
            return validated_results
                
        except Exception as e:
            logger.warning("❌ Web search exception: %s", str(e)[:100])
            return []

    def _run_faiss(self, claim):
        """Run FAISS search with standard 5 evidences."""
        try:
            logger.warning("🔎 FAISS search (k=%d): %s", self.faiss_top_k, claim[:60])
            result = faiss_search(claim, k=self.faiss_top_k)
            
            if not isinstance(result, list):
                logger.warning("   ⚠️  FAISS returned non-list type: %s", type(result).__name__)
                return []
            
            logger.warning("   ✅ Found %d FAISS results", len(result))
            return result
        except Exception as e:
            logger.warning("❌ FAISS search exception: %s", str(e)[:100])
            return []

    def retrieve_all(self, claim):
        """Run retrievals in parallel (Google 10 + FAISS 5)."""
        logger.warning("🌐 Retrieving evidence (Google 10 + FAISS 5 = 15 total)...")
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
                    logger.warning("❌ Exception in %s retrieval: %s", key, str(e)[:100])
                    results[key] = []
        
        logger.warning("✅ Retrieval complete - FAISS: %d, Web: %d", 
                      len(results['faiss']), len(results['web']))
        return results

    def retrieve_and_rank_improved(self, claim):
        """Retrieve from both sources and rank top 10 by semantic relevance."""
        logger.warning("📊 Starting ADVANCED retrieve-and-rank process...")
        
        retrieved = self.retrieve_all(claim)
        all_results = []
        
        for item in retrieved['faiss']:
            if isinstance(item, dict):
                item['_source'] = 'faiss'
                item['_freshness_boost'] = 0.0
            else:
                item = {"content": str(item), "_source": "faiss", "_freshness_boost": 0.0}
            all_results.append(item)
        
        for item in retrieved['web']:
            if isinstance(item, dict):
                item['_source'] = 'web'
                item['_freshness_boost'] = 0.25
            else:
                item = {"content": str(item), "_source": "web", "_freshness_boost": 0.25}
            all_results.append(item)
        
        logger.warning("📦 Combined results: %d total (FAISS: %d, Web: %d)", 
                      len(all_results), len(retrieved['faiss']), len(retrieved['web']))
        
        if not all_results:
            logger.warning("⚠️  No results retrieved from either source")
            return [], {"total_retrieved": 0, "top_k_used": 0, "ranking_applied": True, 
                       "retrieval_sources": {"faiss": 0, "google": 0}, 
                       "source_breakdown": {"faiss_selected": 0, "web_selected": 0}}
        
        ranked_evidence, similarities, source_breakdown = self._rank_with_semantic_scoring(
            query=claim,
            evidence_items=all_results,
            top_k=self.final_top_k
        )
        
        metadata = {
            "total_retrieved": len(all_results),
            "top_k_used": len(ranked_evidence),
            "retrieval_sources": {
                "faiss": len(retrieved['faiss']),
                "google": len(retrieved['web'])
            },
            "ranking_applied": True,
            "source_breakdown": source_breakdown,
            "similarities": similarities
        }
        
        logger.warning("✅ Ranking complete: %d → %d results", 
                      len(all_results), len(ranked_evidence))
        logger.warning("   Source breakdown: %s", source_breakdown)
        
        return ranked_evidence, metadata

    def _rank_with_semantic_scoring(self, query: str, evidence_items: list, 
                                    top_k: int = 10) -> tuple:
        """Rank evidence using semantic similarity + source freshness boost."""
        if not evidence_items:
            return [], [], {"faiss_selected": 0, "web_selected": 0}
        
        logger.warning("🔄 Computing semantic scores for %d items (selecting top %d)...", 
                      len(evidence_items), top_k)
        
        try:
            query_embedding = self.ranker.embeddings.embed_query(query)
            import numpy as np
            query_vector = np.array(query_embedding)
            
            scored_items = []
            
            for i, item in enumerate(evidence_items):
                try:
                    content = item.get('content', str(item))
                    if isinstance(content, str):
                        content = content[:500]
                    else:
                        content = str(content)[:500]
                    
                    content_embedding = self.ranker.embeddings.embed_query(content)
                    content_vector = np.array(content_embedding)
                    semantic_sim = self._cosine_similarity(query_vector, content_vector)
                    
                    freshness_boost = item.get('_freshness_boost', 0.0)
                    final_score = semantic_sim + freshness_boost
                    
                    source = item.get('_source', 'unknown')
                    
                    scored_items.append({
                        'item': item,
                        'semantic_similarity': float(semantic_sim),
                        'freshness_boost': float(freshness_boost),
                        'final_score': float(final_score),
                        'source': source,
                        'index': i
                    })
                    
                    logger.warning("   [%d] %s: semantic=%.3f + boost=%.3f = %.3f", 
                                 i + 1, source, semantic_sim, freshness_boost, final_score)
                    
                except Exception as e:
                    logger.warning("⚠️  Error scoring item %d: %s", i, str(e)[:50])
                    continue
            
            scored_items.sort(key=lambda x: x['final_score'], reverse=True)
            logger.warning("✅ All %d items ranked by score", len(scored_items))
            
            top_items = scored_items[:top_k]
            logger.warning("📊 Selected top %d items from %d", len(top_items), len(scored_items))
            
            source_breakdown = {
                "faiss_selected": sum(1 for item in top_items if item['source'] == 'faiss'),
                "web_selected": sum(1 for item in top_items if item['source'] == 'web')
            }
            
            similarities = [item['final_score'] for item in top_items]
            evidence = [item['item'] for item in top_items]
            
            logger.warning("✅ Final selection (top %d):", len(top_items))
            for i, (item, score) in enumerate(zip(top_items, similarities), 1):
                logger.warning("   %d. %s (final_score: %.3f)", i, item['source'], score)
            
            logger.warning("   Source breakdown: FAISS=%d, Web=%d", 
                          source_breakdown["faiss_selected"], source_breakdown["web_selected"])
            
            return evidence, similarities, source_breakdown
            
        except Exception as e:
            logger.exception("❌ Semantic scoring error: %s", e)
            logger.warning("⚠️  Fallback: returning first K items")
            sorted_items = sorted(evidence_items, 
                                key=lambda x: (0 if x.get('_source') == 'web' else 1))
            return sorted_items[:top_k], [0.0] * min(top_k, len(sorted_items)), \
                   {"faiss_selected": 0, "web_selected": top_k}

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        """Compute cosine similarity."""
        import numpy as np
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def batch_evaluate_evidence(self, claim, evidence_items):
        """
        UPDATED: Batch evaluate with BINARY classification (SUPPORTS/REFUTES only)
        
        Classification logic:
        - SUPPORTS: Evidence is highly relevant to claim
        - REFUTES: Evidence is not relevant or contradicts claim
        - NO NOT_ENOUGH_INFO option
        """
        if not evidence_items:
            logger.warning("❌ No evidence items to evaluate")
            return []
        
        evidence_items = evidence_items[:10]
        logger.warning("📊 Batch evaluating %d evidence items (BINARY classification)", len(evidence_items))
        
        try:
            prompt = self._build_evaluation_prompt_binary(claim, evidence_items)
            
            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            text = response.text if hasattr(response, 'text') else str(response)
            logger.warning("📝 Raw evaluation response: %s", text[:200])
            
            lines = text.strip().split('\n')
            evaluations = []
            label_count = {"SUPPORTS": 0, "REFUTES": 0}
            
            for i, (line, item) in enumerate(zip(lines, evidence_items), 1):
                # BINARY classification - must be either SUPPORTS or REFUTES
                label = self._classify_binary(line, claim, item)
                label_count[label] += 1
                
                evaluations.append({"evidence": item, "label": label, "raw": line.strip()})
                logger.warning("   Evidence %d: %s", i, label)
            
            logger.warning("✅ Evaluation complete - SUPPORTS: %d, REFUTES: %d", 
                          label_count["SUPPORTS"], label_count["REFUTES"])
            
            return evaluations
            
        except Exception as e:
            logger.warning("⚠️  Evaluation failed: %s", str(e)[:100])
            return self._fallback_evaluate_binary(claim, evidence_items)

    def _classify_binary(self, line: str, claim: str, item: dict) -> str:
        """
        Binary classification based on relevance.
        SUPPORTS: High relevance to claim
        REFUTES: Low relevance or contradicts claim
        """
        line_upper = line.upper()
        
        # Strong keywords for SUPPORTS
        support_keywords = ["SUPPORT", "AGREE", "CONFIRM", "CORROBORATE", "VERIFY", "PROVE", "RELEVANT"]
        
        # Strong keywords for REFUTES  
        refute_keywords = ["REFUTE", "CONTRADICT", "DENY", "DISAGREE", "IRRELEVANT", "UNRELATED", "DISPROVE", "FALSE"]
        
        support_score = sum(1 for kw in support_keywords if kw in line_upper)
        refute_score = sum(1 for kw in refute_keywords if kw in line_upper)
        
        if support_score > refute_score:
            return "SUPPORTS"
        elif refute_score > support_score:
            return "REFUTES"
        else:
            # Default: if ambiguous, check content relevance
            content = str(item.get('content', '')).lower()
            claim_lower = claim.lower()
            
            # Simple heuristic: does content contain claim keywords?
            claim_words = set(claim_lower.split())
            content_words = set(content.split())
            overlap = len(claim_words & content_words) / max(len(claim_words), 1)
            
            if overlap > 0.5:
                return "SUPPORTS"
            else:
                return "REFUTES"

    def _build_evaluation_prompt_binary(self, claim, evidence_items):
        """
        Build evaluation prompt for BINARY classification (SUPPORTS/REFUTES only)
        No middle ground - everything is either relevant (SUPPORTS) or irrelevant (REFUTES)
        """
        evidence_text = "\n\n".join([
            f"[Evidence {i}]\nSource: {item.get('_source', 'unknown')}\n"
            f"Content: {item.get('content', str(item))[:300]}"
            for i, item in enumerate(evidence_items, 1)
        ])
        
        prompt = f"""You are an expert fact-checker. Evaluate each evidence piece against the claim.

CLAIM: "{claim}"

CLASSIFICATION RULES (BINARY - NO MIDDLE GROUND):
- SUPPORTS: Evidence is HIGHLY RELEVANT to the claim OR confirms/supports the claim
- REFUTES: Evidence is NOT RELEVANT to the claim OR contradicts/denies the claim

DO NOT use NOT_ENOUGH_INFO or any middle category. Every evidence must be classified as either SUPPORTS or REFUTES based on RELEVANCE.

EVIDENCE ITEMS (10 TOTAL):
{evidence_text}

RESPONSE FORMAT (LABELS ONLY, ONE PER LINE):
Evidence 1: [SUPPORTS/REFUTES]
Evidence 2: [SUPPORTS/REFUTES]
Evidence 3: [SUPPORTS/REFUTES]
Evidence 4: [SUPPORTS/REFUTES]
Evidence 5: [SUPPORTS/REFUTES]
Evidence 6: [SUPPORTS/REFUTES]
Evidence 7: [SUPPORTS/REFUTES]
Evidence 8: [SUPPORTS/REFUTES]
Evidence 9: [SUPPORTS/REFUTES]
Evidence 10: [SUPPORTS/REFUTES]

IMPORTANT: Return ONLY labels. No explanations. No other text."""
        return prompt

    def _fallback_evaluate_binary(self, claim, evidence_items):
        """Fallback evaluation - classify based on content relevance"""
        logger.warning("⚠️  Using fallback binary evaluation")
        evaluations = []
        
        claim_lower = claim.lower()
        
        for item in evidence_items:
            content = str(item.get('content', '')).lower()
            
            # Simple relevance check: keyword overlap
            claim_words = set(claim_lower.split())
            content_words = set(content.split())
            overlap = len(claim_words & content_words) / max(len(claim_words), 1)
            
            label = "SUPPORTS" if overlap > 0.4 else "REFUTES"
            evaluations.append({"evidence": item, "label": label, "raw": "Fallback"})
        
        return evaluations

    def run(self, claim):
        """Run complete verification with binary classification"""
        logger.warning("📊 Verifying claim: %s", claim[:60])
        
        ranked_evidence, metadata = self.retrieve_and_rank_improved(claim)
        
        logger.warning("   → Using %d ranked evidence items", len(ranked_evidence))
        
        if not ranked_evidence:
            logger.warning("⚠️  No evidence after ranking")
            return {
                "retrieved": {"faiss": [], "web": []},
                "evaluations": [],
                "ranking_metadata": metadata
            }
        
        evaluations = self.batch_evaluate_evidence(claim, ranked_evidence)
        
        logger.warning("✅ Verification complete with %d evaluations", len(evaluations))
        
        return {
            "retrieved": {"ranked": ranked_evidence},
            "evaluations": evaluations,
            "ranking_metadata": metadata
        }