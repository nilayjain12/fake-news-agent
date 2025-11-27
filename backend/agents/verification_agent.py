# backend/agents/verification_agent.py (UPDATED)
"""Evidence retrieval with increased evidence collection: Google 10 + FAISS 5 = top 10 selected."""
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
    """Runs evidence retrieval with INCREASED evidence collection.
    
    Configuration:
    - Google Search: 10 evidences
    - FAISS: 5 evidences
    - Total combined: 15 evidences
    - Final selection: Top 10 ranked
    """
    
    def __init__(self, google_top_k: int = 10, faiss_top_k: int = 5, final_top_k: int = 10):
        """
        Initialize with configurable evidence retrieval.
        
        Args:
            google_top_k: Number of results from Google Search (default: 10)
            faiss_top_k: Number of results from FAISS (default: 5)
            final_top_k: Number of final ranked evidences to use (default: 10)
        """
        self.google_top_k = google_top_k
        self.faiss_top_k = faiss_top_k
        self.final_top_k = final_top_k
        self.model = ADK_MODEL_NAME
        self.ranker = SemanticRanker()
        logger.warning("🔍 VerificationAgent initialized")
        logger.warning("   Google Search: %d evidences", google_top_k)
        logger.warning("   FAISS: %d evidences", faiss_top_k)
        logger.warning("   Final Selection: top %d from combined", final_top_k)

    def _run_web_search(self, claim):
        """Run Google Search with increased evidence collection (10 results)."""
        try:
            logger.warning("🔍 Google Search (ADK Agent): %s", claim[:60])
            # Increased from 5 to 10
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
            import traceback
            logger.warning("   Traceback: %s", traceback.format_exc()[:200])
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
        """
        Retrieve from both sources and rank top 10 by semantic relevance.
        
        Process:
        1. Retrieve 10 from Google + 5 from FAISS = 15 total
        2. Rank all 15 by semantic similarity + source boost
        3. Select top 10
        
        Returns: (ranked_evidence, metadata)
        """
        logger.warning("📊 Starting ADVANCED retrieve-and-rank process...")
        
        # STEP 1: Retrieve from both sources (Google 10 + FAISS 5)
        retrieved = self.retrieve_all(claim)
        
        # STEP 2: Combine and normalize results
        all_results = []
        
        # Add FAISS results (5)
        for item in retrieved['faiss']:
            if isinstance(item, dict):
                item['_source'] = 'faiss'
                item['_freshness_boost'] = 0.0
            else:
                item = {"content": str(item), "_source": "faiss", "_freshness_boost": 0.0}
            all_results.append(item)
        
        # Add Google results (10) with freshness boost
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
        
        # STEP 3: Rank all 15 and select top 10
        ranked_evidence, similarities, source_breakdown = self._rank_with_semantic_scoring(
            query=claim,
            evidence_items=all_results,
            top_k=self.final_top_k  # Select top 10 from 15
        )
        
        # Metadata
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
        """
        Rank evidence using semantic similarity + source freshness boost.
        
        Updated to handle 15 evidences → select top 10
        
        Returns: (ranked_items, similarities, source_breakdown)
        """
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
            
            # Sort by final score (descending)
            scored_items.sort(key=lambda x: x['final_score'], reverse=True)
            logger.warning("✅ All %d items ranked by score", len(scored_items))
            
            # Select top K (10 from 15)
            top_items = scored_items[:top_k]
            logger.warning("📊 Selected top %d items from %d", len(top_items), len(scored_items))
            
            # Track source breakdown
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
        """Batch evaluate ranked evidence (now 10 instead of 5)."""
        if not evidence_items:
            logger.warning("❌ No evidence items to evaluate")
            return []
        
        # Use all top 10 (instead of limiting to 5)
        evidence_items = evidence_items[:10]
        logger.warning("📊 Batch evaluating %d evidence items", len(evidence_items))
        
        try:
            prompt = self._build_evaluation_prompt(claim, evidence_items)
            
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
                
                if "REFUTE" in line_upper or ("FALSE" in line_upper and "NOT" not in line_upper):
                    label = "REFUTES"
                elif "SUPPORT" in line_upper and "NOT" not in line_upper:
                    label = "SUPPORTS"
                
                label_count[label] += 1
                evaluations.append({"evidence": item, "label": label, "raw": line.strip()})
                logger.warning("   Evidence %d: %s", i, label)
            
            logger.warning("✅ Evaluation complete - SUPPORTS: %d, REFUTES: %d, NOT_ENOUGH: %d", 
                          label_count["SUPPORTS"], label_count["REFUTES"], label_count["NOT_ENOUGH_INFO"])
            
            return evaluations
            
        except Exception as e:
            logger.warning("⚠️  Evaluation failed: %s", str(e)[:100])
            return self._fallback_evaluate(claim, evidence_items)

    def _build_evaluation_prompt(self, claim, evidence_items):
        """Build evaluation prompt for 10 evidence items."""
        evidence_text = "\n\n".join([
            f"[Evidence {i}]\nSource: {item.get('_source', 'unknown')}\n"
            f"Content: {item.get('content', str(item))[:300]}"
            for i, item in enumerate(evidence_items, 1)
        ])
        
        prompt = f"""You are an expert fact-checker. Evaluate each of the 10 evidence pieces against the claim.

CLAIM: "{claim}"

EVIDENCE ITEMS (10 TOTAL):
{evidence_text}

RULES:
- SUPPORTS: Evidence confirms claim is TRUE
- REFUTES: Evidence shows claim is FALSE
- NOT_ENOUGH_INFO: Vague or doesn't address claim

Response format (labels only, one per line):
Evidence 1: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 2: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 3: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 4: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 5: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 6: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 7: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 8: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 9: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 10: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]"""
        return prompt

    def _fallback_evaluate(self, claim, evidence_items):
        """Fallback evaluation."""
        logger.warning("⚠️  Using fallback evaluation")
        return [{"evidence": item, "label": "NOT_ENOUGH_INFO", "raw": "Fallback"} 
                for item in evidence_items]

    def run(self, claim):
        """Run complete verification with 10 evidences."""
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