# backend/agents/verification_agent.py (FIXED)
"""Evidence retrieval with FIXED web result handling."""
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
    """Runs evidence retrieval with proper web result handling."""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.model = ADK_MODEL_NAME
        self.ranker = SemanticRanker()
        logger.warning("🔍 VerificationAgent initialized (top_k=%d) with advanced semantic ranking", top_k)

    def _run_web_search(self, claim):
        """Run Google Search with proper result handling."""
        try:
            logger.warning("🔍 Google Search (ADK Agent): %s", claim[:60])
            result = google_search_tool(query=claim, top_k=5)
            
            # FIXED: Ensure result is always a list
            if result is None:
                logger.warning("   ⚠️  Google search returned None")
                return []
            
            if not isinstance(result, list):
                logger.warning("   ⚠️  Google search returned non-list type: %s", type(result).__name__)
                # Try to convert to list
                if isinstance(result, dict):
                    result = [result]
                else:
                    return []
            
            logger.warning("   ✅ Found %d web search results", len(result))
            
            # Validate each result
            validated_results = []
            for i, res in enumerate(result, 1):
                if not isinstance(res, dict):
                    logger.warning("   ⚠️  [%d] Result is not dict (type: %s), skipping", i, type(res).__name__)
                    continue
                
                # Check required fields
                title = res.get('title', 'N/A')
                relevance_score = res.get('relevance_score', 0)
                content = res.get('content', '')
                url = res.get('url', '')
                
                if not (title or content or url):
                    logger.warning("   ⚠️  [%d] Result missing all fields, skipping", i)
                    continue
                
                logger.warning("   [%d] Title: %s | Relevance: %.2f | URL: %s", 
                             i, title[:50], relevance_score, url[:40] if url else "N/A")
                
                # Ensure _source tag
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
        """Run FAISS search for the claim."""
        try:
            logger.warning("🔎 FAISS search: %s", claim[:60])
            result = faiss_search(claim, k=self.top_k)
            
            if not isinstance(result, list):
                logger.warning("   ⚠️  FAISS returned non-list type: %s", type(result).__name__)
                return []
            
            logger.warning("   ✅ Found %d FAISS results", len(result))
            return result
        except Exception as e:
            logger.warning("❌ FAISS search exception: %s", str(e)[:100])
            return []

    def retrieve_all(self, claim):
        """Run retrievals in parallel."""
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
                    logger.warning("❌ Exception in %s retrieval: %s", key, str(e)[:100])
                    results[key] = []
        
        logger.warning("✅ Retrieval complete - FAISS: %d, Web: %d", 
                      len(results['faiss']), len(results['web']))
        return results

    def retrieve_and_rank_improved(self, claim):
        """
        Retrieve from both sources and rank by semantic relevance with source boost.
        
        Returns: (ranked_evidence, metadata)
        """
        logger.warning("📊 Starting ADVANCED retrieve-and-rank process...")
        
        # Step 1: Retrieve from both sources
        retrieved = self.retrieve_all(claim)
        
        # Step 2: Combine and normalize results
        all_results = []
        
        # Add FAISS results
        for item in retrieved['faiss']:
            if isinstance(item, dict):
                item['_source'] = 'faiss'
                item['_freshness_boost'] = 0.0  # FAISS = old, no boost
            else:
                item = {"content": str(item), "_source": "faiss", "_freshness_boost": 0.0}
            all_results.append(item)
        
        # Add Google results (with freshness boost)
        for item in retrieved['web']:
            if isinstance(item, dict):
                item['_source'] = 'web'
                item['_freshness_boost'] = 0.25  # Web = fresh, +25% boost
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
        
        # Step 3: Compute semantic similarity and rank
        ranked_evidence, similarities, source_breakdown = self._rank_with_semantic_scoring(
            query=claim,
            evidence_items=all_results,
            top_k=self.top_k
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
                                    top_k: int = 5) -> tuple:
        """
        Rank evidence using semantic similarity + source freshness boost.
        
        Returns: (ranked_items, similarities, source_breakdown)
        """
        if not evidence_items:
            return [], [], {"faiss_selected": 0, "web_selected": 0}
        
        logger.warning("🔄 Computing semantic scores for %d items...", len(evidence_items))
        
        try:
            # Get query embedding
            query_embedding = self.ranker.embeddings.embed_query(query)
            import numpy as np
            query_vector = np.array(query_embedding)
            
            scored_items = []
            
            for i, item in enumerate(evidence_items):
                try:
                    # Extract content
                    content = item.get('content', str(item))
                    if isinstance(content, str):
                        content = content[:500]
                    else:
                        content = str(content)[:500]
                    
                    # Compute semantic similarity
                    content_embedding = self.ranker.embeddings.embed_query(content)
                    content_vector = np.array(content_embedding)
                    semantic_sim = self._cosine_similarity(query_vector, content_vector)
                    
                    # Apply freshness boost for web sources
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
            logger.warning("✅ All items ranked by score")
            
            # Select top K
            top_items = scored_items[:top_k]
            
            # Track source breakdown
            source_breakdown = {
                "faiss_selected": sum(1 for item in top_items if item['source'] == 'faiss'),
                "web_selected": sum(1 for item in top_items if item['source'] == 'web')
            }
            
            similarities = [item['final_score'] for item in top_items]
            evidence = [item['item'] for item in top_items]
            
            logger.warning("✅ Selected top %d items:", len(top_items))
            for i, (item, score) in enumerate(zip(top_items, similarities), 1):
                logger.warning("   %d. %s (final_score: %.3f)", i, item['source'], score)
            
            return evidence, similarities, source_breakdown
            
        except Exception as e:
            logger.exception("❌ Semantic scoring error: %s", e)
            logger.warning("⚠️  Fallback: returning first K items")
            # Fallback: just return first top_k items sorted by source (web first)
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
        """Batch evaluate ranked evidence."""
        if not evidence_items:
            logger.warning("❌ No evidence items to evaluate")
            return []
        
        evidence_items = evidence_items[:10]
        
        try:
            prompt = self._build_evaluation_prompt(claim, evidence_items)
            logger.warning("🔄 Batch evaluating %d evidence items", len(evidence_items))
            
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
        """Build evaluation prompt."""
        evidence_text = "\n\n".join([
            f"[Evidence {i}]\nSource: {item.get('_source', 'unknown')}\n"
            f"Content: {item.get('content', str(item))[:300]}"
            for i, item in enumerate(evidence_items, 1)
        ])
        
        prompt = f"""You are an expert fact-checker. Evaluate each evidence against the claim.

CLAIM: "{claim}"

EVIDENCE ITEMS:
{evidence_text}

RULES:
- SUPPORTS: Evidence confirms claim is TRUE
- REFUTES: Evidence shows claim is FALSE
- NOT_ENOUGH_INFO: Vague or doesn't address claim

Response format (labels only):
Evidence 1: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
Evidence 2: [SUPPORTS/REFUTES/NOT_ENOUGH_INFO]
..."""
        return prompt

    def _fallback_evaluate(self, claim, evidence_items):
        """Fallback evaluation."""
        logger.warning("⚠️  Using fallback evaluation")
        return [{"evidence": item, "label": "NOT_ENOUGH_INFO", "raw": "Fallback"} 
                for item in evidence_items]

    def run(self, claim):
        """Run complete verification."""
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