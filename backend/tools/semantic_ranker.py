# backend/tools/semantic_ranker.py (IMPROVED)
"""Advanced semantic ranking with source-aware relevance scoring."""
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, get_logger
import numpy as np
from typing import List, Dict, Tuple

logger = get_logger(__name__)

class SemanticRanker:
    """Advanced semantic ranking with multi-factor scoring."""
    
    def __init__(self):
        logger.warning("🔄 Initializing Advanced SemanticRanker...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.warning("✅ Embeddings model loaded: %s", EMBEDDING_MODEL)
        except Exception as e:
            logger.warning("❌ Error loading embeddings: %s", str(e)[:100])
            raise
    
    def rank_evidence_by_relevance(self, query: str, evidence_items: List[Dict], 
                                   top_k: int = 5, min_similarity: float = 0.5) -> Tuple[List[Dict], List[float]]:
        """
        LEGACY METHOD - Kept for backwards compatibility.
        Now delegates to advanced ranking.
        """
        logger.warning("🔍 Using legacy rank_evidence_by_relevance (redirects to advanced)")
        
        results, scores, _ = self._advanced_rank_with_multi_factor_scoring(
            query=query,
            evidence_items=evidence_items,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        return results, scores
    
    def _advanced_rank_with_multi_factor_scoring(self, query: str, evidence_items: List[Dict],
                                                 top_k: int = 5, min_similarity: float = 0.5) -> Tuple[List[Dict], List[float], Dict]:
        """
        ADVANCED: Multi-factor scoring combining:
        1. Semantic similarity to query
        2. Source freshness (web > FAISS)
        3. Source reliability (explicit contradictions weighted higher)
        4. Content quality (length and completeness)
        
        Returns: (ranked_evidence, scores, metadata)
        """
        logger.warning("🔄 Running ADVANCED multi-factor ranking for %d items...", len(evidence_items))
        
        if not evidence_items:
            logger.warning("⚠️  No evidence items to rank")
            return [], [], {}
        
        try:
            # FACTOR 1: Compute query embedding once
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array(query_embedding)
            logger.warning("✅ Query embedding computed (dim: %d)", len(query_vector))
            
            scored_items = []
            
            for i, item in enumerate(evidence_items):
                try:
                    # Extract content
                    content = item.get('content', str(item))
                    if isinstance(content, str):
                        content = content[:500]
                    else:
                        content = str(content)[:500]
                    
                    # FACTOR 1: Semantic Similarity (0-1)
                    content_embedding = self.embeddings.embed_query(content)
                    content_vector = np.array(content_embedding)
                    semantic_sim = self._cosine_similarity(query_vector, content_vector)
                    
                    # FACTOR 2: Source Freshness Boost
                    source = item.get('_source', 'unknown')
                    freshness_boost = self._get_freshness_boost(source)
                    
                    # FACTOR 3: Content Quality Score (based on length and completeness)
                    content_quality = self._assess_content_quality(content)
                    
                    # FACTOR 4: Source Reliability (web = more current, FAISS = potentially outdated)
                    source_reliability = self._get_source_reliability(source)
                    
                    # Combine all factors with weighted formula
                    final_score = (
                        semantic_sim * 0.50 +           # 50% weight on semantic relevance
                        freshness_boost * 0.25 +        # 25% weight on freshness
                        content_quality * 0.15 +        # 15% weight on content quality
                        source_reliability * 0.10       # 10% weight on source reliability
                    )
                    
                    scored_items.append({
                        'item': item,
                        'semantic_sim': float(semantic_sim),
                        'freshness_boost': float(freshness_boost),
                        'content_quality': float(content_quality),
                        'source_reliability': float(source_reliability),
                        'final_score': float(final_score),
                        'source': source,
                        'index': i
                    })
                    
                    logger.warning("   [%d] %s: sem=%.2f fresh=%.2f qual=%.2f rel=%.2f → FINAL=%.3f",
                                 i + 1, source, semantic_sim, freshness_boost, 
                                 content_quality, source_reliability, final_score)
                    
                except Exception as e:
                    logger.warning("⚠️  Error scoring item %d: %s", i, str(e)[:50])
                    continue
            
            # Sort by final score
            scored_items.sort(key=lambda x: x['final_score'], reverse=True)
            logger.warning("✅ All %d items ranked", len(scored_items))
            
            # Filter by minimum similarity threshold
            filtered_items = [item for item in scored_items if item['final_score'] >= min_similarity]
            logger.warning("✅ Filtered by threshold (%.2f): %d items remain",
                         min_similarity, len(filtered_items))
            
            # Select top K
            top_items = filtered_items[:top_k]
            scores = [item['final_score'] for item in top_items]
            evidence = [item['item'] for item in top_items]
            
            # Metadata
            metadata = {
                "total_scored": len(scored_items),
                "after_threshold": len(filtered_items),
                "final_selected": len(top_items),
                "top_item_source": top_items[0]['source'] if top_items else 'none',
                "top_item_score": top_items[0]['final_score'] if top_items else 0.0,
                "source_breakdown": {
                    "web": sum(1 for item in top_items if item['source'] == 'web'),
                    "faiss": sum(1 for item in top_items if item['source'] == 'faiss'),
                }
            }
            
            logger.warning("✅ Final selection:")
            for i, item in enumerate(top_items, 1):
                logger.warning("   %d. %s (score: %.3f)", i, item['source'], item['final_score'])
            
            return evidence, scores, metadata
            
        except Exception as e:
            logger.exception("❌ Ranking error: %s", e)
            logger.warning("⚠️  Fallback: returning first K items")
            return evidence_items[:top_k], [0.0] * min(top_k, len(evidence_items)), {}
    
    def _get_freshness_boost(self, source: str) -> float:
        """
        Return freshness boost based on source.
        Web sources get significant boost for being current.
        """
        if source in ['web', 'google_search', 'google_search_fallback']:
            return 0.95  # Web results: maximum freshness
        elif source == 'faiss':
            return 0.3  # FAISS: potentially old, lower freshness
        else:
            return 0.5  # Unknown: neutral
    
    def _get_source_reliability(self, source: str) -> float:
        """
        Return reliability score.
        Web = more authoritative for current events.
        FAISS = outdated, lower reliability.
        """
        if source in ['web', 'google_search']:
            return 1.0  # Web: high reliability for current facts
        elif source == 'faiss':
            return 0.4  # FAISS: lower reliability (potential outdated data)
        else:
            return 0.5
    
    def _assess_content_quality(self, content: str) -> float:
        """
        Assess content quality based on:
        - Length (more detailed = better)
        - Presence of specific facts
        - Completeness
        """
        if not content:
            return 0.0
        
        content_len = len(content)
        
        # Scoring rubric
        if content_len < 50:
            quality = 0.3  # Very short, likely insufficient
        elif content_len < 200:
            quality = 0.6  # Short but acceptable
        elif content_len < 500:
            quality = 0.85  # Good length
        else:
            quality = 1.0  # Detailed content
        
        # Boost if content has numbers/dates (more specific)
        if any(char.isdigit() for char in content):
            quality = min(quality + 0.1, 1.0)
        
        return quality
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))