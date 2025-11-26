"""Semantic ranking of evidence by relevance to query."""
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, get_logger
import numpy as np
from typing import List, Dict, Tuple

logger = get_logger(__name__)

class SemanticRanker:
    """Ranks evidence by semantic relevance to query."""
    
    def __init__(self):
        logger.warning("🔄 Initializing SemanticRanker...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.warning("✅ Embeddings model loaded: %s", EMBEDDING_MODEL)
        except Exception as e:
            logger.warning("❌ Error loading embeddings: %s", str(e)[:100])
            raise
    
    def rank_evidence_by_relevance(self, query: str, evidence_items: List[Dict], 
                                   top_k: int = 5, min_similarity: float = 0.5) -> Tuple[List[Dict], List[float]]:
        """
        Rank evidence items by semantic similarity to query.
        
        Args:
            query: The user's query/claim
            evidence_items: List of evidence dicts with 'content' and '_source' fields
            top_k: Number of top results to return (default: 5)
            min_similarity: Minimum similarity threshold to include (default: 0.5)
        
        Returns:
            (ranked_evidence, similarity_scores) - Top K results ranked by relevance
        """
        logger.warning("🔍 Ranking %d evidence items by relevance to query...", len(evidence_items))
        logger.warning("   Query: %s", query[:80])
        
        if not evidence_items:
            logger.warning("⚠️  No evidence items to rank")
            return [], []
        
        try:
            # Step 1: Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array(query_embedding)
            logger.warning("✅ Query embedding computed (dim: %d)", len(query_vector))
            
            # Step 2: Compute similarity for each evidence item
            ranked_items = []
            
            for i, item in enumerate(evidence_items):
                try:
                    # Extract content
                    content = item.get('content', str(item))
                    if isinstance(content, str):
                        content = content[:500]  # Limit length
                    else:
                        content = str(content)[:500]
                    
                    # Get content embedding
                    content_embedding = self.embeddings.embed_query(content)
                    content_vector = np.array(content_embedding)
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(query_vector, content_vector)
                    
                    # Store with source info for debugging
                    source = item.get('_source', 'unknown')
                    
                    ranked_items.append({
                        'item': item,
                        'similarity': float(similarity),
                        'source': source,
                        'index': i
                    })
                    
                    logger.warning("   [%d] %s: %.3f | %s", 
                                 i + 1, source, similarity, content[:60])
                    
                except Exception as e:
                    logger.warning("⚠️  Error processing item %d: %s", i, str(e)[:50])
                    continue
            
            # Step 3: Sort by similarity (descending)
            ranked_items.sort(key=lambda x: x['similarity'], reverse=True)
            logger.warning("✅ Items ranked by similarity")
            
            # Step 4: Apply minimum threshold
            filtered_items = [item for item in ranked_items if item['similarity'] >= min_similarity]
            logger.warning("✅ Filtered by threshold (%.2f): %d items remain", 
                         min_similarity, len(filtered_items))
            
            # Step 5: Apply tie-breaking (Google first if same similarity)
            self._apply_tiebreaker(filtered_items)
            
            # Step 6: Keep top K
            top_items = filtered_items[:top_k]
            similarities = [item['similarity'] for item in top_items]
            evidence = [item['item'] for item in top_items]
            
            logger.warning("✅ Selected top %d items:", len(top_items))
            for i, (item, sim) in enumerate(zip(top_items, similarities), 1):
                logger.warning("   %d. %s (similarity: %.3f)", i, item['source'], sim)
            
            return evidence, similarities
            
        except Exception as e:
            logger.exception("❌ Ranking error: %s", e)
            # Fallback: return all items with zero similarity
            logger.warning("⚠️  Fallback: returning original items (unsorted)")
            return evidence_items[:top_k], [0.0] * min(top_k, len(evidence_items))
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    @staticmethod
    def _apply_tiebreaker(items: List[Dict]):
        """Apply tiebreaker: Google sources come before FAISS for same similarity."""
        # Group by similarity score
        from itertools import groupby
        
        # Sort to group by similarity
        for similarity_score, group in groupby(items, key=lambda x: round(x['similarity'], 3)):
            group_list = list(group)
            
            # If multiple items with same similarity, prioritize by source
            if len(group_list) > 1:
                google_items = [x for x in group_list if x['source'] in ['web', 'google_search', 'google_search_fallback']]
                faiss_items = [x for x in group_list if x['source'] not in ['web', 'google_search', 'google_search_fallback']]
                
                # Re-order: Google first, then FAISS
                reordered = google_items + faiss_items
                
                # Update indices in original list
                start_idx = items.index(group_list[0])
                for i, item in enumerate(reordered):
                    items[start_idx + i] = item