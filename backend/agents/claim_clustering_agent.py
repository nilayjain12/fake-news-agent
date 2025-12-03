# backend/agents/claim_clustering_agent.py
"""
Claim Clustering Agent - Groups semantically similar claims together
Reduces redundant verification by clustering related claims before processing
"""

from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
import numpy as np
from config import get_logger, EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)


class ClaimClusteringAgent:
    """
    Clusters semantically similar claims to reduce redundant verifications.
    
    Uses DBSCAN clustering on claim embeddings to group related claims.
    Then generates a comprehensive summary claim for each cluster.
    
    Example:
        Input: ["Messi scored 2 goals", "Lionel Messi netted twice", "Bitcoin hit $50K", 
                "BTC reached 50000 dollars", "Ronaldo injured"]
        Output: ["Lionel Messi scored 2 goals in the match", 
                 "Bitcoin price reached $50,000",
                 "Cristiano Ronaldo sustained an injury"]
    """
    
    def __init__(self, eps: float = 0.35, min_samples: int = 1):
        """
        Initialize clustering agent.
        
        Args:
            eps: Maximum distance between claims to be considered similar (0.35 = ~70% similarity)
            min_samples: Minimum claims in a cluster (1 = allow single-claim clusters)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.warning("🔗 ClaimClusteringAgent initialized (eps=%.2f, min_samples=%d)", eps, min_samples)
    
    def cluster_and_summarize(self, claims: List[str]) -> Tuple[List[str], Dict]:
        """
        Main entry point: Cluster claims and generate summaries.
        
        Args:
            claims: List of extracted claims
            
        Returns:
            Tuple of (summarized_claims, metadata)
            - summarized_claims: List of comprehensive summary claims
            - metadata: Dict with clustering statistics
        """
        if not claims:
            logger.warning("⚠️  No claims to cluster")
            return [], {"original_count": 0, "clustered_count": 0, "clusters": []}
        
        if len(claims) == 1:
            logger.warning("📌 Single claim - no clustering needed")
            return claims, {
                "original_count": 1,
                "clustered_count": 1,
                "clusters": [{"size": 1, "claims": claims}]
            }
        
        logger.warning("🔗 Clustering %d claims...", len(claims))
        
        # Step 1: Generate embeddings for all claims
        embeddings = self._generate_embeddings(claims)
        
        # Step 2: Cluster claims by semantic similarity
        clusters = self._cluster_claims(embeddings, claims)
        
        # Step 3: Generate summary for each cluster
        summarized_claims = self._generate_summaries(clusters)
        
        # Metadata
        metadata = {
            "original_count": len(claims),
            "clustered_count": len(summarized_claims),
            "reduction_percentage": (1 - len(summarized_claims) / len(claims)) * 100,
            "clusters": [
                {
                    "size": len(cluster),
                    "claims": cluster,
                    "summary": summary
                }
                for cluster, summary in zip(clusters, summarized_claims)
            ]
        }
        
        logger.warning("✅ Clustering complete: %d → %d claims (%.0f%% reduction)",
                      len(claims), len(summarized_claims), metadata["reduction_percentage"])
        
        return summarized_claims, metadata
    
    def _generate_embeddings(self, claims: List[str]) -> np.ndarray:
        """Generate embeddings for all claims."""
        logger.warning("   📊 Generating embeddings...")
        
        try:
            embeddings_list = self.embeddings.embed_documents(claims)
            embeddings = np.array(embeddings_list)
            logger.warning("   ✅ Generated %d embeddings (dim: %d)", len(embeddings), embeddings.shape[1])
            return embeddings
        except Exception as e:
            logger.warning("   ❌ Embedding error: %s", str(e)[:100])
            raise
    
    def _cluster_claims(self, embeddings: np.ndarray, claims: List[str]) -> List[List[str]]:
        """
        Cluster claims using DBSCAN.
        
        DBSCAN advantages:
        - No need to specify number of clusters
        - Handles noise (outlier claims)
        - Finds arbitrarily shaped clusters
        """
        logger.warning("   🔍 Running DBSCAN clustering...")
        
        try:
            # Use cosine distance (1 - cosine_similarity)
            clustering = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='cosine'
            )
            
            labels = clustering.fit_predict(embeddings)
            
            # Group claims by cluster label
            clusters_dict = {}
            for claim, label in zip(claims, labels):
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(claim)
            
            # Convert to list of clusters
            clusters = list(clusters_dict.values())
            
            logger.warning("   ✅ Found %d clusters:", len(clusters))
            for i, cluster in enumerate(clusters, 1):
                logger.warning("      Cluster %d: %d claims", i, len(cluster))
                for claim in cluster:
                    logger.warning("         - %s", claim[:60])
            
            return clusters
            
        except Exception as e:
            logger.warning("   ❌ Clustering error: %s", str(e)[:100])
            # Fallback: each claim is its own cluster
            return [[claim] for claim in claims]
    
    def _generate_summaries(self, clusters: List[List[str]]) -> List[str]:
        """
        Generate comprehensive summary for each cluster.
        
        Uses LLM to merge related claims into a single, comprehensive claim.
        """
        logger.warning("   📝 Generating summaries for %d clusters...", len(clusters))
        
        summaries = []
        
        for i, cluster in enumerate(clusters, 1):
            if len(cluster) == 1:
                # Single claim - no summarization needed
                summary = cluster[0]
                logger.warning("      Cluster %d: Single claim (no summary needed)", i)
            else:
                # Multiple claims - generate comprehensive summary
                logger.warning("      Cluster %d: Summarizing %d claims...", i, len(cluster))
                summary = self._summarize_cluster_with_llm(cluster)
            
            summaries.append(summary)
        
        logger.warning("   ✅ Generated %d summaries", len(summaries))
        return summaries
    
    def _summarize_cluster_with_llm(self, cluster_claims: List[str]) -> str:
        """
        Use LLM to generate a comprehensive summary of related claims.
        
        This creates a single, well-formed claim that captures all information
        from the cluster without redundancy.
        """
        from google import genai
        import os
        from config import GEMINI_API_KEY, ADK_MODEL_NAME
        
        os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(cluster_claims)])
        
        prompt = f"""You are a claim consolidation expert. You have {len(cluster_claims)} related claims that need to be merged into ONE comprehensive claim.

RELATED CLAIMS:
{claims_text}

TASK:
Merge these claims into a SINGLE, comprehensive claim that:
1. Captures ALL unique information from the claims
2. Removes redundancy and repetition
3. Is clear, specific, and verifiable
4. Preserves key details (names, numbers, dates, locations)
5. Uses proper grammar and natural language

IMPORTANT:
- Return ONLY the merged claim (one sentence or paragraph)
- Do NOT add introductory text like "Here is the summary:"
- Do NOT add explanations or metadata
- Make it self-contained and complete

MERGED CLAIM:"""
        
        try:
            response = client.models.generate_content(
                model=ADK_MODEL_NAME,
                contents=prompt
            )
            
            summary = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            
            # Clean up any unwanted prefixes
            unwanted_prefixes = [
                "Here is the summary:",
                "Here is the merged claim:",
                "Merged claim:",
                "Summary:",
                "The merged claim is:",
            ]
            
            for prefix in unwanted_prefixes:
                if summary.lower().startswith(prefix.lower()):
                    summary = summary[len(prefix):].strip()
            
            logger.warning("         Summary: %s", summary[:80])
            return summary
            
        except Exception as e:
            logger.warning("         ⚠️  LLM summary failed: %s", str(e)[:50])
            # Fallback: concatenate claims with "and"
            fallback = " and ".join(cluster_claims)
            return fallback if len(fallback) < 500 else cluster_claims[0]
    
    def estimate_api_savings(self, original_count: int, clustered_count: int) -> Dict:
        """
        Estimate API call and cost savings from clustering.
        
        Typical verification pipeline:
        - Per claim: 1 claim extraction + 2 searches + 1 evaluation + 1 aggregation + 1 report = 6 API calls
        """
        calls_per_claim = 6
        
        original_calls = original_count * calls_per_claim
        clustered_calls = clustered_count * calls_per_claim
        
        savings = {
            "original_claims": original_count,
            "clustered_claims": clustered_count,
            "original_api_calls": original_calls,
            "clustered_api_calls": clustered_calls,
            "api_calls_saved": original_calls - clustered_calls,
            "percentage_saved": (1 - clustered_calls / original_calls) * 100 if original_calls > 0 else 0,
            "estimated_time_saved_seconds": (original_calls - clustered_calls) * 1.5  # ~1.5s per API call
        }
        
        return savings


# Convenience function for easy import
def cluster_claims(claims: List[str]) -> Tuple[List[str], Dict]:
    """
    Cluster and summarize claims.
    
    Usage:
        claims = ["Claim 1", "Claim 2", "Similar to Claim 1", "Claim 3"]
        summarized, metadata = cluster_claims(claims)
        # Result: ["Merged Claim 1+3", "Claim 2", "Claim 3"]
    """
    agent = ClaimClusteringAgent()
    return agent.cluster_and_summarize(claims)