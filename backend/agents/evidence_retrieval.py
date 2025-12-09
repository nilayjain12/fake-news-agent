# backend/agents/evidence_retrieval.py
"""
Evidence Retrieval Agent - Pure ADK Custom Agent
Performs parallel retrieval from FAISS + Google Search
"""
from google.adk.agents import BaseAgent
from typing import AsyncIterator, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import FAISS_TOP_K, GOOGLE_TOP_K, FINAL_TOP_K, get_logger

logger = get_logger(__name__)


class EvidenceRetrievalAgent(BaseAgent):
    """
    Custom ADK Agent for parallel evidence retrieval
    
    Workflow:
    1. Read claim from session.state["extracted_claim"]
    2. Parallel retrieval from FAISS (5) + Google (10) = 15 total
    3. Rank by semantic relevance
    4. Select top 10 most relevant
    5. Store in session.state["evidence"]
    
    This demonstrates ADK Custom Agent pattern with:
    - Parallel tool execution
    - Session state management
    """

    def __init__(self):
        super().__init__(name="evidence_retrieval_agent")
        logger.warning("✅ Evidence Retrieval Agent created (ADK Custom Agent)")
    
    async def run_async(self, ctx) -> AsyncIterator[str]:
        """Execute parallel evidence retrieval"""
        logger.warning("🔍 Evidence Retrieval: Starting")
        
        # Get claim from session state (set by extraction agent)
        extracted = ctx.state.get("extracted_claim", {})
        claim = extracted.get("main_claim", "")
        
        if not claim:
            logger.warning("⚠️  No claim found in session state")
            yield "Error: No claim to verify"
            return
        
        yield f"Retrieving evidence for: {claim[:80]}..."
        
        # Parallel retrieval from both sources
        faiss_results = []
        google_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._retrieve_faiss, claim): "faiss",
                executor.submit(self._retrieve_google, claim): "google"
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    if source == "faiss":
                        faiss_results = future.result()
                    else:
                        google_results = future.result()
                except Exception as e:
                    logger.warning(f"❌ {source} retrieval failed: {str(e)[:100]}")
        
        yield f"Retrieved {len(faiss_results)} FAISS + {len(google_results)} Google results"
        
        # Combine and rank
        all_evidence = faiss_results + google_results
        
        if not all_evidence:
            logger.warning("⚠️  No evidence retrieved from any source")
            ctx.state["evidence"] = {
                "ranked": [],
                "total_count": 0,
                "faiss_count": 0,
                "google_count": 0
            }
            yield "Warning: No evidence found"
            return
        
        # Rank by semantic relevance
        ranked_evidence = self._rank_evidence(claim, all_evidence)
        
        # Store in session state
        ctx.state["evidence"] = {
            "ranked": ranked_evidence,
            "total_count": len(all_evidence),
            "faiss_count": len(faiss_results),
            "google_count": len(google_results),
            "final_count": len(ranked_evidence)
        }
        
        yield f"Ranked and selected top {len(ranked_evidence)} most relevant evidence items"
        
        logger.warning(f"✅ Evidence retrieval complete: {len(ranked_evidence)} items")
    
    def _retrieve_faiss(self, claim: str) -> List[Dict]:
        """Retrieve from FAISS knowledge base"""
        try:
            from tools.faiss_tool import search_faiss_knowledge_base
            results = search_faiss_knowledge_base(claim, k=FAISS_TOP_K)
            logger.warning(f"   FAISS: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"❌ FAISS error: {str(e)[:100]}")
            return []
    
    def _retrieve_google(self, claim: str) -> List[Dict]:
        """Retrieve from Google Search"""
        try:
            from tools.google_search_tool import search_google_for_current_info
            results = search_google_for_current_info(claim, top_k=GOOGLE_TOP_K)
            logger.warning(f"   Google: {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"❌ Google error: {str(e)[:100]}")
            return []
    
    def _rank_evidence(self, claim: str, evidence_items: List[Dict]) -> List[Dict]:
        """
        Rank evidence by semantic relevance
        Uses existing semantic ranker with multi-factor scoring
        """
        if not evidence_items:
            return []
        
        try:
            from tools.semantic_ranker import SemanticRanker
            
            ranker = SemanticRanker()
            ranked, scores, metadata = ranker._advanced_rank_with_multi_factor_scoring(
                query=claim,
                evidence_items=evidence_items,
                top_k=FINAL_TOP_K,
                min_similarity=0.5
            )
            
            logger.warning(f"   Ranking: {len(evidence_items)} → {len(ranked)} items")
            return ranked
            
        except Exception as e:
            logger.warning(f"⚠️  Ranking failed: {str(e)[:100]}, using first {FINAL_TOP_K}")
            return evidence_items[:FINAL_TOP_K]


def create_evidence_retrieval_agent() -> EvidenceRetrievalAgent:
    """Factory function to create evidence retrieval agent"""
    return EvidenceRetrievalAgent()
