# backend/agents/claim_extraction_agent.py
"""Extracts top claim(s) from article text with minimal API calls."""
import re
from config import get_logger

logger = get_logger(__name__)


class ClaimExtractionAgent:
    """Extracts factual claims from article text WITHOUT API calls."""
    
    def __init__(self, max_claims: int = 3):
        self.max_claims = max_claims
        logger.warning("🎯 ClaimExtractionAgent initialized (max_claims=%d)", max_claims)

    def run(self, article_text: str) -> list:
        """
        Extract claims from text using simple heuristics (NO API calls).
        This saves API quota for verification instead.
        """
        logger.warning("📖 Extracting claims from text (length=%d)", len(article_text) if article_text else 0)
        
        if not article_text or len(article_text.strip()) == 0:
            logger.warning("⚠️  Empty input")
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?\n]', article_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            logger.warning("⚠️  No sentences found")
            return []
        
        # Take first N sentences as claims (they're usually the main assertions)
        claims = sentences[:self.max_claims]
        
        logger.warning("✅ Extracted %d claims using heuristics (NO API calls)", len(claims))
        for i, claim in enumerate(claims, 1):
            logger.warning("   %d. %s", i, claim[:70])
        
        return claims