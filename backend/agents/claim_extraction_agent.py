# backend/agents/claim_extraction_agent.py
"""Extracts meaningful claims from article text with better heuristics."""
import re
from config import get_logger

logger = get_logger(__name__)


class ClaimExtractionAgent:
    """Extracts factual claims from article text using improved heuristics."""
    
    def __init__(self, max_claims: int = 5):
        self.max_claims = max_claims
        logger.warning("🎯 ClaimExtractionAgent initialized (max_claims=%d)", max_claims)

    def run(self, article_text: str) -> list:
        """
        Extract meaningful claims from text using improved heuristics.
        Prioritizes declarative statements that can be fact-checked.
        """
        logger.warning("📖 Extracting claims from text (length=%d)", len(article_text) if article_text else 0)
        
        if not article_text or len(article_text.strip()) == 0:
            logger.warning("⚠️  Empty input")
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?\n]+', article_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            logger.warning("⚠️  No sentences found")
            return []
        
        # Filter for factual claims (exclude questions, opinions, etc.)
        claims = []
        for sentence in sentences:
            if self._is_factual_claim(sentence):
                claims.append(sentence)
                if len(claims) >= self.max_claims:
                    break
        
        # If no factual claims found, use first sentences anyway
        if not claims:
            claims = sentences[:self.max_claims]
        
        logger.warning("✅ Extracted %d claims", len(claims))
        for i, claim in enumerate(claims, 1):
            logger.warning("   %d. %s", i, claim[:80])
        
        return claims
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Check if a sentence is a factual claim that can be verified.
        Returns True if it looks like a verifiable statement.
        """
        
        # Reject questions
        if sentence.strip().endswith('?'):
            return False
        
        # Reject very short sentences
        if len(sentence.strip()) < 15:
            return False
        
        # Reject sentences that are mostly interrogative
        if sentence.lower().startswith(('what', 'why', 'how', 'when', 'where', 'who', 'which')):
            if sentence.strip().endswith('?'):
                return False
            # Some "what is..." statements are OK, but usually not
            if sentence.lower().startswith('what is'):
                return True
        
        # Reject overly opinionated language
        opinion_words = ['believe', 'think', 'feel', 'seems', 'might', 'could', 'should', 
                        'possibly', 'perhaps', 'apparently', 'allegedly']
        sentence_lower = sentence.lower()
        
        # Count opinion words - if too many, it's probably not a factual claim
        opinion_count = sum(1 for word in opinion_words if f' {word} ' in f' {sentence_lower} ')
        if opinion_count > 2:
            return False
        
        # Accept sentences with factual markers
        factual_markers = ['is', 'was', 'are', 'were', 'has', 'have', 'caused', 
                          'occurred', 'happened', 'found', 'discovered', 'measured',
                          'calculated', 'proved', 'shows', 'demonstrates']
        
        has_factual_marker = any(f' {marker} ' in f' {sentence_lower} ' for marker in factual_markers)
        
        return has_factual_marker or len(sentence) > 40  # Long sentences are usually claims