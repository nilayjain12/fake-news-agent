# backend/agents/claim_extraction_agent.py
"""LLM-based claim extraction agent with comprehensive instructions."""
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import json
import os

logger = get_logger(__name__)

# Set up the Gemini client
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


class ClaimExtractionAgent:
    """Extracts factual claims from text using LLM with intelligent reasoning."""
    
    def __init__(self, max_claims: int = 1):
        self.max_claims = max_claims
        self.model = ADK_MODEL_NAME
        logger.warning("🎯 ClaimExtractionAgent (LLM-based) initialized (max_claims=%d)", max_claims)

    def _build_extraction_prompt(self, text: str) -> str:
        """Build instruction prompt that summarizes into ONE claim."""
        prompt = f"""You are an expert fact-checker. Your task is to analyze the given input and extract the PRIMARY/MAIN claim as a single, clear, verifiable statement.

    INSTRUCTIONS:
    1. **Identify Main Claim**: Find the central factual assertion in the input
    2. **Summarize**: Combine related information into ONE comprehensive claim
    3. **Exclude Sub-details**: Don't break into separate claims for date, location, age, etc.
    4. **Preserve Intent**: Keep the core meaning of what user is asking
    5. **Make Self-Contained**: Include enough context to be understood independently

    IMPORTANT:
    - Return ONLY ONE main claim (not multiple)
    - If input has multiple unrelated claims, pick the PRIMARY one
    - Sub-details (date, location, cause) should be PART OF the main claim
    - Do NOT decompose into granular facts

    INPUT TEXT:
    "{text}"

    RESPONSE FORMAT:
    Return ONLY a JSON object (no markdown, no extra text):
    {{
    "main_claim": "The single, comprehensive, verifiable claim",
    "claim_type": "person/event/organization/statistic/other",
    "confidence_in_input": "high/medium/low",
    "summary": "One sentence explaining the claim"
    }}

    EXAMPLES:
    Input: "Dharmendra died recently"
    Output: {{"main_claim": "Bollywood actor Dharmendra died recently", "claim_type": "person", ...}}

    Input: "The Great Wall is visible from space with naked eye"
    Output: {{"main_claim": "The Great Wall of China is visible from space with naked eye", "claim_type": "fact", ...}}

    CRITICAL: Return ONLY valid JSON, nothing else."""
        return prompt

    def run(self, article_text: str) -> list:
        """Extract SINGLE main claim instead of multiple sub-claims."""
        logger.warning("📖 Extracting main claim from text (length=%d)", len(article_text) if article_text else 0)
        
        if not article_text or len(article_text.strip()) == 0:
            logger.warning("⚠️  Empty input")
            return []
        
        try:
            prompt = self._build_extraction_prompt(article_text[:2000])
            
            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            logger.warning("📝 Raw extraction response: %s", response_text[:200])
            
            try:
                # Parse JSON
                json_str = response_text
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                
                parsed = json.loads(json_str.strip())
                
                # Extract the SINGLE main claim
                main_claim = parsed.get("main_claim", "")
                
                if main_claim.strip():
                    logger.warning("✅ Extracted main claim: %s", main_claim[:80])
                    # Return as single-item list for compatibility
                    return [main_claim]
                else:
                    logger.warning("⚠️  Empty claim extracted")
                    return []
                
            except json.JSONDecodeError as e:
                logger.warning("⚠️  Could not parse JSON: %s", str(e)[:50])
                if response_text.strip():
                    logger.warning("⚠️  Using response as fallback claim")
                    return [response_text[:500]]
                return []
        
        except Exception as e:
            logger.warning("❌ Error extracting claim: %s", str(e)[:100])
            return self._fallback_extraction(article_text)
    
    def _fallback_extraction(self, text: str) -> list:
        """Fallback: Return first sentence as the claim."""
        logger.warning("⚠️  Using fallback extraction")
        import re
        
        sentences = re.split(r'[.!?\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if sentences:
            claim = sentences[0]
            logger.warning("✅ Fallback claim: %s", claim[:80])
            return [claim]
        
        return []
    
    def _is_likely_claim(self, sentence: str) -> bool:
        """Quick check if sentence is likely a factual claim."""
        if sentence.strip().endswith('?'):
            return False
        
        factual_markers = ['is', 'was', 'are', 'were', 'has', 'have', 'caused', 
                          'occurred', 'happened', 'found', 'discovered', 'measured',
                          'calculated', 'proved', 'shows', 'demonstrates']
        
        return any(f' {marker} ' in f' {sentence.lower()} ' for marker in factual_markers)