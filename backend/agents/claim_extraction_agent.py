"""Uses Gemini (ADK LLM wrapper) to extract top claim(s) from article text."""
from google.adk.models.google_llm import Gemini
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

# Instantiate model once
_llm = Gemini(model=ADK_MODEL_NAME)

class ClaimExtractionAgent:
    def __init__(self, model=_llm, max_claims: int = 3):
        self.model = model
        self.max_claims = max_claims
        logger.debug("ClaimExtractionAgent initialized max_claims=%d", max_claims)

    def run(self, article_text: str):
        """
        Prompt Gemini to extract the top factual claims from the article.
        Returns a list of claim strings.
        """
        logger.info("Extracting claims from article text len=%d", len(article_text) if article_text else 0)
        prompt = f"""Extract up to {self.max_claims} factual, checkable claims from the following article. Return JSON array of strings only.

Article:
{article_text}
"""
        try:
            resp = self.model.generate(prompt)
            # The response parsing depends on Gemini's API shape. Expect resp.text or resp.output.
            text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
            logger.debug("ClaimExtractionAgent model response (truncated): %s", (text or "")[:500])
            # Try to extract JSON array from output; fallback to simple split
            import json
            try:
                claims = json.loads(text)
                if isinstance(claims, list):
                    parsed = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
                    logger.info("Parsed %d claims from model output", len(parsed))
                    return parsed
            except Exception:
                # fallback: split by newline and take first N non-empty lines
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                logger.info("Fallback parsed %d claim-lines", len(lines))
                return lines[: self.max_claims]
        except Exception as e:
            logger.exception("ClaimExtractionAgent failed: %s", e)
            return []