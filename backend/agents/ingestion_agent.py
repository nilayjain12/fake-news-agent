"""Simple ingestion agent to accept URL or raw text and return cleaned text."""
import requests
from bs4 import BeautifulSoup
from config import get_logger

logger = get_logger(__name__)

class IngestionAgent:
    def __init__(self):
        logger.debug("IngestionAgent initialized")

    def extract_text_from_url(self, url: str) -> str:
        """
        Minimal URL text extractor. For production use a robust extractor.
        """
        logger.info("Extracting text from URL: %s", url)
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # naive: join text paragraphs
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n\n".join(paragraphs)
            logger.debug("Extracted %d paragraphs from URL", len(paragraphs))
            return text
        except Exception as e:
            logger.exception("Failed to extract URL %s: %s", url, e)
            return ""

    def run(self, input_text_or_url: str) -> str:
        """
        If input looks like a URL, scrape; otherwise assume raw text.
        """
        if input_text_or_url.strip().startswith("http"):
            txt = self.extract_text_from_url(input_text_or_url)
            result = txt or input_text_or_url
            logger.debug("IngestionAgent.run returning text len=%d", len(result) if result else 0)
            return result
        logger.debug("IngestionAgent.run treating input as raw text len=%d", len(input_text_or_url))
        return input_text_or_url