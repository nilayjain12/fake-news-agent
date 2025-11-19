# backend/agents/ingestion_agent.py
"""Simple ingestion agent to accept URL or raw text and return cleaned text."""
import requests
from bs4 import BeautifulSoup
from config import get_logger

logger = get_logger(__name__)

class IngestionAgent:
    def __init__(self):
        pass

    def extract_text_from_url(self, url: str) -> str:
        """Minimal URL text extractor."""
        logger.warning("📥 Fetching URL: %s", url[:60])
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n\n".join(paragraphs)
            logger.warning("✅ Extracted %d paragraphs", len(paragraphs))
            return text
        except Exception as e:
            logger.warning("❌ Failed to extract URL: %s", str(e)[:50])
            return ""

    def run(self, input_text_or_url: str) -> str:
        """If input looks like a URL, scrape; otherwise assume raw text."""
        if input_text_or_url.strip().startswith("http"):
            return self.extract_text_from_url(input_text_or_url)
        return input_text_or_url