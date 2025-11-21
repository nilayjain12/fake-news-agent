# backend/tools/google_search_tool.py
"""Google Search tool wrapper for fact-checking with improved handling."""
from google import genai
from config import GEMINI_API_KEY, get_logger
import os
import re

logger = get_logger(__name__)

# Set up the Gemini client
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


def google_search_tool(query: str, top_k: int = 5) -> list:
    """
    Search Google for information about a query using Gemini's grounding.
    Uses Google Search to retrieve current, factual information.
    
    Args:
        query: The search query
        top_k: Number of results to return
    
    Returns:
        List of search results with content
    """
    try:
        logger.warning("🔎 Google Search Tool: %s", query[:60])
        
        # Create a more direct prompt that gets factual information
        prompt = f"""Search the web for factual information about: {query}

Provide ONLY factual statements found in reliable sources. Format each fact on a new line.
Include the source or context where relevant.

Query: {query}

Factual information:"""
        
        # Use Gemini with Google Search grounding via system instruction
        # Note: generation_config is not supported in this API version
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        logger.warning("📝 Search response received (%d chars)", len(text))
        
        if not text or len(text.strip()) < 10:
            logger.warning("⚠️  Empty search response")
            return []
        
        # Parse the response into structured results
        results = []
        lines = text.split('\n')
        
        fact_count = 0
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Skip metadata lines
            if line.startswith("•") or line.startswith("-"):
                line = line[1:].strip()
            
            if line and len(line) > 5:
                results.append({
                    "rank": len(results) + 1,
                    "content": line,
                    "source": "google_search"
                })
                fact_count += 1
                
                if len(results) >= top_k:
                    break
        
        logger.warning("   → Extracted %d factual statements", fact_count)
        return results
        
    except Exception as e:
        logger.warning("❌ Google Search failed: %s", str(e)[:100])
        logger.warning("   Attempting fallback approach...")
        return _fallback_search(query, top_k)


def _fallback_search(query: str, top_k: int) -> list:
    """Fallback search when primary method fails."""
    try:
        logger.warning("🔄 Using fallback search approach")
        
        # Simpler prompt for fallback
        prompt = f"""What do you know about: {query}

Provide 3-5 key facts. Be concise and factual. List each fact on a new line."""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        results = []
        
        for i, line in enumerate(text.split('\n')[:top_k]):
            line = line.strip()
            if line and len(line) > 5:
                results.append({
                    "rank": i + 1,
                    "content": line,
                    "source": "google_search_fallback"
                })
        
        logger.warning("   → Fallback found %d results", len(results))
        return results
        
    except Exception as e:
        logger.warning("❌ Fallback search also failed: %s", str(e)[:50])
        return []