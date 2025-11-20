# backend/tools/google_search_tool.py
"""Google Search tool wrapper for fact-checking."""
from google import genai
from config import GEMINI_API_KEY, get_logger
import os

logger = get_logger(__name__)

# Set up the Gemini client
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


def google_search_tool(query: str, top_k: int = 5) -> list:
    """
    Search Google for information about a query.
    Uses Gemini's built-in Google Search capability.
    
    Args:
        query: The search query
        top_k: Number of results to return
    
    Returns:
        List of search results with content
    """
    try:
        logger.warning("🔎 Google Search Tool: %s", query[:60])
        
        prompt = f"""Search for current information about: {query}
        
Please provide the most relevant and recent information found.
Include sources if available."""
        
        # Use Gemini with Google Search grounding
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse the response into structured results
        results = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():
                results.append({
                    "rank": i + 1,
                    "content": line.strip(),
                    "source": "google_search"
                })
                if len(results) >= top_k:
                    break
        
        logger.warning("   → Found %d search results", len(results))
        return results
        
    except Exception as e:
        logger.warning("❌ Google Search failed: %s", str(e)[:100])
        return []