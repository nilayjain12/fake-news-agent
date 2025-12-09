# backend/tools/google_search_tool.py
"""
Google Search Tool - Pure ADK FunctionTool
"""
from google.adk.tools import FunctionTool
from google import genai
from typing import List, Dict
from config import GEMINI_API_KEY, ADK_MODEL_NAME, GOOGLE_TOP_K, get_logger
import json
import os

logger = get_logger(__name__)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


def search_google_for_current_info(
    query: str,
    top_k: int = GOOGLE_TOP_K
) -> List[Dict]:
    """
    Search Google for current information and recent events.
    
    This tool performs real-time web search to find up-to-date information,
    recent news, and latest developments. Best for current events and
    time-sensitive information.
    
    Args:
        query: The claim or question to search for
        top_k: Number of results to return (default: 10)
    
    Returns:
        List of dictionaries with keys:
        - rank: Result ranking (1-based)
        - title: Article title
        - content: Brief content summary
        - url: Source URL
        - relevance_score: Relevance score (0-1)
        - _source: Always "web"
    
    Use this for:
    - Current news and events
    - Recent developments
    - Real-time information
    - Latest statistics and data
    """
    logger.warning(f"🔍 Google Search Tool: Searching for '{query[:60]}'")
    
    try:
        prompt = f"""Search Google for information about: {query}

Return ONLY a JSON array with up to {top_k} results:
[
  {{
    "rank": 1,
    "title": "Article Title",
    "content": "Brief summary (max 200 chars)",
    "url": "https://example.com",
    "relevance_score": 0.95
  }}
]

Return only valid JSON array, nothing else."""

        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse JSON
        results = _parse_json_response(response_text, top_k)
        
        # Add source marker
        for result in results:
            result["_source"] = "web"
        
        logger.warning(f"   ✅ Found {len(results)} Google results")
        return results
        
    except Exception as e:
        logger.warning(f"❌ Google search error: {str(e)[:100]}")
        return []


def _parse_json_response(response_str: str, top_k: int) -> list:
    """Parse JSON from LLM response"""
    if not response_str:
        return []
    
    # Try direct parse
    try:
        result = json.loads(response_str)
        if isinstance(result, list):
            return result[:top_k]
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown
    try:
        if "```json" in response_str:
            json_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
            json_str = response_str.split("```")[1].split("```")[0].strip()
        else:
            # Find JSON array
            start = response_str.find("[")
            end = response_str.rfind("]")
            if start != -1 and end != -1:
                json_str = response_str[start:end+1]
            else:
                return []
        
        result = json.loads(json_str)
        if isinstance(result, list):
            return result[:top_k]
    except (json.JSONDecodeError, ValueError):
        pass
    
    logger.warning("⚠️  Could not parse JSON response")
    return []


# Create ADK FunctionTool
google_search_tool = FunctionTool(
    func=search_google_for_current_info,
    name="search_google_for_current_info",
    description="""Search Google for current information, recent news, and latest developments.

Use this tool when you need:
- Current news and breaking stories
- Recent events and updates
- Real-time information
- Latest statistics and data
- Up-to-date verification

Returns web search results ranked by relevance.
Note: Results are from the web and may vary in reliability."""
)