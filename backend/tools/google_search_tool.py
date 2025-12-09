# ==============================================================================
# FILE 2: backend/tools/google_search_tool.py (REPLACE EXISTING)
# ==============================================================================

"""
Google Search Tool - Pure ADK FunctionTool using Gemini
"""
from google.adk.tools import FunctionTool
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
from typing import List, Dict
import json
import os

logger = get_logger(__name__)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


def search_google(
    query: str,
    top_k: int = 10
) -> List[Dict]:
    """
    Search Google for current information using Gemini.
    
    This tool performs real-time web searches to find current news,
    recent events, and up-to-date information. Use this for verifying
    claims about recent events or breaking news.
    
    Args:
        query: The claim or question to search for
        top_k: Maximum number of results to return (default: 10)
    
    Returns:
        List of search results with keys:
        - rank: Result ranking (1-based)
        - title: Result title
        - content: Brief summary or snippet
        - url: Source URL
        - relevance_score: Relevance score (0-1)
        - _source: Always "google_search"
    
    Note: Uses Gemini's built-in Google Search capability for accurate results.
    """
    logger.warning(f"🔍 Google Search Tool: Searching for '{query[:60]}'")
    
    try:
        prompt = f"""Search Google for information about: {query}

Return ONLY a JSON array with {top_k} results maximum:
[
  {{
    "rank": 1,
    "title": "Article Title",
    "content": "Brief summary (max 200 chars)",
    "url": "https://example.com",
    "relevance_score": 0.95
  }}
]

Be concise. Return ONLY valid JSON array, nothing else."""

        logger.warning("   📡 Calling Gemini with Google Search...")
        
        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        
        response_text = response.text if hasattr(response, 'text') else str(response)
        logger.warning("   📦 Response received")
        
        # Parse JSON response
        results = _parse_json_response(response_text, top_k)
        
        if results:
            logger.warning(f"   ✅ Found {len(results)} web results")
            for result in results:
                result["_source"] = "google_search"
            return results
        else:
            logger.warning("   ⚠️  No valid results parsed")
            return []
        
    except Exception as e:
        logger.warning(f"❌ Google Search error: {str(e)[:100]}")
        return []


def _parse_json_response(response_str: str, top_k: int) -> List[Dict]:
    """Parse JSON response from Gemini"""
    if not response_str:
        return []
    
    try:
        # Try direct JSON parse
        result = json.loads(response_str)
        if isinstance(result, list):
            return result[:top_k]
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown code blocks
    try:
        if "```json" in response_str:
            json_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
            json_str = response_str.split("```")[1].split("```")[0].strip()
        else:
            # Try finding JSON array
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
    
    logger.warning("   ⚠️  Could not parse JSON from response")
    return []


# Create ADK FunctionTool (correct API)
google_search_tool = FunctionTool(search_google)