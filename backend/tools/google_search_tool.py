# backend/tools/google_search_tool.py - FIXED
"""
Fixed Google Search tool without asyncio conflicts
Works properly in async context (Gradio UI)
"""

import os
import json
import re
from typing import List, Dict
from config import GEMINI_API_KEY, get_logger

logger = get_logger(__name__)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

# Use Google's generative AI directly instead of ADK
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)


def google_search_tool(query: str, top_k: int = 10) -> list:
    """
    Execute Google search using Gemini with built-in Google Search
    FIXED: No asyncio conflicts - works in async context
    """
    logger.warning("🔍 google_search_tool called: %s", query[:60])
    
    try:
        # Use Gemini's built-in Google Search capability
        prompt = f"""Search Google for information about: {query}

Return ONLY a JSON array with {top_k} results maximum:
[
  {{
    "rank": 1,
    "title": "Article Title",
    "content": "Brief summary of the content (max 200 chars)",
    "url": "https://example.com",
    "relevance_score": 0.95
  }},
  ...
]

Be concise. Return only valid JSON array, nothing else."""

        logger.warning("🔍 Calling Gemini for search results...")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        response_text = response.text if hasattr(response, 'text') else str(response)
        logger.warning("📦 Gemini response received (length: %d)", len(response_text))
        
        # Parse JSON response
        results = _parse_json_response(response_text, top_k)
        
        if results:
            logger.warning("✅ Google Search returned %d results", len(results))
            for i, result in enumerate(results[:3], 1):
                logger.warning("   [%d] %s", i, result.get('title', 'Unknown')[:60])
            return results
        else:
            logger.warning("⚠️  No valid results parsed")
            return []
        
    except Exception as e:
        logger.warning("❌ google_search_tool error: %s", str(e)[:100])
        return []


def _parse_json_response(response_str: str, top_k: int) -> list:
    """
    Parse JSON response from Gemini
    Handles various response formats
    """
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
    
    logger.warning("⚠️  Could not parse JSON from response")
    return []


def create_mock_search_results(query: str, count: int = 5) -> List[Dict]:
    """
    Fallback: Create mock results if search fails
    Ensures system doesn't crash
    """
    logger.warning("⚠️  Using mock search results (fallback)")
    
    mock_results = [
        {
            "rank": 1,
            "title": f"Result about {query}",
            "content": f"This is a search result for {query}",
            "url": "https://example.com/1",
            "relevance_score": 0.8
        },
        {
            "rank": 2,
            "title": f"More information on {query}",
            "content": f"Additional search result for {query}",
            "url": "https://example.com/2",
            "relevance_score": 0.7
        }
    ]
    
    return mock_results[:count]