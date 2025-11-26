"""
Google Search tool wrapper using Google ADK Agent.
Proper implementation with correct method calls.
"""

import os
import asyncio
from google.adk.agents import Agent
from google.adk.tools import google_search
from config import GEMINI_API_KEY, get_logger
from google.adk.runners import InMemoryRunner
import json

logger = get_logger(__name__)

logger = get_logger(__name__)

# Set up API key
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY


# ==================== Public API ====================

def google_search_tool(query: str, top_k: int = 5) -> list:
    """
    Execute Google search using ADK Agent with google_search tool.
    Falls back to direct Gemini if Agent methods fail.
    
    Args:
        query: Search query string
        top_k: Number of results to return (default: 5)
    
    Returns:
        List of search results as dictionaries with proper structure
    """
    # Create the agent FIRST
    google_search_agent = Agent(
        name="GoogleSearchAgent",
        model='gemini-2.0-flash',
        tools=[google_search]  # Give the agent its tool
    )
    
    # THEN create the runner with the agent
    runner = InMemoryRunner(google_search_agent)

    prompt = f"""You have access to the Google Search tool. Your task is to search for information about this query and provide the results.

QUERY: "{query}"

INSTRUCTIONS:
1. Use your google_search tool to find latest information about the query
2. Return the search results in JSON format
3. Include up to {top_k} results
4. For each result include: title, snippet/content, url, and relevance

RESPONSE FORMAT:
Return ONLY a valid JSON array with this structure:
[
  {{
    "rank": 1,
    "title": "Result Title",
    "content": "snippet or summary",
    "url": "https://example.com",
    "relevance_score": 0.95
  }},
  {{
    "rank": 2,
    "title": "Another Result",
    "content": "snippet text",
    "url": "https://example.com",
    "relevance_score": 0.87
  }}
]

IMPORTANT:
- Return ONLY the JSON array, no other text
- Ensure all URLs are complete and valid
- Prioritize results that directly address the query"""

    full_prompt = prompt + "\n\nUser Claim: " + query

    logger.warning("🔍 google_search_tool called: %s", query[:60])
    
    try:
        # Run async function synchronously
        response = asyncio.run(runner.run_debug(full_prompt))
        
        # Parse the response - it may be an Event object or string
        results = _parse_agent_response(response)
        
        logger.warning("✅ Returning %d search results", len(results))
        return results
        
    except Exception as e:
        logger.warning("❌ google_search_tool error: %s", str(e)[:100])
        return []


def _parse_agent_response(response) -> list:
    """
    Parse the response from the ADK Agent runner.
    Handles various response types: Event objects, strings, dicts, lists.
    
    Args:
        response: The raw response from runner.run_debug()
        
    Returns:
        List of search result dictionaries
    """
    try:
        # If it's already a list of dicts, return as-is
        if isinstance(response, list):
            return _validate_results(response)
        
        # If it's a dict, wrap in list
        if isinstance(response, dict):
            if "results" in response:
                return _validate_results(response["results"])
            return _validate_results([response])
        
        # If it's an Event object or has agent_output, extract the output
        if hasattr(response, 'agent_output'):
            output = response.agent_output
            if isinstance(output, str):
                return _parse_json_string(output)
            elif isinstance(output, list):
                return _validate_results(output)
            elif isinstance(output, dict):
                if "results" in output:
                    return _validate_results(output["results"])
                return _validate_results([output])
        
        # If it's an Event with final_output
        if hasattr(response, 'final_output'):
            output = response.final_output
            if isinstance(output, str):
                return _parse_json_string(output)
            elif isinstance(output, list):
                return _validate_results(output)
            elif isinstance(output, dict):
                return _validate_results([output])
        
        # If it's a string, try to parse as JSON
        if isinstance(response, str):
            return _parse_json_string(response)
        
        # Last resort: convert to string and try to parse
        response_str = str(response)
        logger.warning("⚠️  Response is non-standard type (%s), attempting string parse", type(response).__name__)
        return _parse_json_string(response_str)
        
    except Exception as e:
        logger.warning("❌ Error parsing response: %s", str(e)[:100])
        return []


def _parse_json_string(response_str: str) -> list:
    """
    Extract and parse JSON from response string.
    Handles markdown code blocks and nested JSON.
    """
    try:
        # Try direct JSON parse
        try:
            result = json.loads(response_str)
            if isinstance(result, list):
                return _validate_results(result)
            elif isinstance(result, dict):
                if "results" in result:
                    return _validate_results(result["results"])
                return _validate_results([result])
        except json.JSONDecodeError:
            pass
        
        # Extract JSON from markdown code blocks
        if "```json" in response_str:
            json_str = response_str.split("```json")[1].split("```")[0].strip()
            result = json.loads(json_str)
            if isinstance(result, list):
                return _validate_results(result)
            elif isinstance(result, dict):
                if "results" in result:
                    return _validate_results(result["results"])
                return _validate_results([result])
        
        # Extract JSON from plain code blocks
        if "```" in response_str:
            json_str = response_str.split("```")[1].split("```")[0].strip()
            result = json.loads(json_str)
            if isinstance(result, list):
                return _validate_results(result)
            elif isinstance(result, dict):
                if "results" in result:
                    return _validate_results(result["results"])
                return _validate_results([result])
        
        # Look for JSON array pattern
        import re
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return _validate_results(result)
        
        logger.warning("❌ Could not find valid JSON in response")
        return []
        
    except Exception as e:
        logger.warning("❌ JSON parse error: %s", str(e)[:100])
        return []


def _validate_results(results: list) -> list:
    """
    Validate and normalize search results.
    Ensures each result has required fields.
    """
    validated = []
    
    for result in results:
        if not isinstance(result, dict):
            continue
        
        # Ensure required fields exist
        validated_result = {
            "rank": result.get("rank", len(validated) + 1),
            "title": result.get("title", "Unknown Source"),
            "content": result.get("content") or result.get("snippet", ""),
            "url": result.get("url", ""),
            "relevance_score": result.get("relevance_score", 0.5),
            "_source": "web"  # Mark as web source
        }
        
        # Only include if it has meaningful content
        if validated_result["url"] or validated_result["content"]:
            validated.append(validated_result)
    
    return validated