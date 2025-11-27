# backend/tools/google_search_tool.py (SIMPLIFIED)
"""
Simplified Google Search tool - Returns JSON only.
"""

import os
import asyncio
import json
from google.adk.agents import Agent
from google.adk.tools import google_search
from config import GEMINI_API_KEY, get_logger
from google.adk.runners import InMemoryRunner

logger = get_logger(__name__)

# Set up API key
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY


def google_search_tool(query: str, top_k: int = 10) -> list:
    """
    Execute Google search using ADK Agent and return JSON results directly.
    
    Args:
        query: Search query string
        top_k: Number of results to return (default: 10)
    
    Returns:
        List of search result dictionaries with keys: rank, title, content, url, relevance_score
    """
    logger.warning("🔍 google_search_tool called: %s", query[:60])
    
    try:
        # Create the agent
        google_search_agent = Agent(
            name="GoogleSearchAgent",
            model='gemini-2.0-flash',
            tools=[google_search]
        )
        
        # Create the runner
        runner = InMemoryRunner(google_search_agent)

        # Prompt that FORCES JSON-only output
        prompt = f"""You are a search results formatter. Your ONLY job is to return valid JSON.

SEARCH QUERY: {query}

INSTRUCTIONS:
1. Use the google_search tool to find information
2. Format results as a JSON array
3. Return ONLY the JSON array, nothing else - no markdown, no explanations
4. Include {top_k} results maximum
5. Each result must have: rank, title, content, url, relevance_score

OUTPUT EXACTLY THIS FORMAT (no other text):
[{{"rank": 1, "title": "...", "content": "...", "url": "...", "relevance_score": 0.95}}, ...]"""

        logger.warning("🔍 Executing ADK agent search...")
        
        # Run the agent
        response = asyncio.run(runner.run_debug(prompt))
        
        # Extract the agent's final output
        agent_output = response.final_output if hasattr(response, 'final_output') else str(response)
        
        logger.warning("📦 Raw agent output type: %s", type(agent_output).__name__)
        
        # Parse JSON from agent output
        results = _parse_json_safely(agent_output)
        
        if not results:
            logger.warning("⚠️  No valid JSON found in agent output")
            return []
        
        logger.warning("✅ Google Search returned %d results", len(results))
        for i, result in enumerate(results[:5], 1):
            logger.warning("   [%d] %s | Score: %.2f", 
                         i, 
                         result.get('title', 'Unknown')[:50],
                         result.get('relevance_score', 0))
        
        return results
        
    except Exception as e:
        logger.warning("❌ google_search_tool error: %s", str(e)[:100])
        return []


def _parse_json_safely(response_str) -> list:
    """
    Safely parse JSON from response.
    Converts response to string if needed, then extracts JSON array.
    """
    if not response_str:
        return []
    
    # Convert to string if not already
    if not isinstance(response_str, str):
        response_str = str(response_str)
    
    try:
        # Try direct JSON parse first
        logger.warning("🔄 Parsing JSON response...")
        result = json.loads(response_str)
        
        # Validate it's a list
        if isinstance(result, list):
            logger.warning("✅ Valid JSON array parsed")
            return result
        
        return []
        
    except json.JSONDecodeError:
        # If direct parse fails, try to extract JSON from string
        logger.warning("⚠️  Direct parse failed, attempting extraction...")
        
        # Remove markdown code blocks if present
        if "```json" in response_str:
            response_str = response_str.split("```json")[1].split("```")[0]
        elif "```" in response_str:
            response_str = response_str.split("```")[1].split("```")[0]
        
        try:
            result = json.loads(response_str.strip())
            if isinstance(result, list):
                logger.warning("✅ JSON extracted from markdown")
                return result
        except json.JSONDecodeError:
            pass
        
        logger.warning("❌ Could not parse JSON")
        return []