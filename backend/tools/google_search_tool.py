# backend/tools/google_search_tool.py - IMPROVED JSON PARSING
"""
Robust Google Search tool with bulletproof JSON parsing
"""

import os
import asyncio
import json
import re
from typing import List, Dict
from google.adk.agents import Agent
from google.adk.tools import google_search
from config import GEMINI_API_KEY, get_logger
from google.adk.runners import InMemoryRunner

logger = get_logger(__name__)

os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY


def google_search_tool(query: str, top_k: int = 10) -> list:
    """
    Execute Google search using ADK Agent and return JSON results.
    IMPROVED: Bulletproof JSON parsing with multiple extraction strategies.
    """
    logger.warning("🔍 google_search_tool called: %s", query[:60])
    
    try:
        google_search_agent = Agent(
            name="GoogleSearchAgent",
            model='gemini-2.0-flash',
            tools=[google_search]
        )
        
        runner = InMemoryRunner(google_search_agent)

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
        
        response = asyncio.run(runner.run_debug(prompt))
        agent_output = response.final_output if hasattr(response, 'final_output') else str(response)
        
        logger.warning("📦 Raw agent output type: %s", type(agent_output).__name__)
        logger.warning("📦 Raw output length: %d chars", len(str(agent_output)))
        
        # Use BULLETPROOF parsing
        results = _parse_json_bulletproof(agent_output)
        
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
        import traceback
        logger.warning("Traceback: %s", traceback.format_exc()[:200])
        return []


def _parse_json_bulletproof(response_str) -> list:
    """
    BULLETPROOF JSON parsing with 6 extraction strategies in priority order.
    Handles all edge cases and malformed JSON.
    """
    if not response_str:
        logger.warning("⚠️  Empty response string")
        return []
    
    # Convert to string if not already
    response_str_clean = str(response_str).strip()
    
    logger.warning("🔄 Starting bulletproof JSON parsing (length: %d)", len(response_str_clean))
    
    # ==========================================
    # STRATEGY 1: Direct JSON parse (fastest)
    # ==========================================
    logger.warning("   Strategy 1: Direct JSON parse...")
    result = _strategy_direct_parse(response_str_clean)
    if result:
        logger.warning("   ✅ Strategy 1 succeeded")
        return result
    
    # ==========================================
    # STRATEGY 2: Extract from markdown code blocks
    # ==========================================
    logger.warning("   Strategy 2: Extract from markdown code blocks...")
    result = _strategy_markdown_extraction(response_str_clean)
    if result:
        logger.warning("   ✅ Strategy 2 succeeded")
        return result
    
    # ==========================================
    # STRATEGY 3: Find JSON array delimiters [ ]
    # ==========================================
    logger.warning("   Strategy 3: Find JSON array delimiters...")
    result = _strategy_array_delimiter(response_str_clean)
    if result:
        logger.warning("   ✅ Strategy 3 succeeded")
        return result
    
    # ==========================================
    # STRATEGY 4: Find JSON object braces { }
    # ==========================================
    logger.warning("   Strategy 4: Extract JSON objects...")
    result = _strategy_object_extraction(response_str_clean)
    if result:
        logger.warning("   ✅ Strategy 4 succeeded")
        return result
    
    # ==========================================
    # STRATEGY 5: Regex-based JSON extraction
    # ==========================================
    logger.warning("   Strategy 5: Regex-based extraction...")
    result = _strategy_regex_extraction(response_str_clean)
    if result:
        logger.warning("   ✅ Strategy 5 succeeded")
        return result
    
    # ==========================================
    # STRATEGY 6: Last resort - rebuild from text
    # ==========================================
    logger.warning("   Strategy 6: Last resort - rebuild from text...")
    result = _strategy_fallback_rebuild(response_str_clean)
    if result:
        logger.warning("   ✅ Strategy 6 succeeded")
        return result
    
    logger.warning("❌ All strategies failed")
    return []


# ==========================================
# STRATEGY IMPLEMENTATIONS
# ==========================================

def _strategy_direct_parse(response_str: str) -> list:
    """Strategy 1: Try direct JSON parse"""
    try:
        result = json.loads(response_str)
        if isinstance(result, list) and len(result) > 0:
            logger.warning("      Direct parse successful")
            return result
    except json.JSONDecodeError as e:
        logger.warning("      Direct parse failed: %s", str(e)[:50])
    except Exception as e:
        logger.warning("      Direct parse error: %s", str(e)[:50])
    
    return None


def _strategy_markdown_extraction(response_str: str) -> list:
    """Strategy 2: Extract JSON from markdown code blocks"""
    try:
        # Try ```json ... ``` first
        if "```json" in response_str:
            json_str = response_str.split("```json")[1].split("```")[0].strip()
            logger.warning("      Found ```json block, parsing...")
            result = json.loads(json_str)
            if isinstance(result, list) and len(result) > 0:
                return result
        
        # Try ``` ... ``` (generic code block)
        if "```" in response_str:
            parts = response_str.split("```")
            for i in range(1, len(parts), 2):  # Check each code block
                json_str = parts[i].strip()
                # Skip if it's a language identifier
                if json_str.split('\n')[0] in ['json', 'python', 'javascript', 'bash', 'shell', 'text']:
                    json_str = '\n'.join(json_str.split('\n')[1:]).strip()
                
                try:
                    result = json.loads(json_str)
                    if isinstance(result, list) and len(result) > 0:
                        logger.warning("      Found markdown code block")
                        return result
                except json.JSONDecodeError:
                    continue
    
    except Exception as e:
        logger.warning("      Markdown extraction error: %s", str(e)[:50])
    
    return None


def _strategy_array_delimiter(response_str: str) -> list:
    """Strategy 3: Find JSON array [ ... ]"""
    try:
        # Find first [ and last ]
        start_idx = response_str.find('[')
        end_idx = response_str.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_str[start_idx:end_idx + 1].strip()
            logger.warning("      Found array delimiters, extracted %d chars", len(json_str))
            
            result = json.loads(json_str)
            if isinstance(result, list) and len(result) > 0:
                logger.warning("      Array parsing successful")
                return result
    
    except json.JSONDecodeError as e:
        logger.warning("      Array parse failed: %s", str(e)[:50])
    except Exception as e:
        logger.warning("      Array extraction error: %s", str(e)[:50])
    
    return None


def _strategy_object_extraction(response_str: str) -> list:
    """Strategy 4: Extract and reconstruct JSON objects"""
    try:
        # Find all { ... } patterns and try to extract objects
        objects = []
        depth = 0
        start_idx = -1
        
        for i, char in enumerate(response_str):
            if char == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_idx != -1:
                    obj_str = response_str[start_idx:i + 1]
                    try:
                        obj = json.loads(obj_str)
                        objects.append(obj)
                        logger.warning("      Extracted object %d", len(objects))
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        
        if len(objects) > 0:
            logger.warning("      Found %d objects, reconstructing array", len(objects))
            return objects
    
    except Exception as e:
        logger.warning("      Object extraction error: %s", str(e)[:50])
    
    return None


def _strategy_regex_extraction(response_str: str) -> list:
    """Strategy 5: Use regex to extract JSON structures"""
    try:
        # Match JSON array pattern
        array_pattern = r'\[\s*\{[^\[\]]*(?:\{[^\{\}]*\}[^\[\]]*)*\s*\]'
        match = re.search(array_pattern, response_str, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            logger.warning("      Regex found JSON array (%d chars)", len(json_str))
            try:
                result = json.loads(json_str)
                if isinstance(result, list) and len(result) > 0:
                    logger.warning("      Regex extraction successful")
                    return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: match individual objects
        object_pattern = r'\{[^{}]*(?:"[^"]*"[^{}]*)*\}'
        matches = re.finditer(object_pattern, response_str, re.DOTALL)
        objects = []
        
        for match in matches:
            obj_str = match.group(0)
            try:
                obj = json.loads(obj_str)
                objects.append(obj)
            except json.JSONDecodeError:
                pass
        
        if len(objects) > 0:
            logger.warning("      Regex found %d objects", len(objects))
            return objects
    
    except Exception as e:
        logger.warning("      Regex extraction error: %s", str(e)[:50])
    
    return None


def _strategy_fallback_rebuild(response_str: str) -> list:
    """Strategy 6: Last resort - rebuild JSON from parsed text"""
    try:
        # Try to salvage by removing problematic characters
        logger.warning("      Attempting character cleanup...")
        
        # Remove control characters except newline/tab
        cleaned = ''.join(char if char.isprintable() or char in '\n\t' else '' for char in response_str)
        
        # Try parsing cleaned version
        try:
            result = json.loads(cleaned)
            if isinstance(result, list) and len(result) > 0:
                logger.warning("      Cleanup successful")
                return result
        except json.JSONDecodeError:
            pass
        
        # Try adding missing array brackets if needed
        if '[' not in cleaned and '{' in cleaned:
            cleaned_with_brackets = '[' + cleaned + ']'
            try:
                result = json.loads(cleaned_with_brackets)
                if isinstance(result, list) and len(result) > 0:
                    logger.warning("      Added array brackets successfully")
                    return result
            except json.JSONDecodeError:
                pass
    
    except Exception as e:
        logger.warning("      Fallback rebuild error: %s", str(e)[:50])
    
    return None