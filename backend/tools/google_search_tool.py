# ==============================================================================
# FILE 2: backend/tools/google_search_tool.py (ADK + Google Search Tool - FINAL)
# ==============================================================================

"""
Google Search Tool - ADK Agent using built-in google_search tool
✅ Fully async, safe for Gradio, FastAPI, and ADK pipelines
✅ No event-loop crashes
✅ Output format preserved
"""

import json
from typing import List, Dict

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

APP_NAME = "google_search_agent_app"
USER_ID = "google_search_user"
SESSION_ID = "google_search_session"


# ------------------------------------------------------------------------------
# ✅ ADK GOOGLE SEARCH AGENT
# ------------------------------------------------------------------------------
google_search_agent = LlmAgent(
    model=ADK_MODEL_NAME,
    name="google_search_agent",
    description="Agent that uses Google Search to fetch real-time factual information.",
    instruction=(
        "You are a web search agent.\n"
        "You MUST use the google_search tool.\n\n"
        "You MUST return STRICTLY VALID JSON.\n"
        "If you cannot find evidence, return this EXACT output:\n"
        "[]\n\n"
        "Never return text outside JSON.\n"
        "Never include explanations.\n"
        "Never include markdown.\n\n"
        "Output format:\n"
        "[\n"
        "  {\n"
        "    \"rank\": 1,\n"
        "    \"title\": \"Article Title\",\n"
        "    \"content\": \"Brief summary (max 200 chars)\",\n"
        "    \"url\": \"https://example.com\",\n"
        "    \"relevance_score\": 0.95\n"
        "  }\n"
        "]"
    ),
    tools=[google_search],
)


# ------------------------------------------------------------------------------
# ✅ ASYNC SEARCH FUNCTION (THIS IS NOW THE REAL TOOL)
# ------------------------------------------------------------------------------
async def search_google(query: str, top_k: int = 10) -> List[Dict]:
    """
    ✅ Async Google search using ADK agent
    ✅ REQUIRED for Gradio / FastAPI / ADK
    ✅ Output format unchanged
    """
    logger.warning(f"🔍 Google Search Agent: Searching for '{query[:60]}'")

    try:
        session_service = InMemorySessionService()

        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )

        runner = Runner(
            agent=google_search_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    text=(
                        f"Search for: {query}. "
                        f"Return at most {top_k} results using the exact schema."
                    )
                )
            ],
        )

        events = runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content,
        )

        final_text = ""

        async for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                part = event.content.parts[0]
                if getattr(part, "text", None):
                    final_text = part.text.strip()

        results = _parse_json_response(final_text, top_k)

        if results:
            for result in results:
                result["_source"] = "google_search"

            logger.warning(f"   ✅ Parsed {len(results)} results")
            return results

        logger.warning("   ⚠️  No valid JSON returned")
        return []

    except Exception as e:
        logger.warning(f"❌ Google Search Agent error: {str(e)[:300]}")
        return []


# ------------------------------------------------------------------------------
# ✅ JSON PARSER (UNCHANGED)
# ------------------------------------------------------------------------------
def _parse_json_response(response_str: str, top_k: int) -> List[Dict]:
    if not response_str:
        return []

    try:
        result = json.loads(response_str)
        if isinstance(result, list):
            return result[:top_k]
    except json.JSONDecodeError:
        pass

    try:
        if "```json" in response_str:
            json_str = response_str.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response_str:
            json_str = response_str.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            start = response_str.find("[")
            end = response_str.rfind("]")
            if start != -1 and end != -1:
                json_str = response_str[start : end + 1]
            else:
                return []

        result = json.loads(json_str)
        if isinstance(result, list):
            return result[:top_k]
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning("   ⚠️  Could not parse JSON from agent response")
    return []
