"""Wraps the built-in Google Search tool from ADK into an AgentTool with logging."""
from google.adk.tools import google_search
from google.adk.agents import Agent
from config import get_logger

logger = get_logger(__name__)

logger.debug("Creating google_search_agent_tool wrapper")

# ADK exposes google_search tool (per your earlier list). Wrap it into an AgentTool for clarity.
google_search_agent_tool = Agent(
    name="basic_google_search_tool",
    model="gemini-2.5-flash",
    description="Use Google Search to find web evidence for a claim.",
    tools=[google_search]
)