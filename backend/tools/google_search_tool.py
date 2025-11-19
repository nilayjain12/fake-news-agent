# backend/tools/google_search_tool.py
"""Google Search tool wrapper for ADK."""
from google.adk.tools import google_search, AgentTool
from google.adk.agents import LlmAgent
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

_google_search_agent = LlmAgent(
    name="google_search_agent",
    model=ADK_MODEL_NAME,
    description="Performs web searches using Google to find current information about claims",
    instruction="""You are a web search specialist. When given a query:
1. Use the google_search tool to find relevant web information
2. Return the most relevant and recent information found
3. Include source URLs when available
4. Summarize key findings clearly""",
    tools=[google_search]
)

google_search_agent_tool = AgentTool(agent=_google_search_agent)