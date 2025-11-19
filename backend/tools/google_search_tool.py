"""Wraps the built-in Google Search tool from ADK into an AgentTool."""
from google.adk.tools import google_search, AgentTool
from google.adk.agents import LlmAgent
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

logger.debug("Creating google_search_agent_tool wrapper")

# Create an agent that uses Google Search
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

# Wrap the agent as a tool so it can be used by other agents
# This is necessary because you can't pass an LlmAgent directly as a tool
google_search_agent_tool = AgentTool(agent=_google_search_agent)