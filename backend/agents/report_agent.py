# ==============================================================================
# FILE 7: backend/agents/report_agent.py (NEW FILE)
# ==============================================================================

from google.adk import Agent
from config import ADK_MODEL_NAME, get_logger
from tools.custom_tools import format_report

logger = get_logger(__name__)


def create_report_agent():
    """Creates the Report LLM Agent"""
    logger.warning("🚀 Creating Report Agent")
    
    agent = Agent(
        model=ADK_MODEL_NAME,
        name="report_agent",
        description="Generates comprehensive fact-check reports",
        instruction="""You are a report generation agent.
Given verification results and verdict:
1. Use format_report tool to create a comprehensive report
2. Include the claim, verdict, confidence, and evidence count
3. Return the formatted report as your final response

Provide only the formatted report.""",
        tools=[format_report]
    )
    
    return agent


report_agent = create_report_agent()