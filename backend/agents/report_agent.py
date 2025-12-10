# ============================================================================
# FILE 5: backend/agents/report_agent.py (FIXED)
# ============================================================================

from google.adk.agents import Agent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import format_report
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class ReportInput(BaseModel):
    main_claim: str = Field(description="The original claim")
    verdict: str = Field(description="Final verdict (TRUE/FALSE/INCONCLUSIVE)")
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    aggregation_result: dict = Field(description="Results from aggregation")
    evaluation_results: dict = Field(description="Evidence evaluation counts")


class ReportOutput(BaseModel):
    report: str = Field(description="Formatted fact-check report")
    summary: str = Field(description="Brief summary for display")


def create_report_agent():
    logger.warning("🚀 Creating Report Agent")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "report_agent",
        "description": "Generates comprehensive fact-check reports",
        "instruction": """You are a report generation agent.

You have all the fact-checking results:

CLAIM: {main_claim}

VERDICT: {verdict}
CONFIDENCE: {confidence}

EVIDENCE EVALUATION:
{evaluation_results}

AGGREGATION DETAILS:
{aggregation_result}

Your job is:
1. Use format_report tool to create a comprehensive report
2. Include the claim, verdict, confidence, and evidence summary
3. Format it professionally for user display
4. Return ONLY the formatted report

Do not add explanations or metadata.""",
        "tools": [format_report],
        "input_schema": ReportInput,
        "output_schema": ReportOutput,
        "output_key": "final_report"
    }
    
    if GENERATE_CONTENT_CONFIG is not None:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    agent = Agent(**agent_kwargs)
    return agent


report_agent = create_report_agent()