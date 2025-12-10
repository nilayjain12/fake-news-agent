# FILE 4: backend/agents/aggregation_agent.py (SIMPLIFIED - Does report too)
# ===========================================================================

from google.adk.agents import Agent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import count_evidence, generate_verdict, format_report
from pydantic import BaseModel, Field

logger = get_logger(__name__)

class AggregationInput(BaseModel):
    evaluation_results: dict
    ingestion_result: dict

class AggregationOutput(BaseModel):
    verdict: str
    confidence: float
    report: str

def create_aggregation_agent():
    logger.warning("🚀 Creating Aggregation Agent (Verdict + Report)")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "aggregation_agent",
        "description": "Generates verdict and report",
        "instruction": """Generate final verdict and report.

Evidence: {evaluation_results}
Claim: {ingestion_result[main_claim]}

Your job:
1. Use count_evidence tool on {evaluation_results}
2. Use generate_verdict tool with counts
3. Use format_report tool to format final report
4. Return verdict, confidence, and report

VERDICT: [TRUE/FALSE/INCONCLUSIVE]
CONFIDENCE: [0.0-1.0]
REPORT: [Formatted report]""",
        "tools": [count_evidence, generate_verdict, format_report],
        "input_schema": AggregationInput,
        "output_schema": AggregationOutput,
        "output_key": "final_result"
    }
    
    if GENERATE_CONTENT_CONFIG:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    return Agent(**agent_kwargs)

aggregation_agent = create_aggregation_agent()