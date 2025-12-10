# ============================================================================
# FILE 4: backend/agents/aggregation_agent.py (FIXED)
# ============================================================================

from google.adk.agents import Agent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import count_evidence, generate_verdict
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class AggregationInput(BaseModel):
    evaluation_results: dict = Field(description="Results from evidence evaluation")
    main_claim: str = Field(description="The claim being evaluated")


class AggregationOutput(BaseModel):
    verdict: str = Field(description="Final verdict: TRUE/FALSE/INCONCLUSIVE")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="Explanation of verdict")


def create_aggregation_agent():
    logger.warning("🚀 Creating Aggregation Agent")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "aggregation_agent",
        "description": "Aggregates evidence and generates final verdict",
        "instruction": """You are an aggregation agent generating fact-check verdicts.

Given the evaluation results:
{evaluation_results}

For claim:
{main_claim}

Your job is:
1. Use count_evidence tool to count SUPPORTS vs REFUTES
2. Use generate_verdict tool to create the final verdict
3. Return ONLY the verdict information in this format:

VERDICT: [TRUE/FALSE/INCONCLUSIVE]
CONFIDENCE: [0.0-1.0 as decimal]
REASONING: [2-3 sentence explanation]

Do not add any other text.""",
        "tools": [count_evidence, generate_verdict],
        "input_schema": AggregationInput,
        "output_schema": AggregationOutput,
        "output_key": "aggregation_result"
    }
    
    if GENERATE_CONTENT_CONFIG is not None:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    agent = Agent(**agent_kwargs)
    return agent


aggregation_agent = create_aggregation_agent()