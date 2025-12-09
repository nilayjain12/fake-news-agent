# backend/agents/aggregator_and_verdict.py - UPDATED
"""
Aggregator Agent - Pure ADK LlmAgent
Uses {placeholder} syntax to read from session state
"""
from google.adk.agents import LlmAgent
from config import ADK_MODEL_NAME, get_logger
from pydantic import BaseModel
from typing import Optional

logger = get_logger(__name__)

class AggregationResultSchema(BaseModel):
    verdict: str
    confidence: float
    reasoning: Optional[str] = None

def create_aggregator_agent() -> LlmAgent:
    """Create ADK LlmAgent for verdict aggregation"""
    model = ADK_MODEL_NAME
    
    agent = LlmAgent(
        name="aggregator_agent",
        model=model,
        instruction="""You are a verdict generator for fact-checking.

You will receive:
- Claim: {extracted_claim}
- Verification Results: {verification_result}

Generate verdict based on evidence balance:
- TRUE: More sources SUPPORT than REFUTE
- FALSE: More sources REFUTE than SUPPORT  
- INCONCLUSIVE: Equal split or no evidence

Examples:

Input:
Claim: "Python invented in 1991"
Verification: supports_count=7, refutes_count=3, confidence=0.78

Output:
{
    "verdict": "TRUE",
    "confidence": 0.78,
    "reasoning": "7 sources support vs 3 refute. Evidence confirms Python created in 1991."
}

Input:
Claim: "Earth is flat"
Verification: supports_count=1, refutes_count=9, confidence=0.95

Output:
{
    "verdict": "FALSE",
    "confidence": 0.95,
    "reasoning": "9 sources refute vs 1 supports. Science proves Earth is spherical."
}

Return ONLY valid JSON.""",
        output_schema=AggregationResultSchema,
        output_key="aggregation_result"
    )
    
    logger.warning("✅ Aggregator Agent created")
    return agent