# ============================================================================
# backend/agents/verification.py
# ============================================================================
"""
Verification Agent - Pure ADK LlmAgent
Uses {placeholder} syntax to read from session state
"""
from google.adk.agents import LlmAgent
from config import ADK_MODEL_NAME, get_logger
from pydantic import BaseModel
from typing import Optional

logger = get_logger(__name__)

class VerificationResultSchema(BaseModel):
    supports_count: int
    refutes_count: int
    total_evidence: int
    confidence: float
    evidence_summary: Optional[str] = None

def create_verification_agent() -> LlmAgent:
    """Create ADK LlmAgent for evidence verification"""
    model = ADK_MODEL_NAME
    
    agent = LlmAgent(
        name="verification_agent",
        model=model,
        instruction="""You are an expert evidence evaluator for fact-checking.

You will receive:
- Claim: {extracted_claim}
- Evidence: {evidence}

Analyze each evidence item and classify as SUPPORTS or REFUTES.

CLASSIFICATION RULES:
- SUPPORTS: Evidence confirms the claim
- REFUTES: Evidence contradicts OR is not relevant

Count the totals and calculate confidence:
- 90%+ majority → 0.95
- 80%+ majority → 0.88
- 70%+ majority → 0.78
- 60%+ majority → 0.68
- 50-60% majority → 0.55-0.65
- Equal split → 0.50

Return JSON with:
{
    "supports_count": <number>,
    "refutes_count": <number>,
    "total_evidence": <number>,
    "confidence": <0.0-1.0>,
    "evidence_summary": "<brief summary>"
}

Example:
Claim: "Python invented in 1991"
Evidence: 10 items (7 support, 3 refute)
Output:
{
    "supports_count": 7,
    "refutes_count": 3,
    "total_evidence": 10,
    "confidence": 0.78,
    "evidence_summary": "7 sources confirm Python created in 1991"
}""",
        output_schema=VerificationResultSchema,
        output_key="verification_result"
    )
    
    logger.warning("✅ Verification Agent created")
    return agent