# backend/agents/claim_extraction.py - FIXED v2
"""
Claim Extraction Agent - Pure ADK LlmAgent with Structured Output
"""
from google.adk.agents import LlmAgent
from pydantic import BaseModel
from typing import Optional
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

class ExtractedClaimSchema(BaseModel):
    main_claim: str
    claim_type: Optional[str] = None
    confidence: Optional[float] = None

def create_claim_extraction_agent() -> LlmAgent:
    """Create ADK LlmAgent for claim extraction"""
    model = ADK_MODEL_NAME
    
    agent = LlmAgent(
        name="claim_extraction_agent",
        model=model,
        instruction="""You are an expert fact-checker specializing in claim extraction.

Read the cleaned_text from session state and extract ONE main verifiable claim.

Make it self-contained with all context (who, what, when, where).

EXAMPLES:

Input: "Dharmendra died recently"
Output:
{
    "main_claim": "Bollywood actor Dharmendra died recently",
    "claim_type": "person",
    "confidence": 0.9
}

Input: "The Sun sets in the east"
Output:
{
    "main_claim": "The Sun sets in the east",
    "claim_type": "fact",
    "confidence": 0.95
}

Input: "Great wall of China can be seen from Moon"
Output:
{
    "main_claim": "The Great Wall of China can be seen from the Moon",
    "claim_type": "fact",
    "confidence": 0.95
}

Return ONLY valid JSON matching the schema.""",
        output_schema=ExtractedClaimSchema,
        output_key="extracted_claim"
    )
    
    logger.warning("✅ Claim Extraction Agent created")
    return agent