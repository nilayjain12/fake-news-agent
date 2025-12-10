# ============================================================================
# FILE 2: backend/agents/extraction_agent.py (FIXED)
# ============================================================================

from google.adk.agents import Agent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import extract_main_claim
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class ExtractionInput(BaseModel):
    processed_text: str = Field(description="Cleaned text from ingestion stage")


class ExtractionOutput(BaseModel):
    main_claim: str = Field(description="The primary verifiable claim")
    claim_type: str = Field(description="Type: 'factual', 'scientific', or 'historical'")


def create_extraction_agent():
    logger.warning("🚀 Creating Extraction Agent (With Retry)")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "extraction_agent",
        "description": "Extracts the main verifiable claim from text",
        "instruction": """You are a claim extraction agent.

Given the following processed text:
{processed_text}

Your job is:
1. Use the extract_main_claim tool to identify the PRIMARY claim
2. Verify the claim is verifiable (not a question or opinion)
3. Return ONLY the extracted claim

Do not add explanations.""",
        "tools": [extract_main_claim],
        "input_schema": ExtractionInput,
        "output_schema": ExtractionOutput,
        "output_key": "main_claim"
    }
    
    if GENERATE_CONTENT_CONFIG is not None:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    agent = Agent(**agent_kwargs)
    return agent


extraction_agent = create_extraction_agent()