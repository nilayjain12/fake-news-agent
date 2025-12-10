# FILE 2: backend/agents/ingestion_agent.py (SIMPLIFIED - Does extract too)
# =========================================================================

from google.adk.agents import Agent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import extract_url_content, validate_and_clean_text, extract_main_claim
from pydantic import BaseModel, Field

logger = get_logger(__name__)

class IngestionInput(BaseModel):
    user_input: str = Field(description="Raw user input")

class IngestionOutput(BaseModel):
    processed_text: str = Field(description="Cleaned text")
    main_claim: str = Field(description="Extracted claim")

def create_ingestion_agent():
    logger.warning("🚀 Creating Ingestion Agent (Input + Extraction)")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "ingestion_agent",
        "description": "Processes input and extracts main claim",
        "instruction": """You are an ingestion & extraction agent.

Your job:
1. If URL: Use extract_url_content tool
   Else: Use validate_and_clean_text tool
2. Use extract_main_claim tool to find the main claim
3. Return both the cleaned text AND the extracted claim

Only return the text and claim, nothing else.""",
        "tools": [extract_url_content, validate_and_clean_text, extract_main_claim],
        "input_schema": IngestionInput,
        "output_schema": IngestionOutput,
        "output_key": "ingestion_result"
    }
    
    if GENERATE_CONTENT_CONFIG:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    return Agent(**agent_kwargs)

ingestion_agent = create_ingestion_agent()