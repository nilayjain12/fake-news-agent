# ==============================================================================
# FILE: backend/agents/ingestion_agent.py (ADK LlmAgent)
# ==============================================================================
"""
Ingestion Agent - First stage of fact-checking pipeline
Processes user input (URL or raw text) and produces cleaned content
"""

from google.adk.agents import Agent
from config import ADK_MODEL_NAME, get_logger
from tools.custom_tools import extract_url_content, validate_and_clean_text
from pydantic import BaseModel, Field

logger = get_logger(__name__)


# ===== INPUT/OUTPUT SCHEMAS (Optional but recommended) =====

class IngestionInput(BaseModel):
    """Input schema for ingestion agent"""
    user_input: str = Field(description="Raw user input (URL or text)")


class IngestionOutput(BaseModel):
    """Output schema for ingestion agent"""
    processed_text: str = Field(description="Cleaned and processed text")
    source_type: str = Field(description="'url' or 'text'")


# ===== AGENT CREATION =====

def create_ingestion_agent():
    """
    Creates the Ingestion LlmAgent
    
    Role: Process raw user input (URL or text) into clean, usable text
    
    Inputs:
    - user_input: Raw input from user
    
    Outputs:
    - Stores processed_text in session state
    
    Tools Used:
    - extract_url_content: Extract text from URLs
    - validate_and_clean_text: Clean raw text input
    """
    logger.warning("🚀 Creating Ingestion Agent")
    
    agent = Agent(
        model=ADK_MODEL_NAME,
        name="ingestion_agent",
        description="Processes and cleans input text or extracts content from URLs",
        
        instruction="""You are an ingestion agent that processes user input.

Your job is to:
1. Check if the input is a URL (starts with http:// or https://)
2. If it's a URL: Use the extract_url_content tool to get the page content
3. If it's text: Use the validate_and_clean_text tool to clean it
4. Return ONLY the processed text, nothing else

Do not add any explanations or metadata.""",
        
        tools=[extract_url_content, validate_and_clean_text],
        
        # Schemas help the LLM understand input/output format
        input_schema=IngestionInput,
        output_schema=IngestionOutput,
        
        # This stores the final response in session state under this key
        output_key="processed_text"
    )
    
    return agent


# ===== SINGLETON INSTANCE =====
ingestion_agent = create_ingestion_agent()