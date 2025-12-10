# ==============================================================================
# FILE 3: backend/agents/ingestion_agent.py (REPLACE EXISTING)
# ==============================================================================

from google.adk import Agent
from config import ADK_MODEL_NAME, get_logger
from tools.custom_tools import extract_url_content, validate_and_clean_text

logger = get_logger(__name__)


def create_ingestion_agent():
    """Creates the Ingestion LLM Agent"""
    logger.warning("🚀 Creating Ingestion Agent")
    
    agent = Agent(
        model=ADK_MODEL_NAME,
        name="ingestion_agent",
        description="Processes and cleans input text or extracts content from URLs",
        instruction="""You are an ingestion agent that processes user input.
If the input is a URL (starts with http):
1. Use the extract_url_content tool to get the content
2. Return the extracted text
 
If the input is raw text:
1. Use validate_and_clean_text tool to clean it
2. Return the cleaned text

Always return the processed text as your final response.""",
        tools=[extract_url_content, validate_and_clean_text]
    )
    
    return agent


ingestion_agent = create_ingestion_agent()