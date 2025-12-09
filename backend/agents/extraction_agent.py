# ==============================================================================
# FILE 4: backend/agents/extraction_agent.py (NEW FILE - REPLACES claim_extraction_agent.py)
# ==============================================================================

from google.adk import Agent
from config import ADK_MODEL_NAME, get_logger
from tools.custom_tools import extract_main_claim

logger = get_logger(__name__)


def create_extraction_agent():
    """Creates the Claim Extraction LLM Agent"""
    logger.warning("🚀 Creating Extraction Agent")
    
    agent = Agent(
        model=ADK_MODEL_NAME,
        name="extraction_agent",
        description="Extracts the main verifiable claim from text",
        instruction="""You are a claim extraction agent.
Given input text:
1. Use the extract_main_claim tool to identify the PRIMARY claim
2. Verify the claim is verifiable and not a question
3. Return the extracted claim as your final response

Only return the claim text, nothing else.""",
        tools=[extract_main_claim]
    )
    
    return agent


extraction_agent = create_extraction_agent()