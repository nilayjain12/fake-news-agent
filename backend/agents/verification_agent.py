# ==============================================================================
# FILE 5: backend/agents/verification_agent.py (REPLACE EXISTING)
# ==============================================================================

from google.adk import Agent
from config import ADK_MODEL_NAME, get_logger
from tools.custom_tools import search_knowledge_base, search_web, evaluate_evidence

logger = get_logger(__name__)


def create_verification_agent():
    """Creates the Verification LLM Agent"""
    logger.warning("🚀 Creating Verification Agent")
    
    agent = Agent(
        model=ADK_MODEL_NAME,
        name="verification_agent",
        description="Retrieves and evaluates evidence for a claim",
        instruction="""You are a verification agent.
Given a claim:
1. Use search_knowledge_base tool to find relevant information
2. Use search_web tool to find current web sources
3. Use evaluate_evidence tool to classify evidence as SUPPORTS or REFUTES
4. Return a summary of your findings

Provide the evaluation results as your final response.""",
        tools=[search_knowledge_base, search_web, evaluate_evidence]
    )
    
    return agent


verification_agent = create_verification_agent()