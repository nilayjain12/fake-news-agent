# ==============================================================================
# FILE 6: backend/agents/aggregation_agent.py (NEW FILE - REPLACES aggregator_and_verdict.py)
# ==============================================================================

from google.adk import Agent
from config import ADK_MODEL_NAME, get_logger
from tools.custom_tools import count_evidence, generate_verdict

logger = get_logger(__name__)


def create_aggregation_agent():
    """Creates the Aggregation LLM Agent"""
    logger.warning("🚀 Creating Aggregation Agent")
    
    agent = Agent(
        model=ADK_MODEL_NAME,
        name="aggregation_agent",
        description="Aggregates evidence and generates final verdict",
        instruction="""You are an aggregation agent.
Given evaluation results:
1. Use count_evidence tool to tally SUPPORTS and REFUTES
2. Use generate_verdict tool to create the final verdict
3. Return the verdict in this format: VERDICT: [TRUE/FALSE/INCONCLUSIVE], CONFIDENCE: [X%]

Provide only the verdict information as your final response.""",
        tools=[count_evidence, generate_verdict]
    )
    
    return agent


aggregation_agent = create_aggregation_agent()