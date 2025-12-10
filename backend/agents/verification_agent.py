# FILE 3: backend/agents/verification_agent.py (KEEP SAME - Uses tools)
# ======================================================================

from google.adk.agents import Agent, SequentialAgent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import search_knowledge_base, search_web, evaluate_evidence
from pydantic import BaseModel, Field

logger = get_logger(__name__)

class SearchInput(BaseModel):
    main_claim: str = Field(description="Claim to search")

class SearchOutput(BaseModel):
    results: list
    count: int

class EvalOutput(BaseModel):
    supports: int
    refutes: int
    total_evaluated: int

def create_search_knowledge_agent():
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "search_knowledge_agent",
        "description": "Searches FAISS knowledge base",
        "instruction": """Search FAISS for: {main_claim}
Use search_knowledge_base tool.
Return results only.""",
        "tools": [search_knowledge_base],
        "input_schema": SearchInput,
        "output_schema": SearchOutput,
        "output_key": "faiss_results"
    }
    if GENERATE_CONTENT_CONFIG:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    return Agent(**agent_kwargs)

def create_search_web_agent():
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "search_web_agent",
        "description": "Searches the web",
        "instruction": """Search web for: {main_claim}
Use search_web tool.
Return results only.""",
        "tools": [search_web],
        "input_schema": SearchInput,
        "output_schema": SearchOutput,
        "output_key": "web_results"
    }
    if GENERATE_CONTENT_CONFIG:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    return Agent(**agent_kwargs)

def create_evaluate_evidence_agent():
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "evaluate_evidence_agent",
        "description": "Evaluates evidence",
        "instruction": """Evaluate combined evidence:
FAISS: {faiss_results}
WEB: {web_results}

Use evaluate_evidence tool.
Return evaluation counts only.""",
        "tools": [evaluate_evidence],
        "input_schema": BaseModel,
        "output_schema": EvalOutput,
        "output_key": "evaluation_results"
    }
    if GENERATE_CONTENT_CONFIG:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    return Agent(**agent_kwargs)

def create_verification_agent():
    logger.warning("🚀 Creating Verification Agent")
    return SequentialAgent(
        name="verification_agent",
        description="Searches and evaluates evidence",
        sub_agents=[
            create_search_knowledge_agent(),
            create_search_web_agent(),
            create_evaluate_evidence_agent()
        ]
    )

verification_agent = create_verification_agent()