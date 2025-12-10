# ============================================================================
# FILE 3: backend/agents/verification_agent.py (FIXED)
# ============================================================================

from google.adk.agents import Agent, SequentialAgent
from config import ADK_MODEL_NAME, get_logger, GENERATE_CONTENT_CONFIG
from tools.custom_tools import search_knowledge_base, search_web, evaluate_evidence
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class SearchInput(BaseModel):
    main_claim: str = Field(description="Claim to search evidence for")


class SearchOutput(BaseModel):
    results: list = Field(description="List of evidence items")
    count: int = Field(description="Number of results found")


class EvaluationOutput(BaseModel):
    supports: int = Field(description="Count of evidence supporting")
    refutes: int = Field(description="Count of evidence refuting")
    total_evaluated: int = Field(description="Total evidence evaluated")


def create_search_knowledge_agent():
    logger.warning("🚀 Creating Search Knowledge Agent")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "search_knowledge_agent",
        "description": "Searches FAISS knowledge base for relevant facts",
        "instruction": """You are a knowledge search agent.

Given this claim:
{main_claim}

Your job is:
1. Use search_knowledge_base tool with the claim as query
2. Search a FAISS database of verified facts
3. Return the search results

Output ONLY the results, nothing else.""",
        "tools": [search_knowledge_base],
        "input_schema": SearchInput,
        "output_schema": SearchOutput,
        "output_key": "faiss_results"
    }
    
    if GENERATE_CONTENT_CONFIG is not None:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    return Agent(**agent_kwargs)


def create_search_web_agent():
    logger.warning("🚀 Creating Search Web Agent")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "search_web_agent",
        "description": "Searches the web for current information",
        "instruction": """You are a web search agent.

Given this claim:
{main_claim}

Your job is:
1. Use search_web tool with the claim as query
2. Search Google for current information
3. Return the search results

Output ONLY the results, nothing else.""",
        "tools": [search_web],
        "input_schema": SearchInput,
        "output_schema": SearchOutput,
        "output_key": "web_results"
    }
    
    if GENERATE_CONTENT_CONFIG is not None:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    return Agent(**agent_kwargs)


def create_evaluate_evidence_agent():
    logger.warning("🚀 Creating Evaluate Evidence Agent")
    
    agent_kwargs = {
        "model": ADK_MODEL_NAME,
        "name": "evaluate_evidence_agent",
        "description": "Evaluates evidence supporting or refuting the claim",
        "instruction": """You are an evidence evaluation agent.

You have evidence from two sources:

From FAISS Knowledge Base:
{faiss_results}

From Web Search:
{web_results}

Your job is:
1. Use evaluate_evidence tool to analyze combined evidence
2. Classify each piece as SUPPORTS or REFUTES the claim
3. Count supports vs refutes
4. Return the evaluation results

Output ONLY the counts of supports and refutes.""",
        "tools": [evaluate_evidence],
        "input_schema": BaseModel,
        "output_schema": EvaluationOutput,
        "output_key": "evaluation_results"
    }
    
    if GENERATE_CONTENT_CONFIG is not None:
        agent_kwargs["generate_content_config"] = GENERATE_CONTENT_CONFIG
    
    return Agent(**agent_kwargs)


def create_verification_agent():
    logger.warning("🚀 Creating Verification Agent (Sequential Pipeline)")
    
    search_knowledge_agent = create_search_knowledge_agent()
    search_web_agent = create_search_web_agent()
    evaluate_evidence_agent = create_evaluate_evidence_agent()
    
    return SequentialAgent(
        name="verification_agent",
        description="Searches and evaluates evidence",
        sub_agents=[
            search_knowledge_agent,
            search_web_agent,
            evaluate_evidence_agent
        ]
    )


verification_agent = create_verification_agent()