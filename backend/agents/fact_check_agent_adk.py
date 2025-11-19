# backend/agents/fact_check_agent_adk.py
"""Top-level agent that orchestrates the fact-checking pipeline."""
from google.adk.agents import LlmAgent, SequentialAgent
from agents.ingestion_agent import IngestionAgent
from tools.faiss_tool import faiss_search_tool
from tools.google_search_tool import google_search_agent_tool
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)


class FactCheckSequentialAgent(SequentialAgent):
    """Orchestrates the fact-checking pipeline sequentially."""

    def __init__(self):
        super().__init__(name="fact_check_sequential_agent")
        logger.warning("🚀 Initializing Fact-Check Agent")
        
        object.__setattr__(self, "ingest_agent", IngestionAgent())
        
        claim_extractor = LlmAgent(
            name="claim_extractor",
            model=ADK_MODEL_NAME,
            instruction="""You are a claim extraction expert. Your task is to:
1. Read the provided article text carefully
2. Extract 2-3 specific, factual, verifiable claims
3. Return ONLY a JSON array of claim strings, nothing else

Example output format:
["Joe Biden is the President of the United States", "Donald Trump won the 2024 election"]

Do not add any explanation, preamble, or markdown formatting. Just the JSON array.""",
            description="Extracts factual claims from article text"
        )
        
        verifier = LlmAgent(
            name="verifier",
            model=ADK_MODEL_NAME,
            instruction="""You are a claim verification expert. For each claim provided:

1. Use the faiss_search_tool to search your knowledge base for relevant information
2. Use the google_search_agent_tool to find current web information
3. Analyze evidence from both sources
4. For each piece of evidence, determine if it SUPPORTS, REFUTES, or provides NOT_ENOUGH_INFO
5. Provide a clear verdict

Return results as JSON with this format:
{
    "claims_analyzed": ["claim 1", "claim 2"],
    "verdicts": [
        {"claim": "claim text", "verdict": "TRUE/FALSE/UNVERIFIED", "evidence_count": 3}
    ]
}""",
            description="Verifies claims using available evidence sources",
            tools=[faiss_search_tool, google_search_agent_tool]
        )
        
        aggregator = LlmAgent(
            name="aggregator",
            model=ADK_MODEL_NAME,
            instruction="""You are a fact-check report generator. Based on the verification results:

1. Review all verdicts
2. Generate a comprehensive final report
3. Include: overall assessment, individual claim verdicts, confidence levels
4. Make it clear and professional for end users

Format as a readable report, not JSON.""",
            description="Aggregates verification results into a final report"
        )
        
        self._sub_agents = [claim_extractor, verifier, aggregator]
        object.__setattr__(self, "sub_agents", self._sub_agents)
    
    def preprocess_input(self, input_text: str) -> str:
        """Handle URL ingestion before passing to the agent."""
        ingest_agent = object.__getattribute__(self, "ingest_agent")
        
        if input_text.strip().startswith("http"):
            logger.warning("📄 Extracting content from URL")
            article_text = ingest_agent.run(input_text)
            if not article_text:
                return f"Error: Could not extract content from URL: {input_text}"
            return article_text
        else:
            return input_text