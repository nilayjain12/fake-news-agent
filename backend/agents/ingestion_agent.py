# backend/agents/ingestion_agent.py - FINAL WORKING VERSION
"""
Ingestion Agent - Pure ADK LlmAgent
Processes and cleans input text for fact-checking
"""
from google.adk.agents import LlmAgent
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

def create_ingestion_agent() -> LlmAgent:
    """Create ADK LlmAgent for input processing"""
    model = ADK_MODEL_NAME
    
    agent = LlmAgent(
        name="ingestion_agent",
        model=model,
        instruction="""You are a content processor for fact-checking systems.

Read the input_text from session state and clean it:
1. Remove HTML artifacts and excessive whitespace
2. Fix formatting issues
3. Preserve all factual content
4. Return ONLY the cleaned text

Examples:
- Input: "Breaking!!! Scientists say <p>climate change</p> is real!!!"
  Output: "Scientists say climate change is real"
  
- Input: "The Sun sets in the east."
  Output: "The Sun sets in the east"

Return ONLY the cleaned text, nothing else.""",
        output_key="cleaned_text"
    )
    
    logger.warning("✅ Ingestion Agent created")
    return agent