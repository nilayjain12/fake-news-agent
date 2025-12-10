# ==============================================================================
# FILE: backend/agents/root_orchestrator.py (ADK Root Agent - NEW)
# ==============================================================================
"""
Root Orchestrator - Main entry point for the fact-checking pipeline

Uses ADK's SequentialAgent to orchestrate a 5-agent pipeline:
1. Ingestion Agent - Process input
2. Extraction Agent - Extract claim
3. Verification Agent - Search evidence
4. Aggregation Agent - Generate verdict
5. Report Agent - Format report

All agents share session state for clean data flow.
"""

import time
import asyncio
from typing import Dict, Optional
from config import get_logger, quota_tracker

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Import all agent factories
from agents.ingestion_agent import ingestion_agent
from agents.extraction_agent import extraction_agent
from agents.verification_agent import verification_agent
from agents.aggregation_agent import aggregation_agent
from agents.report_agent import report_agent

logger = get_logger(__name__)

# ADK app configuration
APP_NAME = "fact_check_agent_app"
USER_ID = "fact_check_user"


# ===== ROOT ORCHESTRATOR CLASS =====

class RootOrchestrator:
    """
    Main orchestrator for the fact-checking pipeline.
    
    Uses ADK's SequentialAgent to coordinate all sub-agents in order.
    Each agent processes input from previous agents via shared session state.
    """
    
    def __init__(self):
        logger.warning("🚀 Initializing Root Orchestrator (ADK-based)")
        
        # Create the root sequential agent
        self.root_agent = SequentialAgent(
            name="root_orchestrator",
            description="Main orchestrator for fact-checking pipeline",
            
            # Execute agents in sequence
            # Each has access to previous state
            sub_agents=[
                ingestion_agent,        # Stage 1: Process input → processed_text
                extraction_agent,       # Stage 2: Extract claim → main_claim
                verification_agent,     # Stage 3: Search evidence → evaluation_results
                aggregation_agent,      # Stage 4: Generate verdict → verdict, confidence
                report_agent           # Stage 5: Format report → final_report
            ]
        )
        
        # Session management
        self.session_service = InMemorySessionService()
        
        # Metrics
        self.api_calls = 0
        self.cache_hits = 0
        
        logger.warning("✅ Root Orchestrator initialized")
        logger.warning("   Pipeline: Ingestion → Extraction → Verification → Aggregation → Report")
    
    async def process_query(self, user_input: str, session_id: str) -> Dict:
        """
        Process a fact-checking query through the entire pipeline.
        
        Args:
            user_input: The claim or question to fact-check
            session_id: Session identifier for state management
        
        Returns:
            Dictionary with success status and results
        """
        start_time = time.time()
        
        logger.warning("🔍 Processing query: %s", user_input[:80])
        
        try:
            # Check quota
            allowed, msg = quota_tracker.check_quota_available(2)
            if not allowed:
                logger.warning("⚠️  Quota limited: %s", msg)
                return {
                    "success": False,
                    "error": msg,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "api_calls": 0
                }
            
            # Create session
            await self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            logger.warning("📌 Session created: %s", session_id)
            
            # Create runner with root agent
            runner = Runner(
                agent=self.root_agent,
                app_name=APP_NAME,
                session_service=self.session_service
            )
            
            # Prepare user message
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
            
            logger.warning("⏳ Starting pipeline execution...")
            
            # Run the entire pipeline
            final_response = ""
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                # Capture final response
                if event.is_final_response() and event.content and event.content.parts:
                    final_response = event.content.parts[0].text
            
            logger.warning("✅ Pipeline execution complete")
            
            # Get session state with all results
            session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            state = session.state if hasattr(session, 'state') else {}
            
            # Increment API calls quota
            quota_tracker.increment_call_count(2)
            self.api_calls += 2
            
            # Extract results from state
            claim = state.get("main_claim", user_input[:100])
            verdict = state.get("verdict", "INCONCLUSIVE")
            confidence = float(state.get("confidence", 0.5))
            report = state.get("final_report", final_response)
            evaluation = state.get("evaluation_results", {})
            
            execution_time = (time.time() - start_time) * 1000
            
            logger.warning("✅ Final Results:")
            logger.warning("   Claim: %s", claim[:60])
            logger.warning("   Verdict: %s (%.0f%%)", verdict, confidence * 100)
            logger.warning("   Time: %.0fms", execution_time)
            
            return {
                "success": True,
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "report": report,
                "execution_time_ms": execution_time,
                "api_calls": 2,
                "evidence_count": evaluation.get("total_evaluated", 0)
            }
        
        except Exception as e:
            logger.exception("❌ Pipeline error: %s", str(e)[:100])
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": False,
                "error": str(e)[:200],
                "execution_time_ms": execution_time,
                "api_calls": 0
            }
    
    async def run_pipeline(self, user_input: str, session_id: str) -> Dict:
        """
        Public API for running the pipeline.
        Alias for process_query for backwards compatibility.
        """
        return await self.process_query(user_input, session_id)
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "quota_status": quota_tracker.get_stats()
        }


# ===== SINGLETON INSTANCE =====
root_orchestrator = RootOrchestrator()