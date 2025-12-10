# ==============================================================================
# FILE: backend/agents/root_orchestrator.py (WORKAROUND - Reuse Single Session)
# ==============================================================================
"""
Root Orchestrator - FIXED with proper session reuse

WORKAROUND:
- Create ONE session at startup
- Reuse it for all queries (ADK limitation)
- Don't try to create new sessions (causes "already exists" error)
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
SESSION_ID = "persistent_session"  # SINGLE session for all queries

# ===== TIMEOUT CONFIGURATION =====
REQUEST_TIMEOUT_SECONDS = 60
ADK_RUNNER_TIMEOUT = 180


# ===== ROOT ORCHESTRATOR CLASS =====

class RootOrchestrator:
    """
    Main orchestrator with SINGLE PERSISTENT SESSION
    
    WORKAROUND for ADK InMemorySessionService limitation:
    - Create session ONCE at init
    - Reuse for all queries
    - Don't create new sessions (causes "already exists" error)
    """
    
    def __init__(self):
        logger.warning("🚀 Initializing Root Orchestrator (ADK-based)")
        
        # Create the root sequential agent
        self.root_agent = SequentialAgent(
            name="root_orchestrator",
            description="Main orchestrator for fact-checking pipeline",
            
            sub_agents=[
                ingestion_agent,        # Stage 1: Process input
                extraction_agent,       # Stage 2: Extract claim
                verification_agent,     # Stage 3: Search evidence
                aggregation_agent,      # Stage 4: Generate verdict
                report_agent           # Stage 5: Format report
            ]
        )
        
        # CRITICAL: Create session service ONCE
        self.session_service = InMemorySessionService()
        self.session_created = False
        
        # Metrics
        self.api_calls = 0
        self.cache_hits = 0
        
        logger.warning("✅ Root Orchestrator initialized")
        logger.warning("   Pipeline: Ingestion → Extraction → Verification → Aggregation → Report")
        logger.warning(f"   Request Timeout: {REQUEST_TIMEOUT_SECONDS}s")
        logger.warning(f"   Pipeline Timeout: {ADK_RUNNER_TIMEOUT}s")
        logger.warning("   Session Strategy: Single persistent session (ADK workaround)")
    
    async def _ensure_session_created(self):
        """
        Create session ONCE at startup, then reuse it
        ADK limitation: Cannot create session with same ID twice
        """
        if not self.session_created:
            try:
                await self.session_service.create_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=SESSION_ID
                )
                self.session_created = True
                logger.warning(f"📌 Single session created: {SESSION_ID}")
                logger.warning("   This session will be reused for all queries (ADK limitation)")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.warning(f"📌 Session already exists: {SESSION_ID}")
                    self.session_created = True
                else:
                    logger.exception(f"❌ Failed to create session: {e}")
                    raise
    
    async def process_query(self, user_input: str, session_id: str) -> Dict:
        """
        Process a fact-checking query
        Uses single persistent session
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
            
            # CRITICAL: Ensure session exists (created once, reused)
            await self._ensure_session_created()
            
            # Create runner with PERSISTENT session
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
            
            # Run with timeout protection
            final_response = ""
            try:
                async with asyncio.timeout(ADK_RUNNER_TIMEOUT):
                    async for event in runner.run_async(
                        user_id=USER_ID,
                        session_id=SESSION_ID,  # Use persistent session
                        new_message=user_content
                    ):
                        # Capture final response
                        if event.is_final_response() and event.content and event.content.parts:
                            final_response = event.content.parts[0].text
            
            except asyncio.TimeoutError:
                logger.warning("⏱️ Pipeline execution TIMEOUT after %ds", ADK_RUNNER_TIMEOUT)
                return self._create_fallback_response(
                    user_input,
                    "Processing took too long - try a simpler claim",
                    start_time
                )
            
            logger.warning("✅ Pipeline execution complete")
            
            # Get session state
            session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=SESSION_ID
            )
            
            state = session.state if hasattr(session, 'state') else {}
            
            # Increment API calls quota
            quota_tracker.increment_call_count(2)
            self.api_calls += 2
            
            # Extract results
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
        
        except asyncio.CancelledError:
            logger.warning("❌ Request was cancelled")
            return self._create_fallback_response(
                user_input,
                "Request was cancelled",
                start_time
            )
        
        except Exception as e:
            logger.exception("❌ Pipeline error: %s", str(e)[:100])
            
            error_str = str(e).lower()
            
            if "timeout" in error_str or "connection" in error_str:
                friendly_msg = "API timeout - The free tier is slow. Try again in a moment."
            elif "429" in error_str or "quota" in error_str:
                friendly_msg = "API quota exceeded - 20/day limit reached"
            else:
                friendly_msg = f"Error: {str(e)[:100]}"
            
            return self._create_fallback_response(
                user_input,
                friendly_msg,
                start_time
            )
    
    def _create_fallback_response(self, user_input: str, error_msg: str, start_time: float) -> Dict:
        """Create fallback response on error"""
        return {
            "success": False,
            "error": error_msg,
            "claim": user_input[:100],
            "verdict": "INCONCLUSIVE",
            "confidence": 0.0,
            "report": f"❌ **Error**\n\n{error_msg}",
            "execution_time_ms": (time.time() - start_time) * 1000,
            "api_calls": 0,
            "evidence_count": 0
        }
    
    async def run_pipeline(self, user_input: str, session_id: str) -> Dict:
        """Public API"""
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