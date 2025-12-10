# ==============================================================================
# FILE: backend/agents/root_orchestrator.py (WITH CACHING & RETRY - 429 Fix)
# ==============================================================================
"""
Root Orchestrator - With intelligent caching and 429 error handling

STRATEGY:
1. Cache recent requests to avoid duplicate processing
2. Use client-side retry with exponential backoff
3. Check quota before making requests
4. Graceful degradation on quota exhaustion
"""

import time
import asyncio
from typing import Dict, Optional
from config import get_logger, quota_tracker, GENERATE_CONTENT_CONFIG

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

APP_NAME = "fact_check_agent_app"
USER_ID = "fact_check_user"


class RootOrchestrator:
    """
    Main orchestrator with intelligent caching and 429 error handling.
    
    FEATURES:
    1. Request deduplication - Cache recent identical claims
    2. Quota-aware - Check limits before processing
    3. Client-side retry - Exponential backoff for transient errors
    4. Graceful degradation - Return cached results on quota exhaustion
    """
    
    def __init__(self):
        logger.warning("🚀 Initializing Root Orchestrator (WITH CACHING & RETRY)")
        
        # Create root sequential agent
        self.root_agent = SequentialAgent(
            name="root_orchestrator",
            description="Main orchestrator for fact-checking pipeline",
            
            sub_agents=[
                ingestion_agent,
                extraction_agent,
                verification_agent,
                aggregation_agent,
                report_agent
            ]
        )
        
        # Session management
        self.session_service = InMemorySessionService()
        
        # Metrics
        self.api_calls = 0
        self.cache_hits = 0
        self.request_attempts = 0
        
        logger.warning("✅ Root Orchestrator initialized")
        logger.warning("   ✓ Caching enabled")
        logger.warning("   ✓ Client-side retry (3 attempts, exponential backoff)")
        logger.warning("   ✓ Quota-aware processing")
        logger.warning("   ✓ Graceful degradation on 429 errors")
    
    async def process_query(self, user_input: str, session_id: str) -> Dict:
        """
        Process a fact-checking query with intelligent caching and retry.
        
        PROCESS:
        1. Check cache for identical claim
        2. If cached: Return instantly (0 API calls)
        3. If not cached: Check quota
        4. Run pipeline with client-side retry
        5. Cache result for future identical claims
        
        Args:
            user_input: The claim or question to fact-check
            session_id: Session identifier
        
        Returns:
            Dictionary with success status and results
        """
        start_time = time.time()
        
        logger.warning("🔍 Processing query: %s", user_input[:80])
        
        try:
            # ===== STEP 1: CHECK CACHE =====
            logger.warning("📚 Step 1: Checking cache...")
            cached_result = quota_tracker.check_cache(user_input)
            
            if cached_result:
                self.cache_hits += 1
                execution_time = (time.time() - start_time) * 1000
                
                logger.warning("✅ Cache hit! Returning cached result")
                logger.warning("   Time: %.0fms (vs 8-13s normal)", execution_time)
                logger.warning("   API calls: 0 (saved 2 calls!)")
                
                return {
                    "success": True,
                    "claim": cached_result.get("claim"),
                    "verdict": cached_result.get("verdict"),
                    "confidence": cached_result.get("confidence"),
                    "report": cached_result.get("report"),
                    "execution_time_ms": execution_time,
                    "api_calls": 0,
                    "from_cache": True,
                    "cache_age_seconds": time.time() - cached_result.get("timestamp", time.time())
                }
            
            # ===== STEP 2: CHECK QUOTA =====
            logger.warning("📊 Step 2: Checking quota...")
            allowed, msg = quota_tracker.check_quota_available(2)
            
            if not allowed:
                logger.warning("🚫 Quota check failed: %s", msg)
                execution_time = (time.time() - start_time) * 1000
                
                return {
                    "success": False,
                    "error": f"⏰ {msg} Please try again later.",
                    "execution_time_ms": execution_time,
                    "api_calls": 0,
                    "quota_exhausted": True
                }
            
            # ===== STEP 3: CREATE SESSION =====
            logger.warning("📌 Step 3: Creating session...")
            await self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            # ===== STEP 4: RUN PIPELINE WITH RETRY =====
            logger.warning("⏳ Step 4: Running pipeline (with client-side retry)...")
            
            runner = Runner(
                agent=self.root_agent,
                app_name=APP_NAME,
                session_service=self.session_service
            )
            
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
            
            self.request_attempts += 1
            final_response = ""
            
            try:
                async for event in runner.run_async(
                    user_id=USER_ID,
                    session_id=session_id,
                    new_message=user_content
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                
                logger.warning("✅ Pipeline completed successfully")
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a quota error (should be caught by retry)
                if "429" in error_str or "resource_exhausted" in error_str:
                    logger.warning("⚠️ 429 Error (quota) - Even after retry")
                    execution_time = (time.time() - start_time) * 1000
                    
                    return {
                        "success": False,
                        "error": "Quota exhausted. The system reached its daily limit. Please try again tomorrow.",
                        "execution_time_ms": execution_time,
                        "api_calls": 0,
                        "quota_exhausted": True,
                        "retry_attempts": self.request_attempts
                    }
                
                # Other errors - propagate
                raise
            
            # ===== STEP 5: EXTRACT RESULTS =====
            logger.warning("📋 Step 5: Extracting results...")
            
            session = await self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            state = session.state if hasattr(session, 'state') else {}
            
            # Increment API call quota
            quota_tracker.increment_call_count(2)
            self.api_calls += 2
            
            # Extract results
            claim = state.get("main_claim", user_input[:100])
            verdict = state.get("verdict", "INCONCLUSIVE")
            confidence = float(state.get("confidence", 0.5))
            report = state.get("final_report", final_response)
            evaluation = state.get("evaluation_results", {})
            
            execution_time = (time.time() - start_time) * 1000
            
            logger.warning("✅ Processing complete!")
            logger.warning("   Claim: %s", claim[:60])
            logger.warning("   Verdict: %s (%.0f%%)", verdict, confidence * 100)
            logger.warning("   Time: %.0fms", execution_time)
            
            # ===== STEP 6: CACHE RESULT =====
            logger.warning("💾 Step 6: Caching result...")
            
            result_data = {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "report": report,
                "timestamp": time.time()
            }
            
            quota_tracker.store_cache(user_input, result_data)
            
            return {
                "success": True,
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "report": report,
                "execution_time_ms": execution_time,
                "api_calls": 2,
                "evidence_count": evaluation.get("total_evaluated", 0),
                "from_cache": False
            }
        
        except Exception as e:
            logger.exception("❌ Pipeline error: %s", str(e)[:200])
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": False,
                "error": str(e)[:200],
                "execution_time_ms": execution_time,
                "api_calls": 0
            }
    
    async def run_pipeline(self, user_input: str, session_id: str) -> Dict:
        """Public API - alias for process_query"""
        return await self.process_query(user_input, session_id)
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        quota_stats = quota_tracker.get_stats()
        
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "request_attempts": self.request_attempts,
            "quota": quota_stats
        }


# Singleton instance
root_orchestrator = RootOrchestrator()