# FILE 5: backend/agents/root_orchestrator.py (SIMPLIFIED 3-AGENT PIPELINE)
# ===========================================================================

import time
from typing import Dict
from config import get_logger, quota_tracker, GENERATE_CONTENT_CONFIG

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.ingestion_agent import ingestion_agent
from agents.verification_agent import verification_agent
from agents.aggregation_agent import aggregation_agent

logger = get_logger(__name__)

class RootOrchestrator:
    """Simplified 3-agent pipeline with proper cleanup"""
    
    def __init__(self):
        logger.warning("🚀 Initializing Root Orchestrator (3-Agent Pipeline)")
        
        # SIMPLIFIED: Only 3 agents (was 5)
        self.root_agent = SequentialAgent(
            name="root_orchestrator",
            description="Fact-checking pipeline",
            sub_agents=[
                ingestion_agent,
                verification_agent,
                aggregation_agent
            ]
        )
        
        self.session_service = InMemorySessionService()
        self.api_calls = 0
        self.cache_hits = 0
        
        logger.warning("✅ Root Orchestrator ready")
        logger.warning("   Pipeline: Ingestion → Verification → Aggregation")
        logger.warning("   Expected time: 10-20 seconds")
    
    async def process_query(self, user_input: str, session_id: str) -> Dict:
        start_time = time.time()
        
        try:
            # Step 1: Cache check
            logger.warning("📚 Checking cache...")
            cached = quota_tracker.check_cache(user_input)
            if cached:
                self.cache_hits += 1
                return {
                    "success": True,
                    "claim": cached["claim"],
                    "verdict": cached["verdict"],
                    "confidence": cached["confidence"],
                    "report": cached["report"],
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "api_calls": 0,
                    "from_cache": True
                }
            
            # Step 2: Quota check
            logger.warning("📊 Checking quota...")
            allowed, msg = quota_tracker.check_quota_available(3)  # 3 calls for 3 agents
            if not allowed:
                return {
                    "success": False,
                    "error": msg,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "api_calls": 0
                }
            
            # Step 3: Create session
            await self.session_service.create_session(
                app_name="fact_check",
                user_id="user",
                session_id=session_id
            )
            
            # Step 4: Run pipeline (only 3 agents now)
            logger.warning("⏳ Running pipeline (3 agents)...")
            runner = Runner(
                agent=self.root_agent,
                app_name="fact_check",
                session_service=self.session_service
            )
            
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
            
            try:
                async for event in runner.run_async(
                    user_id="user",
                    session_id=session_id,
                    new_message=user_content
                ):
                    pass
                
                logger.warning("✅ Pipeline completed")
                
            except Exception as e:
                error_str = str(e).lower()
                if "timeout" in error_str:
                    logger.warning("⏱️ TIMEOUT")
                    return {
                        "success": False,
                        "error": "Pipeline took too long (>90s). Try again.",
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "api_calls": 0
                    }
                raise
            
            # Step 5: Get results
            session = await self.session_service.get_session(
                app_name="fact_check",
                user_id="user",
                session_id=session_id
            )
            
            state = session.state if hasattr(session, 'state') else {}
            
            # Extract from ingestion_result and final_result
            ingestion = state.get("ingestion_result", {})
            final = state.get("final_result", {})
            
            claim = ingestion.get("main_claim", user_input[:100]) if isinstance(ingestion, dict) else user_input[:100]
            verdict = final.get("verdict", "INCONCLUSIVE") if isinstance(final, dict) else "INCONCLUSIVE"
            confidence = float(final.get("confidence", 0.5)) if isinstance(final, dict) else 0.5
            report = final.get("report", "") if isinstance(final, dict) else ""
            
            quota_tracker.increment_call_count(3)
            self.api_calls += 3
            
            execution_time = (time.time() - start_time) * 1000
            
            logger.warning("✅ Complete!")
            logger.warning(f"   Verdict: {verdict}")
            logger.warning(f"   Time: {execution_time:.0f}ms")
            
            # Cache result
            quota_tracker.store_cache(user_input, {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "report": report
            })
            
            return {
                "success": True,
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "report": report,
                "execution_time_ms": execution_time,
                "api_calls": 3,
                "from_cache": False
            }
        
        except Exception as e:
            logger.exception(f"❌ Error: {str(e)[:100]}")
            return {
                "success": False,
                "error": str(e)[:150],
                "execution_time_ms": (time.time() - start_time) * 1000,
                "api_calls": 0
            }
    
    async def run_pipeline(self, user_input: str, session_id: str) -> Dict:
        return await self.process_query(user_input, session_id)
    
    def get_stats(self) -> Dict:
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "quota": quota_tracker.get_stats()
        }

root_orchestrator = RootOrchestrator()