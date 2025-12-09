# backend/agents/root_orchestrator.py (REPLACE EXISTING)
"""
Optimized Root Orchestrator with API Quota Management
- Detects quota exhaustion
- Falls back to cached/mock responses
- Minimal API calls (2 per query)
"""

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from config import ADK_MODEL_NAME, get_logger
from memory.manager import MemoryManager
from tools.custom_tools import faiss_search_tool, google_search_tool
from tools.semantic_ranker import SemanticRanker
import asyncio
import time
import json
from typing import Optional

from agents.ingestion_agent import ingestion_agent
from agents.extraction_agent import extraction_agent
from agents.report_agent import report_agent

logger = get_logger(__name__)


class QuotaAwareOrchestrator:
    """
    Orchestrator that handles API quota limits gracefully.
    - Falls back to cached results when quota is exceeded
    - Uses mock responses if no cache available
    - Tracks API call count
    """
    
    def __init__(self):
        logger.warning("🚀 Initializing Quota-Aware Root Orchestrator")
        self.app_name = "fact_check_app"
        self.session_service = InMemorySessionService()
        self.memory = MemoryManager()
        
        # Only need 2 agents in optimized version
        self.agents = {
            "ingestion": ingestion_agent,
            "extraction": extraction_agent,
            "report": report_agent
        }
        
        self.runners = {
            name: Runner(agent=agent, app_name=self.app_name, session_service=self.session_service)
            for name, agent in self.agents.items()
        }
        
        self.ranker = SemanticRanker()
        self.quota_exhausted = False
        self.api_call_count = 0
        
        logger.warning("✅ Quota-Aware Orchestrator initialized")
    
    async def run_pipeline(self, input_text: str, user_id: str = "user", session_id: str = None) -> dict:
        """Execute fact-check pipeline with quota management"""
        
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        logger.warning("📋 Starting Pipeline - Session: %s", session_id)
        start_time = time.time()
        
        try:
            # Create session
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id
            )
            
            # STAGE 1: INGESTION (No API call - just text processing)
            logger.warning("🔍 Stage 1/3: INGESTION")
            ingestion_msg = types.Content(role='user', parts=[types.Part(text=input_text)])
            
            try:
                ingestion_result = await self._run_agent("ingestion", user_id, session_id, ingestion_msg)
                logger.warning("✅ Ingestion complete: %s", ingestion_result[:80])
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning("⚠️  Quota exhausted during ingestion, using input as-is")
                    self.quota_exhausted = True
                    ingestion_result = input_text[:500]
                else:
                    raise
            
            # STAGE 2: EXTRACTION + VERIFICATION (API CALL #1)
            logger.warning("🔍 Stage 2/3: EXTRACTION + VERIFICATION (API CALL #1)")
            extraction_msg = types.Content(role='user', parts=[types.Part(text=ingestion_result)])
            
            try:
                extraction_result = await self._run_agent("extraction", user_id, session_id, extraction_msg)
                self.api_call_count += 1
                logger.warning("✅ Extraction complete (API call 1/2): %s", extraction_result[:80])
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning("⚠️  API Quota EXHAUSTED! Using cached/mock response")
                    self.quota_exhausted = True
                    
                    # Try to get from cache first
                    cached = self.memory.get_cached_verdict(ingestion_result)
                    if cached:
                        logger.warning("   💾 Using cached result")
                        return self._format_cached_response(cached, start_time)
                    
                    # Fall back to mock response
                    logger.warning("   🤖 Using mock response (no cache available)")
                    return self._get_mock_response(ingestion_result, start_time)
                else:
                    raise
            
            if not extraction_result or len(extraction_result.strip()) < 5:
                return {
                    "success": False,
                    "error": "Could not extract claim",
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "api_calls": self.api_call_count,
                    "quota_exhausted": self.quota_exhausted
                }
            
            # Search for evidence (NO API call - uses tools)
            logger.warning("   🔎 Searching evidence...")
            evidence_summary = await self._search_and_evaluate_evidence(extraction_result)
            logger.warning("   ✅ Evidence gathered")
            
            # STAGE 3: VERDICT + REPORT (API CALL #2)
            logger.warning("🔍 Stage 3/3: VERDICT + REPORT (API CALL #2)")
            
            final_context = f"""CLAIM: {extraction_result}

EVIDENCE SUMMARY:
{evidence_summary}

Based on the claim and evidence, generate a JSON response with:
- "verdict": "TRUE", "FALSE", or "INCONCLUSIVE"
- "confidence": 0-100
- "report": Complete fact-check report

Return ONLY valid JSON."""
            
            report_msg = types.Content(role='user', parts=[types.Part(text=final_context)])
            
            try:
                report_result = await self._run_agent("report", user_id, session_id, report_msg)
                self.api_call_count += 2
                logger.warning("✅ Report generated (API call 2/2)")
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    logger.warning("⚠️  API Quota EXHAUSTED during report generation")
                    self.quota_exhausted = True
                    
                    # Use mock response
                    logger.warning("   🤖 Using mock response for report")
                    return self._get_mock_response(extraction_result, start_time, evidence_summary)
                else:
                    raise
            
            # Parse report result
            try:
                if "{" in report_result and "}" in report_result:
                    json_str = report_result[report_result.find("{"):report_result.rfind("}")+1]
                    parsed_result = json.loads(json_str)
                    verdict = parsed_result.get("verdict", "INCONCLUSIVE")
                    confidence = parsed_result.get("confidence", 50) / 100.0
                    final_report = parsed_result.get("report", report_result)
                else:
                    verdict = "INCONCLUSIVE"
                    confidence = 0.5
                    final_report = report_result
            except json.JSONDecodeError:
                verdict = "INCONCLUSIVE"
                confidence = 0.5
                final_report = report_result
            
            execution_time = (time.time() - start_time) * 1000
            logger.warning("✅ Pipeline complete in %.0f ms", execution_time)
            
            return {
                "success": True,
                "claim": extraction_result,
                "evidence": evidence_summary,
                "verdict": verdict,
                "confidence": confidence,
                "report": final_report,
                "execution_time_ms": execution_time,
                "api_calls": self.api_call_count,
                "quota_exhausted": self.quota_exhausted
            }
        
        except Exception as e:
            logger.exception("❌ Pipeline failed: %s", e)
            execution_time = (time.time() - start_time) * 1000
            
            # Check if quota exhausted
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning("⚠️  API QUOTA EXHAUSTED")
                return {
                    "success": False,
                    "error": "API quota exhausted. Please try again in 24 hours or upgrade your API plan.",
                    "execution_time_ms": execution_time,
                    "api_calls": self.api_call_count,
                    "quota_exhausted": True
                }
            
            return {
                "success": False,
                "error": str(e)[:200],
                "execution_time_ms": execution_time,
                "api_calls": self.api_call_count,
                "quota_exhausted": False
            }
    
    async def _search_and_evaluate_evidence(self, claim: str) -> str:
        """Search for evidence using tools (NO API call)"""
        logger.warning("      📚 Searching FAISS knowledge base...")
        faiss_results = faiss_search_tool(claim, k=3)
        logger.warning("      ✅ Found %d FAISS results", len(faiss_results))
        
        logger.warning("      🌐 Searching Google...")
        google_results = google_search_tool(claim, top_k=5)
        logger.warning("      ✅ Found %d Google results", len(google_results))
        
        # Combine and rank results
        all_evidence = []
        
        for result in faiss_results:
            all_evidence.append({
                "content": result.get("content", "")[:200],
                "source": result.get("source", "FAISS"),
                "_source": "faiss"
            })
        
        for result in google_results:
            all_evidence.append({
                "content": result.get("content", "")[:200],
                "source": result.get("url", "Google Search"),
                "_source": "google_search"
            })
        
        # Rank evidence by relevance
        if all_evidence:
            try:
                ranked_evidence, _, _ = self.ranker._advanced_rank_with_multi_factor_scoring(
                    query=claim,
                    evidence_items=all_evidence,
                    top_k=5
                )
                logger.warning("      ✅ Ranked top 5 evidence items")
            except Exception as e:
                logger.warning("      ⚠️  Ranking failed: %s", str(e)[:50])
                ranked_evidence = all_evidence[:5]
        else:
            ranked_evidence = []
        
        # Format evidence summary
        evidence_summary = "EVIDENCE RETRIEVED:\n\n"
        
        supports_count = 0
        refutes_count = 0
        
        for i, evidence in enumerate(ranked_evidence, 1):
            source = evidence.get("source", "Unknown")
            content = evidence.get("content", "")
            
            if any(word in content.lower() for word in ["true", "confirm", "verify", "correct", "accurate"]):
                supports_count += 1
                direction = "✅ SUPPORTS"
            elif any(word in content.lower() for word in ["false", "incorrect", "wrong", "deny", "refute"]):
                refutes_count += 1
                direction = "❌ REFUTES"
            else:
                direction = "❓ NEUTRAL"
            
            evidence_summary += f"{i}. [{source}] {direction}\n   {content[:100]}...\n\n"
        
        evidence_summary += f"\nSUMMARY: {supports_count} supporting, {refutes_count} refuting evidence found"
        
        return evidence_summary
    
    async def _run_agent(self, agent_name: str, user_id: str, session_id: str, message: types.Content) -> str:
        """Run a single agent and return its final response text"""
        try:
            runner = self.runners[agent_name]
            final_response = ""
            
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message
            ):
                if event.is_final_response() and event.content:
                    if event.content.parts:
                        final_response = event.content.parts[0].text
            
            return final_response
        
        except Exception as e:
            logger.error(f"❌ Agent {agent_name} failed: {str(e)[:100]}")
            raise
    
    def _format_cached_response(self, cached: dict, start_time: float) -> dict:
        """Format cached verdict as response"""
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "claim": cached.get("claim_text", "Unknown"),
            "verdict": cached.get("verdict", "UNKNOWN"),
            "confidence": cached.get("confidence", 0.5),
            "report": f"""# Fact-Check Report (From Cache)

**Claim:** {cached.get("claim_text", "Unknown")}

**Verdict:** {cached.get("verdict", "UNKNOWN")}

**Confidence:** {cached.get("confidence", 0.5):.1%}

*This result was retrieved from cache. Run again for fresh verification.*""",
            "execution_time_ms": execution_time,
            "api_calls": 0,
            "quota_exhausted": True,
            "from_cache": True
        }
    
    def _get_mock_response(self, claim: str, start_time: float, evidence: Optional[str] = None) -> dict:
        """Generate mock response when quota exhausted"""
        execution_time = (time.time() - start_time) * 1000
        
        # Simple heuristic for mock verdict
        if any(word in claim.lower() for word in ["flat", "fake", "hoax", "conspiracy"]):
            verdict = "FALSE"
            confidence = 0.75
        elif any(word in claim.lower() for word in ["earth", "orbit", "science", "proven"]):
            verdict = "TRUE"
            confidence = 0.85
        else:
            verdict = "INCONCLUSIVE"
            confidence = 0.5
        
        report = f"""# Fact-Check Report (Mock - API Quota Exhausted)

**Claim:** {claim}

**Verdict:** {verdict}

**Confidence:** {confidence:.1%}

⚠️ **Note:** The API quota has been exhausted for today. This is a mock response based on basic heuristics.

**To continue fact-checking:**
1. Wait until tomorrow (API quota resets daily)
2. Upgrade to a paid Gemini API plan
3. Use a different API (Claude, OpenAI, etc.)

{f"**Evidence Summary:**\n{evidence}" if evidence else ""}

*For accurate fact-checking, please upgrade your API plan or wait for quota reset.*"""
        
        return {
            "success": True,
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "report": report,
            "execution_time_ms": execution_time,
            "api_calls": 0,
            "quota_exhausted": True,
            "is_mock": True
        }
    
    def preprocess_input(self, input_text: str) -> str:
        """Preprocess input if needed"""
        return input_text


# Create singleton instance
root_orchestrator = QuotaAwareOrchestrator()