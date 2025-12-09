# backend/agents/adk_pipeline.py
"""
Main Fact-Checking Pipeline - Pure Google ADK
Uses SequentialAgent to orchestrate the complete workflow
"""
from google.adk.agents import SequentialAgent
from google.adk.sessions import Session
from typing import Dict, AsyncIterator
from config import get_logger
from memory.manager import MemoryManager
import time
from pydantic import BaseModel, ConfigDict

logger = get_logger(__name__)

class MockPluginManager:
    """
    Minimal mock for ADK PluginManager.
    The ADK agents try to call these callbacks during execution.
    We provide empty async methods to prevent 'NoneType' errors.
    """
    async def run_before_agent_callback(self, *args, **kwargs):
        pass

    async def run_after_agent_callback(self, *args, **kwargs):
        pass

class SimpleInvocationContext(BaseModel):
    """
    Pydantic-compatible invocation context for ADK SequentialAgent.
    This satisfies ADK's internal expectation of `model_copy()`.
    """

    session: object
    state: dict
    agent: object | None = None
    invocation_id: str | None = None
    plugin_manager: object | None = None 

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ✅ FIX: Added end_invocation to satisfy ADK internals cleanup call
    def end_invocation(self, *args, **kwargs):
        pass

class FactCheckPipeline:
    """
    Main ADK-based Fact-Checking Pipeline
    
    Architecture:
    - Uses SequentialAgent for deterministic orchestration
    - All agents communicate via session.state
    - Event-driven execution with streaming support
    - Memory integration for caching
    
    Pipeline Flow:
    1. IngestionAgent → Clean input
    2. ClaimExtractionAgent → Extract claim
    3. EvidenceRetrievalAgent → Get evidence (FAISS + Google)
    4. VerificationAgent → Evaluate evidence
    5. AggregatorAgent → Generate verdict
    6. ReportAgent → Create comprehensive report
    """
    
    def __init__(self):
        logger.warning("🚀 Initializing ADK Fact-Check Pipeline")
        
        # Initialize memory manager
        self.memory = MemoryManager()
        
        # Create the pipeline
        self.pipeline = self._create_pipeline()
        
        logger.warning("✅ ADK Pipeline initialized with 6 agents")
    
    def _create_pipeline(self) -> SequentialAgent:
        """
        Create the main SequentialAgent pipeline
        
        All agents are created using their factory functions
        """
        from agents.ingestion_agent import create_ingestion_agent
        from agents.claim_extraction import create_claim_extraction_agent
        from agents.evidence_retrieval import create_evidence_retrieval_agent
        from agents.verification import create_verification_agent
        from agents.aggregator import create_aggregator_agent
        from agents.report_generator import create_report_agent
        
        pipeline = SequentialAgent(
            name="fact_check_pipeline",
            sub_agents=[
                create_ingestion_agent(),           # Agent 1: Clean input
                create_claim_extraction_agent(),    # Agent 2: Extract claim
                create_evidence_retrieval_agent(),  # Agent 3: Get evidence
                create_verification_agent(),        # Agent 4: Evaluate evidence
                create_aggregator_agent(),          # Agent 5: Generate verdict
                create_report_agent()               # Agent 6: Create report
            ]
        )
        
        return pipeline
    
    async def verify_claim_async(
        self, 
        input_text: str, 
        session_id: str = None
    ) -> Dict:
        """
        Verify a claim using the ADK pipeline (async)
        
        Args:
            input_text: Raw text or claim to verify
            session_id: Optional session ID for tracking
        
        Returns:
            Dictionary with:
            - success: bool
            - verdict: str (TRUE/FALSE/INCONCLUSIVE)
            - confidence: float
            - comprehensive_report: str (markdown)
            - execution_time_ms: float
            - session_state: dict (for debugging)
        """
        logger.warning(f"📋 Starting fact-check: {input_text[:80]}")
        start_time = time.time()
        
        # Create ADK session
        if not session_id:
            session_id = f"session-{int(time.time())}"
        
        session = Session(
                    id=session_id,
                    appName="fact-check-ui",
                    userId="web-user"
                )
        session.state["input_text"] = input_text
        
        # ✅ Wrap in a context object with the Mock Plugin Manager
        ctx = SimpleInvocationContext(
            session=session,
            state=session.state,
            agent=self.pipeline,
            plugin_manager=MockPluginManager()
        )

        try:
            # Run the pipeline
            events = []
            async for event in self.pipeline.run_async(ctx):
                events.append(event)
                
                # Log event
                if hasattr(event, 'author') and hasattr(event, 'text'):
                    logger.warning(f"📨 [{event.author}] {event.text[:100]}")
            
            # Extract results from session state
            aggregation = session.state.get("aggregation_result", {})
            final_report = session.state.get("final_report", "")
            
            execution_time = (time.time() - start_time) * 1000
            
            result = {
                "success": True,
                "verdict": aggregation.get("verdict", "UNKNOWN"),
                "confidence": aggregation.get("confidence", 0.5),
                "comprehensive_report": final_report,
                "execution_time_ms": execution_time,
                "total_events": len(events),
                "session_state": dict(session.state)
            }
            
            logger.warning(f"✅ Fact-check complete in {execution_time:.0f}ms")
            logger.warning(f"   Verdict: {result['verdict']} ({result['confidence']:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.exception(f"❌ Pipeline error: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": False,
                "error": str(e),
                "verdict": "ERROR",
                "confidence": 0.0,
                "comprehensive_report": f"Error during fact-checking: {str(e)}",
                "execution_time_ms": execution_time
            }
    
    def verify_claim(self, input_text: str, session_id: str = None) -> Dict:
        """
        Synchronous wrapper for verify_claim_async
        
        For backward compatibility with existing code
        """
        import asyncio
        return asyncio.run(self.verify_claim_async(input_text, session_id))
    
    def preprocess_input(self, input_text: str) -> str:
        """
        Preprocess input (URL extraction if needed)
        
        If input is a URL, extract content before sending to pipeline
        """
        if input_text.strip().startswith("http"):
            logger.warning("📄 Detected URL - extracting content")
            try:
                import requests
                from bs4 import BeautifulSoup
                
                response = requests.get(input_text, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
                extracted = "\n\n".join(paragraphs)
                
                logger.warning(f"✅ Extracted {len(paragraphs)} paragraphs")
                return extracted
                
            except Exception as e:
                logger.warning(f"⚠️  URL extraction failed: {str(e)[:100]}")
                return input_text
        
        return input_text
    
    def cache_result(
        self, 
        claim: str, 
        verdict: str, 
        confidence: float,
        session_id: str
    ):
        """Cache verification result for future queries"""
        try:
            self.memory.cache_verdict(
                claim=claim[:500],
                verdict=verdict,
                confidence=confidence,
                evidence_count=1,
                session_id=session_id
            )
            logger.warning("💾 Result cached")
        except Exception as e:
            logger.warning(f"❌ Caching failed: {str(e)[:100]}")
    
    async def stream_events(
        self,
        input_text: str,
        session_id: str = None
    ) -> AsyncIterator[Dict]:
        """
        Stream ADK events in real-time
        
        Yields dictionaries with event information for UI updates
        Useful for showing agent progress in real-time
        """
        if not session_id:
            session_id = f"session-{int(time.time())}"
        
        session = Session(
            id=session_id,
            appName="fact-check-ui",
            userId="web-user"
        )
        session.state["input_text"] = input_text
        
        # ✅ Wrap in a context object with the Mock Plugin Manager
        ctx = SimpleInvocationContext(
            session=session,
            state=session.state,
            agent=self.pipeline,
            plugin_manager=MockPluginManager()
        )

        current_agent = None
        
        try:
            async for event in self.pipeline.run_async(ctx):
                event_author = event.author if hasattr(event, 'author') else None
                event_text = event.text if hasattr(event, 'text') else ""
                
                # Detect agent changes
                if event_author and event_author != current_agent:
                    if current_agent:
                        yield {
                            "type": "agent_complete",
                            "agent_name": current_agent
                        }
                    
                    current_agent = event_author
                    yield {
                        "type": "agent_start",
                        "agent_name": current_agent
                    }
                
                # Progress event
                if event_text:
                    yield {
                        "type": "agent_event",
                        "agent_name": current_agent or "unknown",
                        "text": event_text
                    }
            
            # Final completion
            if current_agent:
                yield {
                    "type": "agent_complete",
                    "agent_name": current_agent
                }
            
            # Extract final results
            aggregation = session.state.get("aggregation_result", {})
            final_report = session.state.get("final_report", "")
            
            yield {
                "type": "final_result",
                "result": {
                    "verdict": aggregation.get("verdict", "UNKNOWN"),
                    "confidence": aggregation.get("confidence", 0.5),
                    "comprehensive_report": final_report,
                    "session_state": dict(session.state)
                }
            }
            
        except Exception as e:
            logger.warning(f"❌ Stream error: {str(e)}")
            yield {
                "type": "error",
                "error": str(e)
            }


# Factory function
def create_fact_check_pipeline() -> FactCheckPipeline:
    """Create and return the main pipeline"""
    return FactCheckPipeline()