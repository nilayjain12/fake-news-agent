# backend/agents/fact_check_agent_adk.py
"""Complete fact-checking pipeline with proper tool integration."""
from google.adk.agents import LlmAgent, SequentialAgent
from agents.ingestion_agent import IngestionAgent
from agents.claim_extraction_agent import ClaimExtractionAgent
from agents.verification_agent import VerificationAgent
from agents.aggregator_and_verdict import aggregate_evaluations
from config import ADK_MODEL_NAME, get_logger
from memory.manager import MemoryManager
import json

logger = get_logger(__name__)


class FactCheckSequentialAgent(SequentialAgent):
    """Orchestrates the fact-checking pipeline with proper tool integration."""

    def __init__(self):
        super().__init__(name="fact_check_sequential_agent")
        logger.warning("🚀 Initializing Fact-Check Agent")
        
        # Store custom components
        object.__setattr__(self, "ingest_agent", IngestionAgent())
        object.__setattr__(self, "memory_manager", MemoryManager())
        object.__setattr__(self, "claim_extractor", ClaimExtractionAgent())
        object.__setattr__(self, "verifier", VerificationAgent())
        
        # For ADK compatibility - create a dummy LlmAgent that will be overridden
        dummy_agent = LlmAgent(
            name="fact_checker",
            model=ADK_MODEL_NAME,
            instruction="This agent orchestrates fact-checking",
            description="Main fact-checking orchestrator"
        )
        
        self._sub_agents = [dummy_agent]
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
    
    def run_fact_check_pipeline(self, input_text: str) -> dict:
        """
        Execute the complete fact-checking pipeline:
        1. Extract claims from input
        2. Verify each claim using FAISS + Google Search
        3. Aggregate results and generate report
        """
        try:
            logger.warning("📋 Starting fact-check pipeline")
            
            # STEP 1: Extract Claims
            logger.warning("🔍 Step 1: Extracting claims...")
            claim_extractor = object.__getattribute__(self, "claim_extractor")
            claims = claim_extractor.run(input_text)
            
            if not claims:
                logger.warning("❌ No claims extracted")
                return {
                    "claims": [],
                    "verdicts": [],
                    "report": "No claims could be extracted from the input."
                }
            
            logger.warning("✅ Extracted %d claims", len(claims))
            
            # STEP 2: Verify each claim
            logger.warning("🔍 Step 2: Verifying claims with FAISS + Google Search...")
            verifier = object.__getattribute__(self, "verifier")
            all_evaluations = []
            
            for claim in claims:
                logger.warning("   Verifying: %s", claim[:60])
                result = verifier.run(claim)
                all_evaluations.extend(result.get("evaluations", []))
            
            logger.warning("✅ Verified with %d evidence items", len(all_evaluations))
            
            # STEP 3: Aggregate results
            logger.warning("🔍 Step 3: Aggregating results...")
            if all_evaluations:
                aggregation = aggregate_evaluations(all_evaluations)
                verdict = aggregation["verdict"]
                confidence = abs(aggregation["score"]) if aggregation["score"] != 0 else 0.5
            else:
                verdict = "UNVERIFIED"
                confidence = 0.0
            
            logger.warning("✅ Aggregation complete - Verdict: %s", verdict)
            
            # STEP 4: Generate human-readable report
            report = self._generate_report(claims, verdict, confidence, all_evaluations)
            
            return {
                "claims": claims,
                "verdict": verdict,
                "confidence": confidence,
                "evaluations": all_evaluations,
                "report": report
            }
            
        except Exception as e:
            logger.exception("❌ Pipeline error: %s", e)
            return {
                "claims": [],
                "verdict": "ERROR",
                "confidence": 0.0,
                "evaluations": [],
                "report": f"Error during fact-checking: {str(e)[:100]}"
            }
    
    def _generate_report(self, claims: list, verdict: str, confidence: float, evaluations: list) -> str:
        """Generate a professional fact-check report."""
        report = "### Fact-Check Report\n\n"
        
        if not claims:
            return report + "No claims to verify."
        
        report += f"**Overall Verdict:** **{verdict}** ({confidence:.1%} confidence)\n\n"
        report += "---\n\n"
        report += "### Claims Analyzed\n\n"
        
        for i, claim in enumerate(claims, 1):
            report += f"{i}. **Claim:** {claim}\n\n"
            
            # Find evaluations for this claim
            claim_evals = [e for e in evaluations if claim.lower() in str(e).lower()]
            
            if claim_evals:
                report += f"   **Evidence Found:** {len(claim_evals)} sources\n\n"
                supports = sum(1 for e in claim_evals if e.get("label") == "SUPPORTS")
                refutes = sum(1 for e in claim_evals if e.get("label") == "REFUTES")
                
                if refutes > supports:
                    report += f"   **Status:** ❌ Refuted ({refutes} refuting sources)\n\n"
                elif supports > refutes:
                    report += f"   **Status:** ✅ Supported ({supports} supporting sources)\n\n"
                else:
                    report += f"   **Status:** ⚠️  Mixed evidence\n\n"
            else:
                report += "   **Evidence Found:** None\n\n"
        
        report += "---\n\n"
        report += f"**Final Assessment:** {verdict}\n\n"
        report += "*Generated using FAISS knowledge base + Google Search verification*\n"
        
        return report
    
    def extract_confidence_from_verdict(self, verdict_str: str) -> float:
        """Extract confidence level (0.0-1.0) from verdict string."""
        if not verdict_str:
            return 0.5
        
        verdict_lower = verdict_str.lower()
        
        if "error" in verdict_lower:
            return 0.0
        elif "false" in verdict_lower and "mostly" not in verdict_lower:
            return 0.1
        elif "mostly false" in verdict_lower:
            return 0.3
        elif "unverified" in verdict_lower or "mixed" in verdict_lower:
            return 0.5
        elif "mostly true" in verdict_lower:
            return 0.75
        elif "true" in verdict_lower and "false" not in verdict_lower:
            return 0.9
        else:
            return 0.5

    def cache_result(self, claim: str, verdict: str, confidence: float, 
                     evidence_count: int, session_id: str):
        """Cache the verification result to memory."""
        try:
            memory_manager = object.__getattribute__(self, "memory_manager")
            memory_manager.cache_verdict(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence_count=evidence_count,
                session_id=session_id
            )
            logger.warning("✅ Result successfully cached")
        except Exception as e:
            logger.warning("❌ Error caching result: %s", str(e)[:100])
    
    def check_cache(self, claim: str) -> dict:
        """Check if we've already verified this claim."""
        try:
            memory_manager = object.__getattribute__(self, "memory_manager")
            cached = memory_manager.get_cached_verdict(claim)
            if cached:
                return {
                    "from_cache": True,
                    "verdict": cached["verdict"],
                    "confidence": cached["confidence"],
                    "evidence_count": cached["evidence_count"]
                }
            return {"from_cache": False}
        except Exception as e:
            logger.warning("⚠️  Error checking cache: %s", str(e)[:50])
            return {"from_cache": False}