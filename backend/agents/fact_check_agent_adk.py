# backend/agents/fact_check_agent_adk.py
"""Complete fact-checking pipeline with proper tool integration."""
from google.adk.agents import LlmAgent
from agents.ingestion_agent import IngestionAgent
from agents.claim_extraction_agent import ClaimExtractionAgent
from agents.verification_agent import VerificationAgent
from agents.aggregator_and_verdict import aggregate_evaluations
from config import ADK_MODEL_NAME, get_logger
from memory.manager import MemoryManager
import json

logger = get_logger(__name__)


class CustomToolRegistry:
    """Simple custom tool registry (replaces MCP to avoid conflicts)."""
    
    def __init__(self):
        self.tools = {}
        logger.warning("🔧 Custom Tool Registry initialized")
    
    def register_tool(self, name, description, input_schema, handler):
        """Register a tool."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "handler": handler
        }
        logger.warning(f"📍 Tool registered: {name}")
    
    def list_tools(self):
        """List all tools."""
        return [
            {
                "name": name,
                "description": data["description"],
                "input_schema": data["input_schema"]
            }
            for name, data in self.tools.items()
        ]
    
    def call_tool(self, tool_name, arguments):
        """Call a tool."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(self.tools.keys())
            }
        
        try:
            logger.warning(f"🔨 Calling tool: {tool_name}")
            handler = self.tools[tool_name]["handler"]
            result = handler(**arguments)
            return {
                "success": True,
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            logger.warning(f"❌ Tool execution failed: {str(e)[:100]}")
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e)
            }


class FactCheckSequentialAgent:
    """Orchestrates the fact-checking pipeline with tool integration.
    
    NOT inheriting from SequentialAgent to avoid Pydantic conflicts.
    Implements the same interface without the constraints.
    """

    def __init__(self):
        logger.warning("🚀 Initializing Fact-Check Agent")
        
        # Initialize all components as regular attributes (no Pydantic restrictions)
        self.ingest_agent = IngestionAgent()
        self.memory_manager = MemoryManager()
        self.claim_extractor = ClaimExtractionAgent()
        self.verifier = VerificationAgent()
        
        # Initialize Tool Registry
        self.tool_registry = CustomToolRegistry()
        self._register_tools()
        
        logger.warning("✅ Tool Registry initialized with %d tools", len(self.tool_registry.tools))
        
        # Try to import ImageProcessingAgent if available
        self.image_processor = None
        try:
            from agents.image_processing_agent import ImageProcessingAgent
            self.image_processor = ImageProcessingAgent()
            logger.warning("✅ ImageProcessingAgent loaded")
        except ImportError as e:
            logger.warning("⚠️  ImageProcessingAgent not available: %s", str(e)[:50])
        except Exception as e:
            logger.warning("⚠️  Error loading ImageProcessingAgent: %s", str(e)[:100])
        
        # For ADK compatibility (optional)
        self.name = "fact_check_sequential_agent"
        self._sub_agents = []
    
    def _register_tools(self):
        """Register all tools with the registry."""
        
        # Tool 1: FAISS Search
        self.tool_registry.register_tool(
            name="search_faiss",
            description="Search FAISS knowledge base for factual information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The claim to search for"},
                    "k": {"type": "integer", "description": "Number of results", "default": 5}
                },
                "required": ["query"]
            },
            handler=self._tool_faiss_search
        )
        
        # Tool 2: Google Search
        self.tool_registry.register_tool(
            name="search_google",
            description="Search Google for real-time information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The claim to verify"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5}
                },
                "required": ["query"]
            },
            handler=self._tool_google_search
        )
        
        # Tool 3: Image Processing
        self.tool_registry.register_tool(
            name="process_image",
            description="Extract verifiable claims from an image",
            input_schema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file"}
                },
                "required": ["image_path"]
            },
            handler=self._tool_process_image
        )
        
        # Tool 4: Cache Lookup
        self.tool_registry.register_tool(
            name="check_cache",
            description="Check if a claim has been previously verified",
            input_schema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The claim to look up"}
                },
                "required": ["claim"]
            },
            handler=self._tool_check_cache
        )
        
        logger.warning("✅ All tools registered")
    
    # ==================== Tool Handlers ====================
    
    def _tool_faiss_search(self, query: str, k: int = 5) -> dict:
        """Handle FAISS search."""
        try:
            from tools.faiss_tool import faiss_search
            results = faiss_search(query, k)
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.warning(f"FAISS search error: {str(e)[:100]}")
            return {"success": False, "error": str(e), "results": []}
    
    def _tool_google_search(self, query: str, top_k: int = 5) -> dict:
        """Handle Google search."""
        try:
            from tools.google_search_tool import google_search_tool
            results = google_search_tool(query, top_k)
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.warning(f"Google search error: {str(e)[:100]}")
            return {"success": False, "error": str(e), "results": []}
    
    def _tool_process_image(self, image_path: str) -> dict:
        """Handle image processing."""
        try:
            if self.image_processor is None:
                return {"success": False, "error": "ImageProcessingAgent not available"}
            return self.image_processor.run(image_path)
        except Exception as e:
            logger.warning(f"Image processing error: {str(e)[:100]}")
            return {"success": False, "error": str(e)}
    
    def _tool_check_cache(self, claim: str) -> dict:
        """Handle cache lookup."""
        try:
            cached = self.memory_manager.get_cached_verdict(claim)
            return {
                "success": True,
                "found": cached is not None,
                "cached_result": dict(cached) if cached else None
            }
        except Exception as e:
            logger.warning(f"Cache lookup error: {str(e)[:100]}")
            return {"success": False, "error": str(e)}
    
    # ==================== Public Tool Interface ====================
    
    def get_tools_list(self):
        """Return list of available tools."""
        return self.tool_registry.list_tools()
    
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool."""
        return self.tool_registry.call_tool(tool_name, arguments)
    
    # ==================== Fact-Check Pipeline ====================
    
    def preprocess_input(self, input_text: str, image_path: str = None) -> str:
        """Handle URL ingestion, image processing, or raw text."""
        
        # Process image if provided
        if image_path and self.image_processor:
            logger.warning("📸 Processing image input")
            try:
                result = self.image_processor.run(image_path)
                if result["success"]:
                    return result["extracted_text"]
                else:
                    logger.warning("❌ Image processing failed: %s", result.get("error"))
                    return ""
            except Exception as e:
                logger.warning("❌ Error processing image: %s", str(e)[:50])
                return ""
        
        # Handle URL
        if input_text.strip().startswith("http"):
            logger.warning("📄 Extracting content from URL")
            article_text = self.ingest_agent.run(input_text)
            if not article_text:
                return f"Error: Could not extract content from URL: {input_text}"
            return article_text
        
        # Return raw text
        return input_text
    
    def run_fact_check_pipeline(self, input_text: str) -> dict:
        """Execute the complete fact-checking pipeline."""
        try:
            logger.warning("📋 Starting fact-check pipeline")
            
            # STEP 1: Extract Claims
            logger.warning("🔍 Step 1: Extracting claims...")
            claims = self.claim_extractor.run(input_text)
            
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
            all_evaluations = []
            
            for claim in claims:
                logger.warning("   Verifying: %s", claim[:60])
                result = self.verifier.run(claim)
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
            
            # STEP 4: Generate report
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
    
    def run_fact_check_pipeline_with_image(self, image_path: str) -> dict:
        """Execute fact-checking pipeline starting from image."""
        try:
            logger.warning("📋 Starting image-based fact-check pipeline")
            
            if self.image_processor is None:
                return {
                    "claims": [],
                    "verdict": "ERROR",
                    "confidence": 0.0,
                    "report": "Image processing not available"
                }
            
            # Step 1: Extract claims from image
            logger.warning("🖼️  Step 1: Processing image...")
            image_result = self.image_processor.run(image_path)
            
            if not image_result["success"]:
                return {
                    "claims": [],
                    "verdicts": [],
                    "report": f"Failed to process image: {image_result.get('error', 'Unknown error')}"
                }
            
            claims = image_result["claims"]
            extracted_text = image_result["extracted_text"]
            
            if not claims:
                logger.warning("❌ No claims extracted from image")
                return {
                    "claims": [],
                    "verdicts": [],
                    "report": "No verifiable claims found in the image."
                }
            
            logger.warning("✅ Extracted %d claims from image", len(claims))
            
            # Step 2: Verify claims
            logger.warning("🔍 Step 2: Verifying claims...")
            all_evaluations = []
            
            for claim in claims:
                logger.warning("   Verifying: %s", claim[:60])
                result = self.verifier.run(claim)
                all_evaluations.extend(result.get("evaluations", []))
            
            # Step 3: Aggregate
            logger.warning("🔍 Step 3: Aggregating results...")
            if all_evaluations:
                aggregation = aggregate_evaluations(all_evaluations)
                verdict = aggregation["verdict"]
                confidence = abs(aggregation["score"]) if aggregation["score"] != 0 else 0.5
            else:
                verdict = "UNVERIFIED"
                confidence = 0.0
            
            # Generate report
            report = self._generate_image_report(claims, verdict, confidence, extracted_text)
            
            return {
                "claims": claims,
                "verdict": verdict,
                "confidence": confidence,
                "evaluations": all_evaluations,
                "extracted_text": extracted_text,
                "report": report
            }
            
        except Exception as e:
            logger.exception("❌ Image pipeline error: %s", e)
            return {
                "claims": [],
                "verdict": "ERROR",
                "confidence": 0.0,
                "report": f"Error during image fact-checking: {str(e)[:100]}"
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
    
    def _generate_image_report(self, claims: list, verdict: str, confidence: float, extracted_text: str) -> str:
        """Generate report for image-based verification."""
        report = "### Image-Based Fact-Check Report\n\n"
        report += f"**Overall Verdict:** **{verdict}** ({confidence:.1%} confidence)\n\n"
        report += "---\n\n"
        report += "### Extracted Text\n\n"
        report += f"{extracted_text[:500]}...\n\n" if len(extracted_text) > 500 else f"{extracted_text}\n\n"
        report += "---\n\n"
        report += "### Claims Identified and Verified\n\n"
        
        for i, claim in enumerate(claims, 1):
            report += f"{i}. **Claim:** {claim}\n\n"
        
        report += "---\n\n"
        report += f"**Final Assessment:** {verdict}\n"
        report += "*Generated using image OCR + FAISS + Google Search verification*\n"
        
        return report
    
    def extract_confidence_from_verdict(self, verdict_str: str) -> float:
        """Extract confidence level from verdict string."""
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
            self.memory_manager.cache_verdict(
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
            cached = self.memory_manager.get_cached_verdict(claim)
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