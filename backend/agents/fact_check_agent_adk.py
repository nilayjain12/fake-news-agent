# backend/agents/fact_check_agent_adk.py - IMAGE VERIFICATION UPDATE
"""
UPDATED: Added unified report generation for image-based verification
Now image verification produces the same professional format as text verification
"""

from google import genai
import asyncio
from typing import Optional, Dict, List
from config import ADK_MODEL_NAME, GEMINI_API_KEY, get_logger
from memory.manager import MemoryManager
import json
import time
import os

logger = get_logger(__name__)
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


# ... (keep all existing agent functions: ingestion_agent, claim_extraction_agent, 
# verification_agent, aggregator_agent, report_agent - unchanged) ...


def ingestion_agent(input_text: str) -> str:
    """Agent 1: Processes and cleans input text"""
    logger.warning("🔍 Step 1: Ingestion Agent processing input...")
    
    prompt = f"""You are a content processor. Process this input for fact-checking:

INPUT: {input_text}

If it's a URL content, note that it has been extracted.
If it's raw text, clean it and prepare it.
Remove noise and identify the main topic.

Return: Cleaned text ready for claim extraction"""
    
    try:
        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        result = response.text if hasattr(response, 'text') else str(response)
        logger.warning("✅ Ingestion complete")
        return result
    except Exception as e:
        logger.warning("❌ Ingestion error: %s", str(e)[:100])
        return input_text


def claim_extraction_agent(cleaned_text: str) -> str:
    """Agent 2: Extracts main verifiable claims"""
    logger.warning("🔍 Step 2: Claim Extraction Agent identifying main claim...")
    
    prompt = f"""You are an expert fact-checker. Extract the PRIMARY claim from this text:

TEXT: {cleaned_text[:1000]}

RULES:
1. Extract ONE main claim (not multiple)
2. Make it self-contained and verifiable
3. Include all necessary context
4. Return the claim as a single sentence

Return ONLY the claim, nothing else."""
    
    try:
        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        claim = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.warning("✅ Claim extraction complete: %s", claim[:80])
        return claim
    except Exception as e:
        logger.warning("❌ Claim extraction error: %s", str(e)[:100])
        return cleaned_text[:100]


def verification_agent(claim: str, faiss_results: list, google_results: list) -> Dict:
    """Agent 3: Verifies claims with evidence"""
    logger.warning("🔍 Step 3: Verification Agent analyzing evidence...")
    
    faiss_summary = f"Knowledge base found {len(faiss_results)} results"
    if faiss_results:
        faiss_summary += f": {faiss_results[0].get('content', '')[:100]}"
    
    google_summary = f"Web search found {len(google_results)} results"
    if google_results:
        google_summary += f": {google_results[0].get('content', '')[:100]}"
    
    prompt = f"""You are an evidence evaluator. For this claim: "{claim}"

Evidence Retrieved:
- FAISS: {faiss_summary}
- Google: {google_summary}

Analyze: Does this evidence SUPPORT or REFUTE the claim?

Respond in JSON format:
{{
    "supports_count": <number>,
    "refutes_count": <number>,
    "primary_evidence": "<key finding>",
    "confidence": <0-1>
}}"""
    
    try:
        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        result_text = response.text if hasattr(response, 'text') else str(response)
        
        try:
            json_str = result_text
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0]
            elif "{" in result_text:
                json_str = result_text[result_text.find("{"):result_text.rfind("}")+1]
            
            result = json.loads(json_str)
            logger.warning("✅ Verification complete")
            return result
        except:
            logger.warning("⚠️  Could not parse verification JSON, using defaults")
            return {
                "supports_count": len(faiss_results) + len(google_results),
                "refutes_count": 0,
                "confidence": 0.7
            }
    except Exception as e:
        logger.warning("❌ Verification error: %s", str(e)[:100])
        return {
            "supports_count": 0,
            "refutes_count": 0,
            "confidence": 0.5
        }


def aggregator_agent(claim: str, verification_result: Dict) -> Dict:
    """Agent 4: Aggregates evidence into verdict"""
    logger.warning("🔍 Step 4: Aggregator Agent generating verdict...")
    
    supports = verification_result.get("supports_count", 0)
    refutes = verification_result.get("refutes_count", 0)
    
    prompt = f"""You are a verdict generator. Given:
- Claim: "{claim}"
- Evidence SUPPORTS count: {supports}
- Evidence REFUTES count: {refutes}

Generate a verdict: TRUE, FALSE, or INCONCLUSIVE

Respond in JSON format:
{{
    "verdict": "TRUE/FALSE/INCONCLUSIVE",
    "confidence": <0-1>,
    "reasoning": "<explanation>"
}}"""
    
    try:
        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        result_text = response.text if hasattr(response, 'text') else str(response)
        
        try:
            json_str = result_text
            if "```json" in result_text:
                json_str = result_text.split("```json")[1].split("```")[0]
            elif "{" in result_text:
                json_str = result_text[result_text.find("{"):result_text.rfind("}")+1]
            
            result = json.loads(json_str)
            logger.warning("✅ Aggregation complete: Verdict = %s", result.get("verdict", "UNKNOWN"))
            return result
        except:
            verdict = "TRUE" if supports > refutes else ("FALSE" if refutes > supports else "INCONCLUSIVE")
            confidence = abs(supports - refutes) / max(supports + refutes, 1)
            logger.warning("✅ Aggregation complete (fallback): Verdict = %s", verdict)
            return {
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": f"Based on {supports} supporting and {refutes} refuting sources"
            }
    except Exception as e:
        logger.warning("❌ Aggregation error: %s", str(e)[:100])
        return {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.5,
            "reasoning": "Unable to determine verdict"
        }


def report_agent(claim: str, aggregation_result: Dict, verification_result: Dict) -> str:
    """Agent 5: Generates comprehensive report"""
    logger.warning("🔍 Step 5: Report Agent creating comprehensive report...")
    
    verdict = aggregation_result.get("verdict", "UNKNOWN")
    confidence = aggregation_result.get("confidence", 0)
    reasoning = aggregation_result.get("reasoning", "")
    
    prompt = f"""You are a report writer. Create a professional fact-check report:

CLAIM: {claim}

VERDICT: {verdict} (Confidence: {confidence:.0%})

REASONING: {reasoning}

Please format this as a professional report with:
1. The original claim
2. Verdict with confidence percentage
3. Summary of findings
4. Conclusion

Make it readable for non-technical users."""
    
    try:
        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )
        report = response.text if hasattr(response, 'text') else str(response)
        logger.warning("✅ Report generation complete")
        return report
    except Exception as e:
        logger.warning("❌ Report generation error: %s", str(e)[:100])
        return f"""# Fact-Check Report

**Claim:** {claim}

**Verdict:** {verdict}

**Confidence:** {confidence:.0%}

**Analysis:** {reasoning}"""


class FactCheckSequentialAgent:
    """Main orchestrator with unified image verification reports"""
    
    def __init__(self):
        logger.warning("🚀 Initializing FactCheckSequentialAgent")
        self.memory = MemoryManager()
        logger.warning("✅ Agent initialized successfully")
    
    async def run_fact_check_async(self, input_text: str, session_id: str = None) -> Dict:
        """Execute fact-check pipeline asynchronously"""
        logger.warning("📋 Starting fact-check pipeline")
        start_time = time.time()
        
        try:
            cleaned_text = ingestion_agent(input_text)
            claim = claim_extraction_agent(cleaned_text)
            
            logger.warning("🔍 Step 3: Searching for evidence...")
            faiss_results = []
            google_results = []
            
            try:
                from tools.faiss_tool import faiss_search
                faiss_results = faiss_search(claim, k=5)
                logger.warning("   ✅ FAISS search: Found %d results", len(faiss_results))
            except Exception as e:
                logger.warning("   ⚠️  FAISS unavailable: %s", str(e)[:50])
            
            try:
                from tools.google_search_tool import google_search_tool
                google_results = google_search_tool(claim, top_k=10)
                logger.warning("   ✅ Google search: Found %d results", len(google_results))
            except Exception as e:
                logger.warning("   ⚠️  Google search unavailable: %s", str(e)[:50])
            
            verification_result = verification_agent(claim, faiss_results, google_results)
            aggregation_result = aggregator_agent(claim, verification_result)
            final_report = report_agent(claim, aggregation_result, verification_result)
            
            execution_time = (time.time() - start_time) * 1000
            logger.warning("✅ Pipeline complete in %.0f ms", execution_time)
            
            overall_verdict = aggregation_result.get("verdict", "UNKNOWN")
            
            return {
                "success": True,
                "comprehensive_report": final_report,
                "overall_verdict": overall_verdict,
                "execution_time_ms": execution_time,
                "total_claims": 1
            }
            
        except Exception as e:
            logger.exception("❌ Pipeline error: %s", e)
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "comprehensive_report": f"Error during fact-checking: {str(e)}",
                "overall_verdict": "ERROR",
                "execution_time_ms": execution_time
            }
    
    def run_fact_check_pipeline(self, input_text: str, session_id: str = None) -> Dict:
        """Synchronous wrapper for backward compatibility"""
        return asyncio.run(self.run_fact_check_async(input_text, session_id))
    
    def preprocess_input(self, input_text: str, image_path: str = None) -> str:
        """Preprocess input (URL extraction)"""
        if input_text.strip().startswith("http"):
            from agents.ingestion_agent import IngestionAgent
            ingestor = IngestionAgent()
            logger.warning("📄 Extracting content from URL")
            return ingestor.run(input_text)
        return input_text
    
    def run_fact_check_pipeline_with_image(self, image_path: str) -> Dict:
        """
        UPDATED: Process image with unified report format
        Now generates the same professional report as text verification
        """
        try:
            logger.warning("📋 Starting image-based fact-check pipeline")
            
            from agents.image_processing_agent import ImageProcessingAgent
            from datetime import datetime
            
            image_processor = ImageProcessingAgent()
            logger.warning("🖼️  Processing image...")
            
            image_result = image_processor.run(image_path)
            
            if not image_result["success"]:
                return {
                    "success": False,
                    "error": image_result.get("error", "Image processing failed"),
                    "claims": [],
                    "verdict": "ERROR",
                    "report": "Failed to process image"
                }
            
            claims = image_result.get("claims", [])
            extracted_text = image_result.get("extracted_text", "")
            
            if not claims:
                logger.warning("❌ No claims extracted from image")
                return {
                    "success": False,
                    "error": "No verifiable claims found in image",
                    "claims": [],
                    "verdict": "UNVERIFIED",
                    "report": "Could not extract any verifiable claims from the image.",
                    "extracted_text": extracted_text
                }
            
            logger.warning("✅ Extracted %d claims from image", len(claims))
            
            # **NEW: Generate unified professional reports for each claim**
            all_verdicts = []
            detailed_reports = []
            
            for i, claim in enumerate(claims, 1):
                logger.warning("   Checking claim %d/%d: %s", i, len(claims), claim[:60])
                result = asyncio.run(self.run_fact_check_async(claim))
                
                verdict = result.get("overall_verdict", "UNKNOWN")
                all_verdicts.append(verdict)
                
                # Store detailed report for each claim
                detailed_reports.append({
                    "claim_number": i,
                    "claim": claim,
                    "verdict": verdict,
                    "report": result.get("comprehensive_report", "No report")
                })
            
            # Determine overall verdict
            if "FALSE" in all_verdicts:
                overall_verdict = "FALSE"
            elif "TRUE" in all_verdicts:
                overall_verdict = "TRUE"
            else:
                overall_verdict = "INCONCLUSIVE"
            
            # **NEW: Generate unified professional report (matching text format)**
            unified_report = self._generate_unified_image_report(
                extracted_text=extracted_text,
                claims=claims,
                detailed_reports=detailed_reports,
                overall_verdict=overall_verdict,
                image_path=image_path
            )
            
            return {
                "success": True,
                "claims": claims,
                "verdict": overall_verdict,
                "verdicts": all_verdicts,
                "extracted_text": extracted_text,
                "report": unified_report,
                "detailed_reports": detailed_reports
            }
            
        except Exception as e:
            logger.exception("❌ Image pipeline error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "claims": [],
                "verdict": "ERROR",
                "report": f"Error during image processing: {str(e)}"
            }
    
    def cache_result(self, claim: str, verdict: str, confidence: float, 
                     evidence_count: int, session_id: str):
        """Cache verification result (fixed missing method)"""
        try:
            self.memory.cache_verdict(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence_count=evidence_count,
                session_id=session_id
            )
            logger.warning("💾 Result cached")
        except Exception as e:
            logger.warning("❌ Caching failed: %s", str(e)[:50])
    
    def _generate_unified_image_report(self, extracted_text: str, claims: list, 
                                      detailed_reports: list, overall_verdict: str,
                                      image_path: str) -> str:
        """
        NEW: Generate unified professional report for image verification
        Matches the format of text verification reports
        """
        from datetime import datetime
        from pathlib import Path
        import html
        
        date_str = datetime.now().strftime("%B %d, %Y")
        image_name = Path(image_path).name
        
        # Clean and format extracted text properly
        cleaned_text = extracted_text.strip()
        # Replace multiple newlines with double newline for proper spacing
        cleaned_text = '\n'.join(line.strip() for line in cleaned_text.split('\n') if line.strip())
        
        report = f"""# 📋 Fact-Check Report: Image Verification

**Date:** {date_str}  
**Source:** {image_name}  
**Overall Verdict:** **{overall_verdict}**

---

## 1. Extracted Text from Image

<div style="background-color: #002B57; padding: 20px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196F3; font-family: monospace; white-space: pre-wrap; line-height: 1.6;">
{html.escape(cleaned_text[:800])}{"..." if len(cleaned_text) > 800 else ""}
</div>

---

## 2. Claims Identified and Verified

"""
        
        # Add each claim's verification with better formatting
        for detail in detailed_reports:
            claim_num = detail["claim_number"]
            claim = detail["claim"]
            verdict = detail["verdict"]
            
            # Verdict emoji and color
            verdict_info = {
                "TRUE": ("✅", "#2ecc71", "True"),
                "FALSE": ("❌", "#e74c3c", "False"),
                "INCONCLUSIVE": ("❓", "#f39c12", "Inconclusive")
            }
            emoji, color, label = verdict_info.get(verdict, ("⚠️", "#95a5a6", "Unknown"))
            
            report += f"""<div style="margin: 20px 0; padding: 20px; background-color: #002B57; border-radius: 8px; border-left: 4px solid {color};">

### Claim {claim_num}: {emoji} **{label}**

<div style="background-color: #2E96FF; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 3px solid {color};">
<strong>Claim:</strong> "{claim}"
</div>

{detail["report"]}

</div>

---

"""
        
        # Overall summary
        true_count = sum(1 for v in [d["verdict"] for d in detailed_reports] if v == "TRUE")
        false_count = sum(1 for v in [d["verdict"] for d in detailed_reports] if v == "FALSE")
        inconclusive_count = sum(1 for v in [d["verdict"] for d in detailed_reports] if v == "INCONCLUSIVE")
        
        report += f"""## 3. Summary of Findings

Our verification process analyzed {len(claims)} claim(s) extracted from the image:

- **True Claims:** {true_count}
- **False Claims:** {false_count}
- **Inconclusive Claims:** {inconclusive_count}

"""
        
        if false_count > 0:
            report += """**Warning:** At least one claim in this image was found to be false or misleading.
"""
        elif true_count == len(claims):
            report += """**Conclusion:** All claims in the image have been verified as true based on available evidence.
"""
        else:
            report += """**Conclusion:** The image contains a mix of verified and unverified information.
"""
        
        report += """
---

## 4. Methodology

This image verification was performed using:

- **Gemini Vision API:** Text extraction (OCR) from image
- **Claim Clustering:** Intelligent grouping of related claims to reduce redundancy
- **Multi-Source Verification:**
  - FAISS Knowledge Base: Semantic search on verified information
  - Google Search: Real-time web verification
- **AI-Powered Analysis:** Automated evidence evaluation and verdict generation

---

*Generated by Fake News Detection Agent*
"""
        
        return report