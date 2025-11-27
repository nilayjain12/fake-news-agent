# backend/agents/report_generator.py - WITH CORRECTED FACT SECTION
"""
Updated report generator that includes "Corrected Fact" or "What's Actually True"
section showing the actual truth based on retrieved sources
"""
from config import get_logger
from typing import List, Dict
import json
from google import genai
import os

logger = get_logger(__name__)

# Set up Gemini client for generating corrected facts
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


class DetailedReportGenerator:
    """Generates comprehensive fact-check reports with corrected fact section."""
    
    def __init__(self):
        logger.warning("📊 DetailedReportGenerator initialized")
        self.model = "gemini-2.5-flash"
    
    def generate_claim_report(self, claim: str, evaluations: List[Dict], 
                            aggregation_result: Dict) -> Dict:
        """
        Generate a detailed report for a single claim.
        
        Args:
            claim: The claim being verified
            evaluations: List of evaluation results from evidence
            aggregation_result: Aggregated verdict and score
            
        Returns:
            Dictionary with structured report data
        """
        logger.warning("📋 Generating detailed report for claim: %s", claim[:60])
        
        verdict = aggregation_result.get("verdict", "UNVERIFIED")
        score = aggregation_result.get("score", 0.0)
        confidence = aggregation_result.get("confidence", 0.0)
        breakdown = aggregation_result.get("breakdown", {})
        
        verdict_result = self._format_verdict(verdict, confidence, score)
        
        # Generate the corrected fact/actual truth (NEW)
        corrected_fact = self._generate_corrected_fact(claim, evaluations, verdict)
        
        explanation = self._generate_explanation(
            verdict_result, 
            evaluations, 
            breakdown
        )
        
        sources = self._extract_sources(evaluations)
        
        scoring_report = self._generate_scoring_report(
            breakdown,
            evaluations,
            score,
            aggregation_result.get("raw_score", 0),
            aggregation_result.get("total_weight", 0)
        )
        
        report = {
            "claim": claim,
            "corrected_fact": corrected_fact,  # NEW FIELD
            "result": verdict_result,
            "explanation": explanation,
            "sources": sources,
            "scoring_breakdown": scoring_report,
            "metadata": {
                "total_evaluations": len(evaluations),
                "supports_count": breakdown.get("SUPPORTS", 0),
                "refutes_count": breakdown.get("REFUTES", 0),
                "neutral_count": breakdown.get("NOT_ENOUGH_INFO", 0),
                "normalized_score": score,
                "confidence_percentage": confidence * 100
            }
        }
        
        logger.warning("✅ Report generated successfully")
        return report

    def _generate_corrected_fact(self, claim: str, evaluations: List[Dict], verdict: str) -> str:
        """
        Generate the actual/corrected fact based on retrieved sources.
        This shows what IS actually true based on the evidence.
        
        Args:
            claim: The original claim
            evaluations: List of evidence evaluations
            verdict: The verdict (TRUE, FALSE, INCONCLUSIVE)
            
        Returns:
            String with the corrected/actual fact
        """
        logger.warning("📝 Generating corrected fact based on retrieved sources")
        
        # Extract supporting and refuting evidence
        supporting_evidence = []
        refuting_evidence = []
        
        for eval_item in evaluations:
            evidence = eval_item.get("evidence", {})
            label = eval_item.get("label", "")
            
            if isinstance(evidence, dict):
                content = evidence.get("content", str(evidence))
                source = evidence.get("source", "")
            else:
                content = str(evidence)
                source = ""
            
            if label == "SUPPORTS":
                supporting_evidence.append({
                    "content": content[:300],
                    "source": source
                })
            elif label == "REFUTES":
                refuting_evidence.append({
                    "content": content[:300],
                    "source": source
                })
        
        # Use LLM to generate corrected fact (if client available)
        if client:
            corrected_fact = self._generate_corrected_fact_with_llm(
                claim, 
                supporting_evidence, 
                refuting_evidence, 
                verdict
            )
        else:
            corrected_fact = self._generate_corrected_fact_fallback(
                claim,
                supporting_evidence,
                refuting_evidence,
                verdict
            )
        
        logger.warning("✅ Corrected fact generated")
        return corrected_fact

    def _generate_corrected_fact_with_llm(self, claim: str, supporting: List[Dict], 
                                          refuting: List[Dict], verdict: str) -> str:
        """
        Use LLM to generate a well-written corrected fact statement.
        """
        try:
            # Prepare evidence summaries
            supporting_summary = " | ".join([e["content"][:100] for e in supporting])[:300]
            refuting_summary = " | ".join([e["content"][:100] for e in refuting])[:300]
            
            prompt = f"""Based on the following information, generate a brief statement of what's actually true.

ORIGINAL CLAIM: "{claim}"
VERDICT: {verdict}

EVIDENCE RETRIEVED:
Supporting Sources: {supporting_summary if supporting_summary else 'None found'}
Refuting Sources: {refuting_summary if refuting_summary else 'None found'}

TASK: Write a 2-3 sentence statement of what IS actually true based on the retrieved evidence.
- If verdict is FALSE: Explain what IS true instead
- If verdict is TRUE: Explain what the correct understanding is
- If verdict is INCONCLUSIVE: Explain the conflicting information

Be clear, factual, and cite the actual correct information from sources.
Start with "Actually:" or "In fact:" or "The truth is:"

OUTPUT ONLY the corrected fact statement, nothing else."""

            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            corrected_fact = response.text.strip() if hasattr(response, 'text') else str(response)
            logger.warning("   LLM corrected fact: %s", corrected_fact[:80])
            return corrected_fact
            
        except Exception as e:
            logger.warning("⚠️  LLM generation failed: %s", str(e)[:100])
            return self._generate_corrected_fact_fallback(
                claim, supporting, refuting, verdict
            )

    def _generate_corrected_fact_fallback(self, claim: str, supporting: List[Dict], 
                                         refuting: List[Dict], verdict: str) -> str:
        """
        Fallback method: Extract corrected fact from retrieved sources directly.
        """
        logger.warning("⚠️  Using fallback method for corrected fact")
        
        if verdict.lower() == "false":
            # Claim is FALSE - extract what's actually true from refuting sources
            if refuting:
                actual_fact = refuting[0]["content"][:200].strip()
                return f"Actually, {actual_fact}"
            else:
                return f"The claim '{claim}' is not supported by available evidence."
        
        elif verdict.lower() == "true":
            # Claim is TRUE - extract the supporting evidence
            if supporting:
                actual_fact = supporting[0]["content"][:200].strip()
                return f"In fact, {actual_fact}"
            else:
                return f"The claim '{claim}' is supported by evidence."
        
        else:
            # INCONCLUSIVE
            if supporting and refuting:
                support_text = supporting[0]["content"][:100].strip()
                refute_text = refuting[0]["content"][:100].strip()
                return f"The evidence is mixed: Some sources suggest '{support_text}...' while others indicate '{refute_text}...'"
            else:
                return f"There is insufficient clear evidence about '{claim}'."

    def generate_comprehensive_report_single_claim(self, main_claim: str, claim_report: Dict) -> str:
        """
        Generate report for single claim with CORRECTED FACT section added.
        Positioned after Confidence and before Explanation.
        """
        logger.warning("📋 Generating single-claim comprehensive report with corrected fact")
        
        result = claim_report["result"]
        corrected_fact = claim_report["corrected_fact"]  # NEW
        sources = claim_report["sources"]
        scoring = claim_report["scoring_breakdown"]
        
        report = "# 📋 Fact-Check Report\n\n"
        
        # Overall Verdict Box
        report += f"## ✅ Result\n\n"
        report += f"**Claim:** {main_claim}\n\n"
        report += f"**Verdict:** {result['category']}\n\n"
        report += f"**Confidence:** {result['confidence_percentage']}% ({result['confidence_level']})\n\n"
        
        report += "---\n\n"
        
        # NEW: Corrected Fact Section
        report += f"## 📌 What's Actually True\n\n"
        report += f"**Reason:** {corrected_fact}\n\n"
        
        report += "---\n\n"
        
        # Explanation
        report += f"## 📝 Explanation\n\n"
        report += f"{claim_report['explanation']}\n\n"
        
        report += "---\n\n"
        
        # Sources
        if sources:
            report += f"## 🔗 Sources ({len(sources)} found)\n\n"
            
            supports = [s for s in sources if s.get("supports")]
            refutes = [s for s in sources if s.get("refutes")]
            neutral = [s for s in sources if not s.get("supports") and not s.get("refutes")]
            
            if supports:
                report += f"### ✅ Supporting Sources ({len(supports)})\n"
                for source in supports:
                    report += f"- **[{source['title']}]({source['url']})**\n"
                    if source['snippet']:
                        report += f"  > {source['snippet'][:120]}\n\n"
            
            if refutes:
                report += f"### ❌ Refuting Sources ({len(refutes)})\n"
                for source in refutes:
                    report += f"- **[{source['title']}]({source['url']})**\n"
                    if source['snippet']:
                        report += f"  > {source['snippet'][:120]}\n\n"
            
            if neutral:
                report += f"### ❓ Neutral/Insufficient Sources ({len(neutral)})\n"
                for source in neutral:
                    report += f"- **[{source['title']}]({source['url']})**\n\n"
        else:
            report += f"## 🔗 Sources\n\nNo sources found.\n\n"
        
        report += "---\n\n"
        
        # Scoring Breakdown
        report += f"## 📊 Scoring Breakdown\n\n"
        breakdown = scoring["breakdown"]
        calc = scoring["calculation"]
        
        report += f"**Evidence Summary:**\n"
        report += f"- Supporting Evidence: {breakdown['supporting_evidence']['count']} sources\n"
        report += f"- Refuting Evidence: {breakdown['refuting_evidence']['count']} sources\n"
        report += f"- Insufficient Evidence: {breakdown['insufficient_evidence']['count']} sources\n\n"
        
        report += f"**Calculation:**\n"
        report += f"- Raw Score: {calc['raw_score']}\n"
        report += f"- Normalized Score: {calc['normalized_score']} {calc['score_range']}\n\n"
        
        if scoring["critical_factors"]:
            report += f"**Critical Factors:**\n"
            for factor in scoring["critical_factors"]:
                report += f"- {factor}\n"
            report += "\n"
        
        report += "---\n\n"
        
        report += "## 🔍 Methodology\n\n"
        report += """This fact-check was performed using:
- **FAISS Knowledge Base:** Semantic search on verified information
- **Google Search:** Real-time web verification with 1.5x weight
- **AI-Powered Evaluation:** Automated analysis of supporting/refuting evidence
- **Weighted Scoring:** Evidence aggregation with source prioritization
- **Confidence Calculation:** Based on evidence consensus and data quality
"""
        
        return report

    # ============ HELPER METHODS (existing) ============
    
    def _format_verdict(self, verdict: str, confidence: float, score: float) -> Dict:
        """Format verdict with confidence percentage."""
        verdict_text = verdict.strip()
        
        if verdict_text.lower() == "true":
            category = "True"
            color = "green"
        elif verdict_text.lower() == "false":
            category = "False"
            color = "red"
        elif verdict_text.lower() == "inconclusive":
            category = "Inconclusive"
            color = "gray"
        else:
            category = "Unverified"
            color = "gray"
        
        return {
            "category": category,
            "confidence_percentage": round(confidence * 100, 1),
            "confidence_level": self._confidence_label(confidence),
            "color": color,
            "full_verdict": verdict_text
        }
    
    def _confidence_label(self, confidence: float) -> str:
        """Convert confidence score to human-readable label."""
        if confidence >= 0.85:
            return "Very High"
        elif confidence >= 0.70:
            return "High"
        elif confidence >= 0.50:
            return "Moderate"
        elif confidence >= 0.30:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_explanation(self, verdict_result: Dict, 
                             evaluations: List[Dict], 
                             breakdown: Dict) -> str:
        """Generate human-readable explanation of the verdict."""
        verdict = verdict_result["category"]
        confidence = verdict_result["confidence_percentage"]
        
        supports_count = breakdown.get("SUPPORTS", 0)
        refutes_count = breakdown.get("REFUTES", 0)
        neutral_count = breakdown.get("NOT_ENOUGH_INFO", 0)
        
        explanation = f"Based on the verification process with {len(evaluations)} sources reviewed:\n\n"
        
        if refutes_count > 0:
            explanation += f"• **{refutes_count} source(s) contradict or refute this claim**\n"
        
        if supports_count > 0:
            explanation += f"• **{supports_count} source(s) support this claim**\n"
        
        if neutral_count > 0:
            explanation += f"• **{neutral_count} source(s) provide insufficient information**\n"
        
        explanation += f"\n**Verdict: {verdict}** ({confidence}% confidence)\n\n"
        
        if refutes_count > supports_count * 1.5:
            explanation += "The preponderance of evidence contradicts this claim."
        elif supports_count > refutes_count * 1.5:
            explanation += "The weight of evidence supports this claim."
        elif supports_count > 0 or refutes_count > 0:
            explanation += "Evidence is mixed or conflicting."
        else:
            explanation += "There is insufficient information to verify this claim."
        
        return explanation
    
    def _extract_sources(self, evaluations: List[Dict]) -> List[Dict]:
        """Extract and format sources from evaluations."""
        sources = []
        seen_urls = set()
        
        for eval_item in evaluations:
            evidence = eval_item.get("evidence", {})
            
            if isinstance(evidence, dict):
                source_url = evidence.get("source", "")
                content = evidence.get("content", "")
                label = eval_item.get("label", "NOT_ENOUGH_INFO")
            else:
                source_url = ""
                content = str(evidence)[:200]
                label = eval_item.get("label", "NOT_ENOUGH_INFO")
            
            if source_url and source_url in seen_urls:
                continue
            
            if source_url:
                seen_urls.add(source_url)
                sources.append({
                    "url": source_url,
                    "title": self._extract_title(source_url),
                    "snippet": content[:150] if content else "",
                    "evidence_type": label,
                    "supports": label == "SUPPORTS",
                    "refutes": label == "REFUTES"
                })
        
        logger.warning("✅ Extracted %d unique sources", len(sources))
        return sources
    
    def _extract_title(self, url: str) -> str:
        """Extract a title from URL or use domain name."""
        if not url:
            return "Unknown Source"
        
        url_clean = url.replace("https://", "").replace("http://", "")
        domain = url_clean.split("/")[0].replace("www.", "")
        
        return domain.title()
    
    def _generate_scoring_report(self, breakdown: Dict, evaluations: List[Dict],
                                normalized_score: float, raw_score: float,
                                total_weight: float) -> Dict:
        """Generate detailed scoring breakdown explanation."""
        supports = breakdown.get("SUPPORTS", 0)
        refutes = breakdown.get("REFUTES", 0)
        neutral = breakdown.get("NOT_ENOUGH_INFO", 0)
        
        total_evidence = supports + refutes
        
        report = {
            "summary": "How the score was calculated:",
            "breakdown": {
                "supporting_evidence": {
                    "count": supports,
                    "description": f"Number of sources supporting the claim",
                    "weight": "1.5x for web sources, 1.0x for FAISS"
                },
                "refuting_evidence": {
                    "count": refutes,
                    "description": f"Number of sources contradicting the claim",
                    "weight": "1.5x boost for web sources"
                },
                "insufficient_evidence": {
                    "count": neutral,
                    "description": "Sources that were too vague"
                }
            },
            "calculation": {
                "method": "Weighted scoring with refutation prioritization",
                "raw_score": round(raw_score, 2),
                "total_weighted_evidence": round(total_weight, 2),
                "normalized_score": round(normalized_score, 3),
                "score_range": "[-1.0 (completely false) to +1.0 (completely true)]"
            },
            "critical_factors": self._identify_critical_factors(breakdown, total_evidence)
        }
        
        return report
    
    def _identify_critical_factors(self, breakdown: Dict, total_evidence: int) -> List[str]:
        """Identify critical factors that influenced the verdict."""
        factors = []
        
        supports = breakdown.get("SUPPORTS", 0)
        refutes = breakdown.get("REFUTES", 0)
        
        if refutes > 0:
            factors.append(f"Presence of {refutes} refuting source(s) significantly impacts verdict")
        
        if supports > refutes * 1.5:
            factors.append(f"Strong consensus: {supports} supporting sources vs {refutes} refuting")
        
        if total_evidence < 3:
            factors.append("Limited evidence available reduces confidence")
        
        if total_evidence == 0:
            factors.append("No verifiable evidence found")
        
        return factors if factors else ["Evaluation based on available evidence"]