# backend/agents/report_generator.py
"""Detailed report generator with comprehensive breakdown of fact-checking results."""
from config import get_logger
from typing import List, Dict
import json

logger = get_logger(__name__)


class DetailedReportGenerator:
    """Generates comprehensive fact-check reports with multiple sections."""
    
    def __init__(self):
        logger.warning("📊 DetailedReportGenerator initialized")
    
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
        
        # Convert verdict to simpler format with confidence
        verdict_result = self._format_verdict(verdict, confidence, score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            verdict_result, 
            evaluations, 
            breakdown
        )
        
        # Extract sources
        sources = self._extract_sources(evaluations)
        
        # Generate scoring breakdown
        scoring_report = self._generate_scoring_report(
            breakdown,
            evaluations,
            score,
            aggregation_result.get("raw_score", 0),
            aggregation_result.get("total_weight", 0)
        )
        
        report = {
            "claim": claim,
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

    def generate_comprehensive_report_single_claim(self, main_claim: str, claim_report: Dict) -> str:
        """Generate report for single claim (cleaner, simpler format)."""
        logger.warning("📋 Generating single-claim comprehensive report")
        
        result = claim_report["result"]
        sources = claim_report["sources"]
        scoring = claim_report["scoring_breakdown"]
        
        report = "# 📋 Fact-Check Report\n\n"
        
        # Overall Verdict Box
        report += f"## ✅ Result\n\n"
        report += f"**Claim:** {main_claim}\n\n"
        report += f"**Verdict:** {result['category']}\n\n"
        report += f"**Confidence:** {result['confidence_percentage']}% ({result['confidence_level']})\n\n"
        
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

    def _format_verdict(self, verdict: str, confidence: float, score: float) -> Dict:
        """Format verdict with confidence percentage."""
        # Normalize verdict text
        verdict_text = verdict.strip()
        
        # Map to simple categories
        if "true" in verdict_text.lower() and "false" not in verdict_text.lower():
            if "mostly" in verdict_text.lower():
                category = "Mostly True"
                color = "orange"
            else:
                category = "True"
                color = "green"
        elif "false" in verdict_text.lower():
            if "mostly" in verdict_text.lower():
                category = "Mostly False"
                color = "orange"
            else:
                category = "False"
                color = "red"
        elif "mixed" in verdict_text.lower() or "inconclusive" in verdict_text.lower():
            category = "Mixed Evidence"
            color = "blue"
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
        
        # Add reasoning based on evidence balance
        if refutes_count > supports_count * 1.5:
            explanation += "The preponderance of evidence contradicts this claim. Multiple sources provide conflicting or contradictory information."
        elif supports_count > refutes_count * 1.5:
            explanation += "The weight of evidence supports this claim. Most sources corroborate the statement."
        elif supports_count > 0 or refutes_count > 0:
            explanation += "Evidence is mixed or conflicting. Some sources support while others contradict the claim."
        else:
            explanation += "There is insufficient information to verify this claim from available sources."
        
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
                # If evidence is not a dict, try to extract info
                source_url = ""
                content = str(evidence)[:200]
                label = eval_item.get("label", "NOT_ENOUGH_INFO")
            
            # Deduplicate by URL
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
        
        # Remove protocol
        url_clean = url.replace("https://", "").replace("http://", "")
        
        # Extract domain
        domain = url_clean.split("/")[0].replace("www.", "")
        
        # Capitalize
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
                    "weight": "1.5x boost for web sources (more authoritative for current facts)"
                },
                "insufficient_evidence": {
                    "count": neutral,
                    "description": "Sources that were too vague or didn't directly address the claim"
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
        neutral = breakdown.get("NOT_ENOUGH_INFO", 0)
        
        if refutes > 0:
            factors.append(f"Presence of {refutes} refuting source(s) significantly impacts verdict")
        
        if supports > refutes * 1.5:
            factors.append(f"Strong consensus: {supports} supporting sources vs {refutes} refuting")
        
        if total_evidence < 3:
            factors.append("Limited evidence available reduces confidence in verdict")
        
        if neutral > total_evidence:
            factors.append("High proportion of inconclusive evidence weakens confidence")
        
        if total_evidence == 0:
            factors.append("No verifiable evidence found - claim remains unverified")
        
        return factors if factors else ["Insufficient evidence to determine verdict"]
    
    def generate_comprehensive_report(self, claims: List[str], claim_reports: List[Dict]) -> str:
        """Generate comprehensive markdown report for all claims."""
        logger.warning("📋 Generating comprehensive fact-check report")
        
        report = "# 📋 Comprehensive Fact-Check Report\n\n"
        
        # Summary section
        true_count = sum(1 for r in claim_reports if "true" in r["result"]["category"].lower() and "false" not in r["result"]["category"].lower())
        false_count = sum(1 for r in claim_reports if "false" in r["result"]["category"].lower())
        mixed_count = len(claim_reports) - true_count - false_count
        
        report += "## 📊 Summary\n\n"
        report += f"- **Total Claims Analyzed:** {len(claim_reports)}\n"
        report += f"- **Verified as True:** {true_count}\n"
        report += f"- **Verified as False:** {false_count}\n"
        report += f"- **Mixed/Unverified:** {mixed_count}\n\n"
        
        # Individual claim reports
        report += "---\n\n"
        
        for idx, claim_report in enumerate(claim_reports, 1):
            report += f"## Claim {idx}: {claim_report['claim']}\n\n"
            
            # Result
            result = claim_report["result"]
            report += f"### ✅ Result\n\n"
            report += f"**Verdict:** {result['category']}  \n"
            report += f"**Confidence:** {result['confidence_percentage']}% ({result['confidence_level']})\n\n"
            
            # Explanation
            report += f"### 📝 Explanation\n\n"
            report += f"{claim_report['explanation']}\n\n"
            
            # Sources
            if claim_report["sources"]:
                report += f"### 🔗 Sources\n\n"
                for source in claim_report["sources"]:
                    badge = "✅" if source["supports"] else "❌" if source["refutes"] else "❓"
                    report += f"- {badge} [{source['title']}]({source['url']})\n"
                    if source["snippet"]:
                        report += f"  > {source['snippet'][:100]}...\n"
                report += "\n"
            
            # Scoring breakdown
            report += f"### 📊 Scoring Breakdown\n\n"
            scoring = claim_report["scoring_breakdown"]
            report += f"{scoring['summary']}\n\n"
            
            breakdown = scoring["breakdown"]
            report += f"**Supporting Evidence:** {breakdown['supporting_evidence']['count']} sources  \n"
            report += f"**Refuting Evidence:** {breakdown['refuting_evidence']['count']} sources  \n"
            report += f"**Insufficient Evidence:** {breakdown['insufficient_evidence']['count']} sources  \n\n"
            
            calc = scoring["calculation"]
            report += f"**Raw Score:** {calc['raw_score']}  \n"
            report += f"**Normalized Score:** {calc['normalized_score']} {calc['score_range']}\n\n"
            
            if scoring["critical_factors"]:
                report += f"**Critical Factors:**\n"
                for factor in scoring["critical_factors"]:
                    report += f"- {factor}\n"
                report += "\n"
            
            report += "---\n\n"
        
        report += "## 🔍 Methodology\n\n"
        report += """This fact-check was performed using:
- **FAISS Knowledge Base:** Semantic search on verified information
- **Google Search:** Real-time web verification
- **AI-Powered Evaluation:** Automated analysis of supporting/refuting evidence
- **Weighted Scoring:** Evidence aggregation with source prioritization
- **Confidence Calculation:** Based on evidence consensus and data quality
"""
        
        logger.warning("✅ Comprehensive report generated")
        return report