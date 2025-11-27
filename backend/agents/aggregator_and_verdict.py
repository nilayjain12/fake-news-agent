# backend/agents/aggregator_and_verdict.py
"""
Advanced LLM-based verdict agent using Google ADK for robust evidence aggregation.
FIXED VERSION: Vote counting first, then credibility-weighted analysis.
"""
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import json
import os
from typing import List, Dict
from dataclasses import dataclass

logger = get_logger(__name__)
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class EvidenceScore:
    """Structured evidence scoring result"""
    evidence_index: int
    label: str  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
    confidence: float  # 0.0-1.0: How confident in this label
    reasoning: str  # Why this label was assigned
    relevance_score: float  # 0.0-1.0: How relevant to claim
    source_credibility: float  # 0.0-1.0: Source credibility assessment
    contradiction_strength: float  # 0.0-1.0: If refutes, how strong is contradiction


@dataclass
class VerdictResult:
    """Final verdict with detailed breakdown"""
    verdict: str  # True, False, Mostly True, Mostly False, Inconclusive
    confidence: float  # 0.0-1.0
    reasoning: str  # Explanation of verdict
    evidence_breakdown: Dict  # Detailed breakdown
    critical_evidence: List[Dict]  # Most important pieces
    potential_biases: List[str]  # Potential issues identified
    recommendation: str  # How to use this verdict


class AdvancedVerdictAgent:
    """
    Robust verdict agent using LLM for intelligent evidence aggregation.
    
    FIXED: Vote counting first as primary factor, then credibility-weighted secondary.
    
    Key improvements:
    1. Vote counting as PRIMARY factor
    2. Credibility-weighted analysis for edge cases
    3. Smart contradiction detection
    4. Nuanced verdict generation (not binary)
    5. Bias detection
    """
    
    def __init__(self):
        self.model = ADK_MODEL_NAME
        logger.warning("🚀 AdvancedVerdictAgent initialized (FIXED VERSION)")
    
    def aggregate_with_advanced_scoring(self, claim: str, evaluations: List[Dict]) -> VerdictResult:
        """
        Main entry point: Aggregate evaluations using advanced LLM-based scoring.
        
        FIXED: Vote counting first, then weighted analysis.
        
        Args:
            claim: The original claim being verified
            evaluations: List of evaluations with evidence
            
        Returns:
            VerdictResult with detailed breakdown
        """
        logger.warning("📊 Starting advanced evidence aggregation for: %s", claim[:60])
        
        if not evaluations:
            logger.warning("⚠️  No evaluations provided")
            return VerdictResult(
                verdict="Unverified",
                confidence=0.0,
                reasoning="No evidence available for analysis",
                evidence_breakdown={"SUPPORTS": 0, "REFUTES": 0, "NOT_ENOUGH_INFO": 0},
                critical_evidence=[],
                potential_biases=["No evidence to assess"],
                recommendation="Claim could not be verified due to lack of available evidence"
            )
        
        logger.warning("📝 Processing %d evaluations", len(evaluations))
        
        # STEP 1: VOTE COUNTING (PRIMARY FACTOR)
        logger.warning("Step 1️⃣: Vote counting (PRIMARY)...")
        vote_result = self._vote_counting(evaluations)
        logger.warning("   Votes - SUPPORTS: %d, REFUTES: %d, NOT_ENOUGH: %d",
                      vote_result["supports"], vote_result["refutes"], vote_result["not_enough"])
        
        # STEP 2: Deep analysis of each evidence piece
        logger.warning("Step 2️⃣: Analyzing individual evidence pieces...")
        scored_evidence = self._score_individual_evidence(claim, evaluations)
        
        # STEP 3: Credibility-weighted analysis (SECONDARY FACTOR)
        logger.warning("Step 3️⃣: Credibility-weighted analysis...")
        weighted_result = self._calculate_weighted_scores(scored_evidence)
        logger.warning("   Weighted - SUPPORTS: %.2f, REFUTES: %.2f",
                      weighted_result["supports_weighted"], weighted_result["refutes_weighted"])
        
        # STEP 4: Contradiction strength analysis
        logger.warning("Step 4️⃣: Analyzing contradiction strength...")
        contradiction_analysis = self._analyze_contradictions(scored_evidence)
        
        # STEP 5: Generate verdict (VOTE COUNTING FIRST, then credibility)
        logger.warning("Step 5️⃣: Generating final verdict...")
        verdict = self._generate_intelligent_verdict(
            claim=claim,
            vote_result=vote_result,
            weighted_result=weighted_result,
            scored_evidence=scored_evidence,
            contradiction_analysis=contradiction_analysis
        )
        
        logger.warning("✅ Verdict generated: %s (confidence: %.1f%%)", 
                      verdict.verdict, verdict.confidence * 100)
        
        return verdict
    
    def _vote_counting(self, evaluations: List[Dict]) -> Dict:
        """
        Simple vote counting - PRIMARY FACTOR.
        Returns count of SUPPORTS, REFUTES, NOT_ENOUGH_INFO.
        """
        supports = sum(1 for e in evaluations if e.get("label") == "SUPPORTS")
        refutes = sum(1 for e in evaluations if e.get("label") == "REFUTES")
        not_enough = sum(1 for e in evaluations if e.get("label") == "NOT_ENOUGH_INFO")
        total = len(evaluations)
        
        return {
            "supports": supports,
            "refutes": refutes,
            "not_enough": not_enough,
            "total": total,
            "supports_pct": supports / total if total > 0 else 0,
            "refutes_pct": refutes / total if total > 0 else 0
        }
    
    def _score_individual_evidence(self, claim: str, evaluations: List[Dict]) -> List[EvidenceScore]:
        """
        Use LLM to deeply analyze each evidence piece.
        Goes beyond simple SUPPORTS/REFUTES labels.
        """
        logger.warning("🔍 Deep-scoring %d evidence items...", len(evaluations))
        
        scored_list = []
        
        for idx, eval_item in enumerate(evaluations, 1):
            try:
                evidence = eval_item.get("evidence", {})
                original_label = eval_item.get("label", "NOT_ENOUGH_INFO")
                
                # Extract evidence content
                if isinstance(evidence, dict):
                    content = evidence.get("content", str(evidence))
                    source = evidence.get("source", "unknown")
                else:
                    content = str(evidence)
                    source = "unknown"
                
                # Use LLM to deeply analyze this evidence
                score = self._deep_analyze_evidence(
                    claim=claim,
                    evidence_content=content,
                    evidence_source=source,
                    original_label=original_label,
                    evidence_index=idx
                )
                
                scored_list.append(score)
                logger.warning("   [%d] %s (confidence: %.2f, relevance: %.2f, credibility: %.2f)", 
                             idx, score.label, score.confidence, score.relevance_score, 
                             score.source_credibility)
                
            except Exception as e:
                logger.warning("   ⚠️  Error scoring evidence %d: %s", idx, str(e)[:50])
                # Fallback: use original label with neutral credibility
                scored_list.append(EvidenceScore(
                    evidence_index=idx,
                    label=original_label,
                    confidence=0.5,
                    reasoning="Error in analysis",
                    relevance_score=0.5,
                    source_credibility=0.5,
                    contradiction_strength=0.0
                ))
                continue
        
        logger.warning("✅ Scored %d evidence items", len(scored_list))
        return scored_list
    
    def _deep_analyze_evidence(self, claim: str, evidence_content: str, 
                               evidence_source: str, original_label: str, 
                               evidence_index: int) -> EvidenceScore:
        """
        Use Gemini to perform deep contextual analysis of a single evidence piece.
        IMPORTANT: Validate the original label assignment.
        """
        prompt = f"""You are an expert fact-checker. Analyze this evidence piece in depth.

CLAIM TO VERIFY:
"{claim}"

EVIDENCE PIECE:
Source: {evidence_source}
Content: {evidence_content[:500]}

ORIGINAL_LABEL: {original_label}

TASK: Provide a detailed analysis:

1. **Label Validation**: Is the original label CORRECT?
   - SUPPORTS: Does evidence support/confirm the claim?
   - REFUTES: Does evidence contradict/deny the claim?
   - NOT_ENOUGH_INFO: Is evidence unclear or irrelevant?
   - Suggest correct label if wrong.

2. **Confidence**: How certain in the label? (0.0-1.0)
   - 1.0 = completely clear, explicit
   - 0.5 = somewhat ambiguous
   - 0.1 = very unclear

3. **Relevance**: How relevant to claim? (0.0-1.0)
   - 1.0 = directly addresses claim
   - 0.5 = tangentially related
   - 0.1 = barely related

4. **Source Credibility**: Trustworthiness (0.0-1.0)
   - Academic/peer-reviewed = 0.95
   - Major news outlet = 0.85
   - Official news source = 0.80
   - Wikipedia = 0.75
   - Blog/social media = 0.30
   - Unknown = 0.50

5. **Contradiction Strength** (if refutes): (0.0-1.0)
   - 1.0 = Direct explicit contradiction
   - 0.5 = Indirect contradiction
   - 0.0 = Not a refutation

RESPONSE FORMAT (ONLY JSON):
{{
  "label": "SUPPORTS|REFUTES|NOT_ENOUGH_INFO",
  "label_correct": true,
  "confidence": 0.95,
  "relevance": 0.90,
  "source_credibility": 0.85,
  "contradiction_strength": 0.0,
  "reasoning": "Brief 1-2 sentence explanation"
}}"""
        
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Parse JSON
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            
            parsed = json.loads(json_str.strip())
            
            # Use the label from LLM analysis (it validates)
            final_label = parsed.get("label", original_label)
            
            return EvidenceScore(
                evidence_index=evidence_index,
                label=final_label,
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", ""),
                relevance_score=float(parsed.get("relevance", 0.5)),
                source_credibility=float(parsed.get("source_credibility", 0.5)),
                contradiction_strength=float(parsed.get("contradiction_strength", 0.0))
            )
            
        except Exception as e:
            logger.warning("⚠️  Error in deep analysis: %s", str(e)[:50])
            # Fallback: use original label
            return EvidenceScore(
                evidence_index=evidence_index,
                label=original_label,
                confidence=0.5,
                reasoning="Could not perform deep analysis",
                relevance_score=0.5,
                source_credibility=0.5,
                contradiction_strength=0.0
            )
    
    def _calculate_weighted_scores(self, scored_evidence: List[EvidenceScore]) -> Dict:
        """
        Calculate credibility-weighted scores (SECONDARY FACTOR).
        """
        supports_weighted = sum(
            e.confidence * e.relevance_score * e.source_credibility
            for e in scored_evidence if e.label == "SUPPORTS"
        )
        
        refutes_weighted = sum(
            e.confidence * e.relevance_score * e.source_credibility * e.contradiction_strength
            for e in scored_evidence if e.label == "REFUTES"
        )
        
        return {
            "supports_weighted": supports_weighted,
            "refutes_weighted": refutes_weighted,
            "total_weighted": supports_weighted + refutes_weighted
        }
    
    def _analyze_contradictions(self, scored_evidence: List[EvidenceScore]) -> Dict:
        """
        Analyze contradictions - are there strong refutations?
        """
        refutations = [e for e in scored_evidence if e.label == "REFUTES"]
        
        if not refutations:
            return {
                "has_strong_contradictions": False,
                "strong_contradiction_count": 0,
                "strongest_contradiction_strength": 0.0
            }
        
        strong_contradictions = [
            r for r in refutations
            if r.confidence > 0.7 and r.relevance_score > 0.7 and r.contradiction_strength > 0.7
        ]
        
        return {
            "has_strong_contradictions": len(strong_contradictions) > 0,
            "strong_contradiction_count": len(strong_contradictions),
            "strongest_contradiction_strength": max(
                (r.contradiction_strength for r in refutations), default=0.0
            )
        }
    
    def _generate_intelligent_verdict(self, claim: str, vote_result: Dict,
                                    weighted_result: Dict, scored_evidence: List[EvidenceScore],
                                    contradiction_analysis: Dict) -> VerdictResult:
        """
        Generate verdict using VOTE COUNTING as PRIMARY, credibility as SECONDARY.
        
        Priority order:
        1. Vote counting (SUPPORTS > REFUTES = True, etc.)
        2. Weighted analysis (if votes are close)
        3. Contradiction strength (if conflicting)
        """
        logger.warning("🎯 Generating intelligent verdict (VOTE COUNTING FIRST)...")
        
        supports = vote_result["supports"]
        refutes = vote_result["refutes"]
        not_enough = vote_result["not_enough"]
        total = vote_result["total"]
        
        logger.warning("   Vote counts: SUPPORTS=%d, REFUTES=%d, NOT_ENOUGH=%d", 
                      supports, refutes, not_enough)
        
        # RULE 1: Vote counting (PRIMARY)
        if supports > refutes:
            # More SUPPORTS than REFUTES
            if supports > total * 0.7:
                # Strong consensus for SUPPORTS
                verdict = "True"
                confidence = 0.95
                reasoning = f"Strong evidence consensus: {supports} sources support vs {refutes} refute."
            elif supports > total * 0.5:
                # Mild consensus for SUPPORTS
                if contradiction_analysis["has_strong_contradictions"]:
                    verdict = "Mostly True"
                    confidence = 0.70
                    reasoning = f"Majority support ({supports} vs {refutes}) but with notable contradictions."
                else:
                    verdict = "Mostly True"
                    confidence = 0.75
                    reasoning = f"Majority of sources support the claim ({supports} vs {refutes})."
            else:
                verdict = "Inconclusive"
                confidence = 0.50
                reasoning = f"Marginally more support ({supports} vs {refutes}) but too close to determine."
        
        elif refutes > supports:
            # More REFUTES than SUPPORTS
            if refutes > total * 0.7:
                # Strong consensus for REFUTES
                verdict = "False"
                confidence = 0.95
                reasoning = f"Strong evidence against claim: {refutes} sources refute vs {supports} support."
            elif refutes > total * 0.5:
                # Mild consensus for REFUTES
                if contradiction_analysis["has_strong_contradictions"]:
                    verdict = "Mostly False"
                    confidence = 0.70
                    reasoning = f"Majority refute the claim ({refutes} vs {supports})."
                else:
                    verdict = "Mostly False"
                    confidence = 0.75
                    reasoning = f"Majority of sources refute the claim ({refutes} vs {supports})."
            else:
                verdict = "Inconclusive"
                confidence = 0.50
                reasoning = f"Marginally more refutations ({refutes} vs {supports}) but unclear."
        
        else:
            # EQUAL votes - use weighted analysis as tiebreaker
            logger.warning("   Equal votes detected - using weighted analysis as tiebreaker")
            weighted_diff = weighted_result["supports_weighted"] - weighted_result["refutes_weighted"]
            
            if weighted_diff > 0:
                verdict = "Mostly True"
                confidence = 0.60
                reasoning = "Equal vote count, but credibility-weighted analysis slightly favors support."
            elif weighted_diff < 0:
                verdict = "Mostly False"
                confidence = 0.60
                reasoning = "Equal vote count, but credibility-weighted analysis slightly favors refutation."
            else:
                verdict = "Inconclusive"
                confidence = 0.50
                reasoning = "Evidence is perfectly balanced - cannot determine verdict."
        
        # Identify critical evidence
        critical_indices = [
            e.evidence_index - 1 for e in sorted(
                scored_evidence, 
                key=lambda x: x.confidence * x.relevance_score * x.source_credibility,
                reverse=True
            )[:2]
        ]
        
        critical_evidence = [
            {
                "index": idx,
                "label": scored_evidence[idx].label,
                "reasoning": scored_evidence[idx].reasoning,
                "confidence": scored_evidence[idx].confidence,
                "source_credibility": scored_evidence[idx].source_credibility
            }
            for idx in critical_indices if idx < len(scored_evidence)
        ]
        
        return VerdictResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_breakdown={
                "total_evidence": total,
                "supports": supports,
                "refutes": refutes,
                "not_enough": not_enough,
                "supports_pct": vote_result["supports_pct"],
                "refutes_pct": vote_result["refutes_pct"],
                "supports_weighted": weighted_result["supports_weighted"],
                "refutes_weighted": weighted_result["refutes_weighted"]
            },
            critical_evidence=critical_evidence,
            potential_biases=self._identify_potential_biases(vote_result, contradiction_analysis),
            recommendation=self._generate_recommendation(verdict, confidence)
        )
    
    def _identify_potential_biases(self, vote_result: Dict, 
                                  contradiction_analysis: Dict) -> List[str]:
        """Identify potential biases in the evidence set."""
        biases = []
        
        supports_pct = vote_result["supports_pct"]
        refutes_pct = vote_result["refutes_pct"]
        
        if supports_pct > 0.8 or refutes_pct > 0.8:
            biases.append("Evidence heavily skewed in one direction (may indicate selection bias)")
        
        if contradiction_analysis["has_strong_contradictions"] and vote_result["supports"] > vote_result["refutes"]:
            biases.append("Multiple strong contradictions exist despite majority support")
        
        if not biases:
            biases.append("No major biases detected")
        
        return biases
    
    def _generate_recommendation(self, verdict: str, confidence: float) -> str:
        """Generate usage recommendation based on verdict and confidence."""
        if confidence > 0.85:
            return f"High confidence verdict. Can be used reliably for decision-making."
        elif confidence > 0.65:
            return f"Moderate confidence verdict. Use with caution, consider additional sources."
        else:
            return f"Low confidence verdict. Inconclusive result, seek additional evidence."
    
    @staticmethod
    def _calculate_variance(values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


# Adapter function for backward compatibility
def aggregate_evaluations(evaluations: List[Dict]) -> Dict:
    """
    Adapter function that converts old heuristic aggregation to new advanced verdict agent.
    Maintains backward compatibility with existing code.
    """
    claim = "Unknown claim"
    
    agent = AdvancedVerdictAgent()
    verdict_result = agent.aggregate_with_advanced_scoring(claim, evaluations)
    
    # Convert to old format for compatibility
    return {
        "verdict": verdict_result.verdict,
        "score": _convert_verdict_to_score(verdict_result.verdict),
        "confidence": verdict_result.confidence,
        "raw_score": 0,
        "total_weight": len(evaluations),
        "breakdown": verdict_result.evidence_breakdown,
        "detailed_analysis": verdict_result
    }


def _convert_verdict_to_score(verdict: str) -> float:
    """Convert verdict string to numerical score (-1.0 to 1.0)."""
    verdict_lower = verdict.lower()
    if verdict_lower == "true":
        return 0.95
    elif verdict_lower == "mostly true":
        return 0.65
    elif verdict_lower == "inconclusive":
        return 0.0
    elif verdict_lower == "mostly false":
        return -0.65
    elif verdict_lower == "false":
        return -0.95
    else:
        return 0.0