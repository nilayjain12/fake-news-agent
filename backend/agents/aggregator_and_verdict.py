# backend/agents/aggregator_and_verdict.py - UPDATED
"""
UPDATED: Simple 3-level verdict system with binary evidence classification
Verdicts: TRUE, FALSE, INCONCLUSIVE
Evidence: SUPPORTS or REFUTES (binary only)
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
class VerdictResult:
    """Simple verdict result with 3-level classification"""
    verdict: str  # TRUE, FALSE, INCONCLUSIVE
    confidence: float  # 0.0-1.0
    reasoning: str  # Explanation
    supports_count: int  # Number of supporting evidence
    refutes_count: int  # Number of refuting evidence
    total_evidence: int  # Total evidence evaluated


class AdvancedVerdictAgent:
    """
    UPDATED: Simple 3-level verdict system
    - TRUE: More SUPPORTS than REFUTES
    - FALSE: More REFUTES than SUPPORTS
    - INCONCLUSIVE: Equal SUPPORTS and REFUTES
    
    Confidence based on evidence majority strength
    """
    
    def __init__(self):
        self.model = ADK_MODEL_NAME
        logger.warning("🚀 AdvancedVerdictAgent initialized (3-LEVEL SYSTEM)")

    def aggregate_with_advanced_scoring(self, claim: str, evaluations: List[Dict]) -> VerdictResult:
        """
        Main entry point: Aggregate binary evaluations into 3-level verdict
        
        Args:
            claim: The original claim being verified
            evaluations: List of evaluations with SUPPORTS/REFUTES labels only
            
        Returns:
            VerdictResult with 3-level verdict
        """
        logger.warning("📊 Starting evidence aggregation for: %s", claim[:60])
        
        if not evaluations:
            logger.warning("⚠️  No evaluations provided")
            return VerdictResult(
                verdict="INCONCLUSIVE",
                confidence=0.0,
                reasoning="No evidence available for analysis",
                supports_count=0,
                refutes_count=0,
                total_evidence=0
            )
        
        logger.warning("📝 Processing %d evaluations (BINARY: SUPPORTS/REFUTES)", len(evaluations))
        
        # STEP 1: Count evidence
        verdict_result = self._count_and_classify(evaluations)
        
        logger.warning("✅ Verdict: %s (confidence: %.1f%%)", 
                      verdict_result.verdict, verdict_result.confidence * 100)
        
        return verdict_result

    def _count_and_classify(self, evaluations: List[Dict]) -> VerdictResult:
        """
        Count SUPPORTS vs REFUTES and generate 3-level verdict
        
        Logic:
        - More SUPPORTS → TRUE
        - More REFUTES → FALSE
        - Equal → INCONCLUSIVE
        
        Confidence based on how strong the majority is
        """
        supports = 0
        refutes = 0
        
        for eval_item in evaluations:
            label = eval_item.get("label", "").upper()
            
            if label == "SUPPORTS":
                supports += 1
            elif label == "REFUTES":
                refutes += 1
            # Note: NO NOT_ENOUGH_INFO - binary classification only
        
        total = supports + refutes
        
        logger.warning("📊 Evidence count - SUPPORTS: %d, REFUTES: %d, TOTAL: %d", 
                      supports, refutes, total)
        
        # Calculate verdict and confidence
        if supports > refutes:
            verdict = "TRUE"
            confidence = self._calculate_confidence(supports, refutes, total)
            reasoning = self._generate_reasoning_true(supports, refutes, total)
            
        elif refutes > supports:
            verdict = "FALSE"
            confidence = self._calculate_confidence(refutes, supports, total)
            reasoning = self._generate_reasoning_false(supports, refutes, total)
            
        else:
            # Equal counts
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            reasoning = self._generate_reasoning_inconclusive(supports, refutes, total)
        
        logger.warning("   Verdict: %s, Confidence: %.2f", verdict, confidence)
        
        return VerdictResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            supports_count=supports,
            refutes_count=refutes,
            total_evidence=total
        )

    def _calculate_confidence(self, majority: int, minority: int, total: int) -> float:
        """
        Calculate confidence based on majority strength
        
        Confidence scale:
        - 90%+: Very high confidence (e.g., 9 supports vs 1 refute)
        - 70-90%: High confidence (e.g., 7 supports vs 3 refute)
        - 50-70%: Moderate confidence (e.g., 6 supports vs 4 refute)
        - 50%: No confidence (equal votes)
        
        Formula: (majority_pct - 50%) / 50% * scaling_factor + base
        """
        if total == 0:
            return 0.0
        
        majority_pct = majority / total
        
        # Simple formula: how much does majority exceed 50%?
        # 51% → 0.52 confidence
        # 70% → 0.75 confidence
        # 90% → 0.95 confidence
        
        if majority_pct >= 0.9:
            confidence = 0.95
        elif majority_pct >= 0.8:
            confidence = 0.88
        elif majority_pct >= 0.7:
            confidence = 0.78
        elif majority_pct >= 0.6:
            confidence = 0.68
        elif majority_pct > 0.5:
            confidence = min(0.55 + (majority_pct - 0.5) * 0.2, 0.65)
        else:
            confidence = 0.5
        
        logger.warning("      Confidence calculation: %d/%d = %.1f%% → confidence: %.2f",
                      majority, total, majority_pct * 100, confidence)
        
        return confidence

    def _generate_reasoning_true(self, supports: int, refutes: int, total: int) -> str:
        """Generate reasoning for TRUE verdict"""
        pct = (supports / total * 100) if total > 0 else 0
        
        if supports == total:
            return f"All {total} evidence sources support this claim with no contradictions."
        elif refutes == 0:
            return f"All {supports} evidence sources support this claim. No refuting evidence found."
        else:
            return f"{supports} sources support this claim vs {refutes} that refute it ({pct:.0f}% agreement)."

    def _generate_reasoning_false(self, supports: int, refutes: int, total: int) -> str:
        """Generate reasoning for FALSE verdict"""
        pct = (refutes / total * 100) if total > 0 else 0
        
        if refutes == total:
            return f"All {total} evidence sources refute this claim with no support."
        elif supports == 0:
            return f"All {refutes} evidence sources refute this claim. No supporting evidence found."
        else:
            return f"{refutes} sources refute this claim vs {supports} that support it ({pct:.0f}% refutation)."

    def _generate_reasoning_inconclusive(self, supports: int, refutes: int, total: int) -> str:
        """Generate reasoning for INCONCLUSIVE verdict"""
        if total == 0:
            return "No evidence available to determine verdict."
        else:
            return f"Evidence is evenly split: {supports} sources support and {supports} sources refute the claim. Requires additional investigation."

    def convert_to_old_format(self, verdict_result: VerdictResult) -> Dict:
        """
        Convert to old report format for backward compatibility with report generator
        """
        # Map 3-level verdicts to legacy format
        verdict_text = verdict_result.verdict
        
        # Convert to report-friendly format
        return {
            "verdict": verdict_text,
            "score": self._verdict_to_score(verdict_text),
            "confidence": verdict_result.confidence,
            "raw_score": 0,
            "total_weight": verdict_result.total_evidence,
            "breakdown": {
                "SUPPORTS": verdict_result.supports_count,
                "REFUTES": verdict_result.refutes_count,
                "NOT_ENOUGH_INFO": 0  # Binary classification - no middle ground
            },
            "reasoning": verdict_result.reasoning,
            "detailed_analysis": verdict_result
        }

    @staticmethod
    def _verdict_to_score(verdict: str) -> float:
        """Convert verdict to numerical score for backward compatibility"""
        verdict_lower = verdict.lower()
        
        if verdict_lower == "true":
            return 0.95
        elif verdict_lower == "false":
            return -0.95
        else:  # INCONCLUSIVE
            return 0.0


def aggregate_evaluations(evaluations: List[Dict]) -> Dict:
    """
    Adapter function for backward compatibility with existing code
    Converts binary evaluations to simple 3-level verdict
    """
    agent = AdvancedVerdictAgent()
    verdict_result = agent.aggregate_with_advanced_scoring("Unknown claim", evaluations)
    
    # Convert to old format
    return agent.convert_to_old_format(verdict_result)