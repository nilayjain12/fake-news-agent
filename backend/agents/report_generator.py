# backend/agents/report_generator.py - FINAL FIX
"""
Report Generator Agent - Pure ADK LlmAgent
Uses {placeholder} syntax to read from session state
"""
from google.adk.agents import LlmAgent
from config import ADK_MODEL_NAME, get_logger

logger = get_logger(__name__)

def create_report_agent() -> LlmAgent:
    """Create ADK LlmAgent for report generation"""
    model = ADK_MODEL_NAME
    
    agent = LlmAgent(
        name="report_agent",
        model=model,
        instruction="""You are a professional report writer for fact-checking.

You will receive:
- Claim: {extracted_claim}
- Evidence Summary: {evidence}
- Verification: {verification_result}
- Verdict: {aggregation_result}

Create a comprehensive markdown report:

# 📋 Fact-Check Report

**Date:** [Today's date]
**Claim:** [The claim]
**Verdict:** **[TRUE/FALSE/INCONCLUSIVE]** ✅/❌/❓
**Confidence:** [X%] ([Very High/High/Moderate/Low])

---

## 1. Summary

[2-3 sentence summary of verdict and reasoning]

---

## 2. Evidence Analysis

**Total Evidence:** [X from FAISS + Y from Google]

**Supporting:** [X sources]
[What supporting evidence says]

**Refuting:** [Y sources]  
[What refuting evidence says]

---

## 3. Reasoning

[Detailed explanation from verdict reasoning]
[Confidence rationale]
[Any caveats]

---

## 4. Conclusion

**Final Verdict:** [Plain language verdict]

[One sentence conclusion]

---

## 5. Methodology

- **FAISS Knowledge Base:** Semantic search on verified info
- **Google Search:** Real-time web verification
- **AI Evaluation:** Gemini 2.5 Flash

CONFIDENCE LEVELS:
- 0.85+: Very High
- 0.70-0.84: High
- 0.50-0.69: Moderate
- <0.50: Low

Use markdown formatting. Be clear and professional.""",
        output_key="final_report"
    )
    
    logger.warning("✅ Report Agent created")
    return agent