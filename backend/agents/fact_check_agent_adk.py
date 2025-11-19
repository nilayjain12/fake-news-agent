"""Top-level agent that ties ingestion → claim extraction → verification → aggregation → report."""
from google.adk.agents import SequentialAgent
from agents.ingestion_agent import IngestionAgent
from agents.claim_extraction_agent import ClaimExtractionAgent
from agents.verification_agent import VerificationAgent
from agents.aggregator_and_verdict import aggregate_evaluations
from config import get_logger

logger = get_logger(__name__)

class FactCheckSequentialAgent(SequentialAgent):
    """
    A SequentialAgent wrapper that orchestrates the simple pipeline:
      Ingestion -> Claim Extraction -> Verification -> Aggregation -> Report
    """

    def __init__(self):
        super().__init__(name="fact_check_sequential_agent")
        logger.info("Initializing FactCheckSequentialAgent")
        object.__setattr__(self, "ingest", IngestionAgent())
        object.__setattr__(self, "claim_extractor", ClaimExtractionAgent())
        object.__setattr__(self, "verifier", VerificationAgent())
        # no explicit Gemini here for verdict; aggregator does simple math

    def run(self, input_text_or_url: str):
        logger.info("FactCheckSequentialAgent.run called with input len=%d", len(input_text_or_url) if input_text_or_url else 0)
        # Step 1: ingest
        article_text = self.ingest.run(input_text_or_url)
        logger.debug("Ingested text len=%d", len(article_text) if article_text else 0)

        # Step 2: extract claims
        claims = self.claim_extractor.run(article_text)
        # Ensure a list
        if not isinstance(claims, list):
            claims = [claims]

        outputs = []
        for claim in claims:
            # Step 3: verify
            logger.debug("Verifying claim: %s", claim)
            result = self.verifier.run(claim)
            # Step 4: aggregate
            verdict = aggregate_evaluations(result.get("evaluations", []))
            outputs.append({
                "claim": claim,
                "verifier_result": result,
                "verdict": verdict
            })
        # Final report (list of per-claim verdicts)
        logger.info("Finished pipeline; claims=%d outputs=%d", len(claims), len(outputs))
        return outputs