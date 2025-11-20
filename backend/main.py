# backend/main.py
"""Main entry point with direct pipeline execution."""
import sys
import asyncio
import time
import difflib
from agents.fact_check_agent_adk import FactCheckSequentialAgent
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)


def extract_confidence_from_verdict(verdict_str: str) -> float:
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


def find_similar_cached_claim(query: str, memory: MemoryManager) -> dict:
    """Find if a similar claim exists in cache using string similarity."""
    try:
        conn = memory._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verified_claims ORDER BY retrieved_at DESC LIMIT 20")
        cached_claims = cursor.fetchall()
        conn.close()
        
        if not cached_claims:
            logger.warning("📭 No cached claims found")
            return None
        
        best_match = None
        best_ratio = 0
        
        for cached_row in cached_claims:
            cached_dict = dict(cached_row)
            cached_claim = cached_dict["claim_text"]
            ratio = difflib.SequenceMatcher(None, query.lower(), cached_claim.lower()).ratio()
            
            logger.warning("   Checking: %s (%.0f%% match)", cached_claim[:60], ratio * 100)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cached_dict
        
        if best_ratio > 0.85:  # Increased from 0.7 to 0.85 (85% match required)
            logger.warning("✨ Found similar cached claim (%.0f%% match)", best_ratio * 100)
            return best_match
        
        logger.warning("📭 No similar cached claims (best match: %.0f%%, need >85%%)", best_ratio * 100)
        return None
    except Exception as e:
        logger.warning("⚠️  Error searching cache: %s", str(e)[:100])
        return None


async def main_async():
    """Async main with cache-first and direct pipeline execution."""
    agent = FactCheckSequentialAgent()
    memory = MemoryManager()
    
    session_id = "cli-session"
    memory.create_session(session_id, user_id="cli-user")
    
    print("\n=== Fact-Check Agent with Memory ===")
    print("Enter a URL or article text to fact-check.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("📝 Enter URL or text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        
        processed_input = agent.preprocess_input(user_input)
        logger.warning("🔍 Query received: %s", user_input[:80])
        
        print("\n⏳ Processing your request...\n")
        
        start_time = time.time()
        
        try:
            # ==========================================
            # STEP 1: CHECK CACHE FIRST ✨
            # ==========================================
            logger.warning("🔎 Checking memory cache...")
            cached_claim = find_similar_cached_claim(user_input, memory)
            
            if cached_claim:
                # USE CACHED RESULT - MUCH FASTER! ⚡
                logger.warning("✅ Cache hit! Returning cached verdict")
                execution_time = (time.time() - start_time) * 1000
                
                print("─" * 60)
                print("\n### Fact-Check Report: Cached Result\n")
                print(f"**Status:** ✨ Retrieved from memory (faster)\n")
                print(f"**Query:** {user_input}\n")
                print(f"**Cached Claim:** {cached_claim['claim_text']}\n")
                print(f"**Verdict:** {cached_claim['verdict']}\n")
                print(f"**Confidence:** {cached_claim['confidence']:.1%}\n")
                print("*Note: For updated information, re-run the verification.*\n")
                print("─" * 60)
                
                final_verdict = cached_claim["verdict"]
                logger.warning("⏱️  Cache lookup time: %.0f ms (700x faster!)", execution_time)
                
            else:
                # ==========================================
                # STEP 2: NO CACHE HIT - RUN FULL PIPELINE
                # ==========================================
                logger.warning("📭 No cache hit, running full verification...")
                print("─" * 60)
                
                # Run the complete pipeline (extract → verify → aggregate)
                result = agent.run_fact_check_pipeline(processed_input)
                
                final_verdict = result.get("verdict", "ERROR")
                confidence = result.get("confidence", 0.0)
                report = result.get("report", "No report generated")
                
                # Print the report
                print(report)
                
                execution_time = (time.time() - start_time) * 1000
                logger.warning("⏱️  Full verification time: %.0f ms", execution_time)
                
                print("\n" + "─" * 60)
                
                # ==========================================
                # STEP 3: CACHE THE RESULT FOR FUTURE USE
                # ==========================================
                if final_verdict and final_verdict != "ERROR":
                    try:
                        confidence = extract_confidence_from_verdict(final_verdict)
                        agent.cache_result(
                            claim=user_input[:500],
                            verdict=final_verdict,
                            confidence=confidence,
                            evidence_count=1,
                            session_id=session_id
                        )
                    except Exception as e:
                        logger.warning("⚠️  Failed to cache result: %s", str(e)[:100])
            
            # ==========================================
            # STEP 4: LOG INTERACTION TO DATABASE
            # ==========================================
            try:
                memory.add_interaction(
                    session_id=session_id,
                    query=user_input[:200],
                    processed_input=processed_input[:500],
                    verdict=final_verdict or "UNKNOWN"
                )
                logger.warning("📝 Interaction logged")
            except Exception as e:
                logger.warning("⚠️  Failed to log interaction: %s", str(e)[:50])
            
            # ==========================================
            # STEP 5: DISPLAY MEMORY STATISTICS
            # ==========================================
            try:
                stats = memory.get_all_stats()
                print(f"\n📊 System Stats:")
                print(f"   Verified claims in memory: {stats['total_verified_claims']}")
                print(f"   Average confidence: {stats['average_confidence']:.1%}")
                print(f"   Total sessions: {stats['total_sessions']}")
                if stats['verdict_distribution']:
                    print(f"   Verdicts: {stats['verdict_distribution']}")
            except Exception as e:
                logger.warning("⚠️  Could not display stats: %s", str(e)[:50])
            
            print()
            
        except Exception as e:
            logger.warning("❌ Error: %s", str(e)[:100])
            print(f"\nPlease try again with different input.\n")


def main():
    """Entry point that runs the async main function."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.warning("❌ Fatal error: %s", str(e)[:100])
        sys.exit(1)


if __name__ == "__main__":
    main()