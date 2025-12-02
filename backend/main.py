# backend/main.py - FIXED VERSION
"""
Main entry point using working Google ADK Agent implementation
Demonstrates capstone requirements without import errors
"""

import sys
import asyncio
import time
import difflib
from pathlib import Path

# Set up Python path
BACKEND_PATH = Path(__file__).parent
sys.path.insert(0, str(BACKEND_PATH))

from agents.fact_check_agent_adk import FactCheckSequentialAgent
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)


def extract_confidence_from_verdict(verdict_str: str) -> float:
    """Extract confidence level from verdict string"""
    if not verdict_str:
        return 0.5
    
    verdict_lower = verdict_str.lower()
    
    if "error" in verdict_lower:
        return 0.0
    elif "false" in verdict_lower and "mostly" not in verdict_lower:
        return 0.1
    elif "mostly false" in verdict_lower:
        return 0.3
    elif "inconclusive" in verdict_lower or "mixed" in verdict_lower:
        return 0.5
    elif "mostly true" in verdict_lower:
        return 0.75
    elif "true" in verdict_lower and "false" not in verdict_lower:
        return 0.9
    else:
        return 0.5


def find_similar_cached_claim(query: str, memory: MemoryManager) -> dict:
    """Find similar claim in cache using string similarity"""
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
        
        if best_ratio > 0.85:
            logger.warning("✨ Found similar cached claim (%.0f%% match)", best_ratio * 100)
            return best_match
        
        logger.warning("📭 No similar cached claims (best: %.0f%%, need >85%%)", best_ratio * 100)
        return None
        
    except Exception as e:
        logger.warning("⚠️  Error searching cache: %s", str(e)[:100])
        return None


async def main_async():
    """
    Main async entry point demonstrating ADK agents
    
    This demonstrates:
    ✅ Multiple specialized agents (5 agents)
    ✅ Sequential orchestration
    ✅ Async/Await for efficient execution
    ✅ Session Management & Memory
    ✅ Evidence retrieval from multiple sources
    """
    
    agent = FactCheckSequentialAgent()
    memory = MemoryManager()
    
    session_id = "cli-session"
    memory.create_session(session_id, user_id="cli-user")
    
    print("\n" + "="*70)
    print("🎯 Fact-Check Agent with Google ADK Agents")
    print("="*70)
    print("\nCapstone Features Demonstrated:")
    print("✅ Multi-Agent System (5 specialized agents in sequence)")
    print("   1. Ingestion Agent - processes input")
    print("   2. Claim Extraction Agent - identifies main claim")
    print("   3. Verification Agent - searches for evidence")
    print("   4. Aggregator Agent - processes evidence")
    print("   5. Report Agent - generates comprehensive report")
    print("\n✅ Evidence Retrieval (FAISS + Google Search)")
    print("✅ Sessions & Memory (Persistent SQLite storage)")
    print("✅ Async Execution (Non-blocking operations)")
    print("\n" + "="*70)
    print("\nEnter URL or text to fact-check.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("📝 Enter claim or URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        
        processed_input = agent.preprocess_input(user_input)
        logger.warning("🔍 Query received: %s", user_input[:80])
        
        print("\n⏳ Processing your request...\n")
        
        start_time = time.time()
        
        try:
            # ==========================================
            # STEP 1: CHECK CACHE FIRST
            # ==========================================
            logger.warning("🔎 Checking memory cache...")
            cached_claim = find_similar_cached_claim(user_input, memory)
            
            if cached_claim:
                # Cache hit - return immediately
                logger.warning("✅ Cache hit! Using cached result")
                execution_time = (time.time() - start_time) * 1000
                
                print("─" * 70)
                print("\n### 📊 Fact-Check Report: Cached Result\n")
                print(f"**Status:** ✨ Retrieved from memory cache")
                print(f"**Query:** {user_input}\n")
                print(f"**Cached Claim:** {cached_claim['claim_text']}\n")
                print(f"**Verdict:** {cached_claim['verdict']}\n")
                print(f"**Confidence:** {cached_claim['confidence']:.1%}\n")
                print(f"**Execution Time:** {execution_time:.0f}ms ⚡\n")
                print("─" * 70 + "\n")
                
                final_verdict = cached_claim["verdict"]
            
            else:
                # ==========================================
                # STEP 2: RUN ADK AGENT PIPELINE
                # ==========================================
                logger.warning("📭 No cache hit, running ADK agent pipeline...")
                print("─" * 70)
                print("\n🚀 Running ADK Agent Pipeline:\n")
                print("   Stage 1: Ingestion Agent → Processing input")
                print("   Stage 2: Claim Extraction Agent → Identifying main claim")
                print("   Stage 3: Verification Agent → Searching for evidence")
                print("   Stage 4: Aggregator Agent → Processing evidence")
                print("   Stage 5: Report Agent → Generating report\n")
                print("─" * 70 + "\n")
                
                # Run the ADK agent pipeline
                result = await agent.run_fact_check_async(processed_input)
                
                final_verdict = result.get("overall_verdict", "UNKNOWN")
                execution_time = result.get("execution_time_ms", 0)
                report = result.get("comprehensive_report", "No report generated")
                
                # Print the report
                print(report)
                
                logger.warning("⏱️  Pipeline execution time: %.0f ms", execution_time)
                
                print("\n" + "─" * 70)
                print(f"✅ Verification Complete in {execution_time:.0f}ms\n")
                
                # ==========================================
                # STEP 3: CACHE THE RESULT
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
                        logger.warning("💾 Result cached for future queries")
                    except Exception as e:
                        logger.warning("⚠️  Failed to cache: %s", str(e)[:100])
            
            # ==========================================
            # STEP 4: LOG INTERACTION
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
                logger.warning("⚠️  Failed to log: %s", str(e)[:50])
            
            # ==========================================
            # STEP 5: DISPLAY STATISTICS
            # ==========================================
            try:
                stats = memory.get_all_stats()
                print(f"📊 System Statistics:")
                print(f"   • Verified claims in memory: {stats['total_verified_claims']}")
                print(f"   • Average confidence: {stats['average_confidence']:.1%}")
                print(f"   • Total sessions: {stats['total_sessions']}")
                if stats['verdict_distribution']:
                    print(f"   • Verdicts: {stats['verdict_distribution']}")
            except Exception as e:
                logger.warning("⚠️  Could not display stats: %s", str(e)[:50])
            
            print()
        
        except Exception as e:
            logger.warning("❌ Error: %s", str(e)[:100])
            print(f"\n❌ Error during processing: {str(e)[:200]}\n")


def main():
    """Entry point"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        logger.warning("❌ Fatal error: %s", str(e)[:100])
        sys.exit(1)


if __name__ == "__main__":
    main()