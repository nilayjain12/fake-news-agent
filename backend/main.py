# backend/main.py
"""
Main CLI Entry Point - Pure Google ADK Implementation
Demonstrates the complete ADK-based fact-checking pipeline
"""
import sys
import asyncio
from pathlib import Path

# Setup path
BACKEND_PATH = Path(__file__).parent
sys.path.insert(0, str(BACKEND_PATH))

from agents.adk_pipeline import create_fact_check_pipeline
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)


async def main_async():
    """
    Main async entry point demonstrating pure ADK pipeline
    
    Features:
    ✅ Complete ADK SequentialAgent pipeline
    ✅ 6 specialized agents (5 LlmAgent + 1 Custom Agent)
    ✅ Async/await for efficient execution
    ✅ Session management & memory caching
    ✅ Event streaming for progress updates
    ✅ Parallel evidence retrieval (FAISS + Google)
    """
    
    # Initialize pipeline
    pipeline = create_fact_check_pipeline()
    memory = pipeline.memory
    
    # Create session
    session_id = "cli-session"
    memory.create_session(session_id, user_id="cli-user")
    
    print("\n" + "="*70)
    print("🎯 Fact-Check Agent - Pure Google ADK Implementation")
    print("="*70)
    print("\n✨ ADK Features:")
    print("  • SequentialAgent orchestration")
    print("  • 5 LlmAgent + 1 Custom Agent")
    print("  • FunctionTool integration (FAISS + Google)")
    print("  • Session-based state management")
    print("  • Event streaming for real-time updates")
    print("  • Memory caching for fast repeated queries")
    print("\n" + "="*70)
    print("\n📝 Enter a claim to fact-check (or 'exit' to quit)\n")
    
    while True:
        try:
            user_input = input("Claim: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Exiting...")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        
        # Preprocess (URL extraction if needed)
        processed_input = pipeline.preprocess_input(user_input)
        
        print("\n⏳ Running ADK pipeline...\n")
        print("─" * 70)
        
        try:
            # Run the ADK pipeline
            result = await pipeline.verify_claim_async(
                processed_input,
                session_id=session_id
            )
            
            # Display results
            if result["success"]:
                print("\n" + result["comprehensive_report"])
                print("\n" + "─" * 70)
                print(f"✅ Completed in {result['execution_time_ms']:.0f}ms")
                print(f"📊 Verdict: {result['verdict']}")
                print(f"📈 Confidence: {result['confidence']:.1%}")
                
                # Cache result
                pipeline.cache_result(
                    claim=user_input[:500],
                    verdict=result["verdict"],
                    confidence=result["confidence"],
                    session_id=session_id
                )
                
                # Log interaction
                memory.add_interaction(
                    session_id=session_id,
                    query=user_input[:200],
                    processed_input=processed_input[:500],
                    verdict=result["verdict"]
                )
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            
            # Display statistics
            print("\n📊 Session Statistics:")
            stats = memory.get_all_stats()
            print(f"   • Total verified claims: {stats['total_verified_claims']}")
            print(f"   • Average confidence: {stats['average_confidence']:.1%}")
            if stats['verdict_distribution']:
                print(f"   • Verdicts: {stats['verdict_distribution']}")
            
            print()
            
        except Exception as e:
            logger.exception(f"❌ Error: {e}")
            print(f"\n❌ Error during processing: {str(e)[:200]}\n")


def main():
    """Entry point"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()