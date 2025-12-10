# ==============================================================================
# FILE: backend/main.py (Refactored for ADK)
# ==============================================================================

import sys
import asyncio
from pathlib import Path

BACKEND_PATH = Path(__file__).parent
sys.path.insert(0, str(BACKEND_PATH))

from agents.root_orchestrator import root_orchestrator
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)


async def main_async():
    """Main async entry point"""
    
    memory = MemoryManager()
    session_id = "cli-session"
    memory.create_session(session_id, user_id="cli-user")
    
    print("\n" + "="*70)
    print("🎯 Fact-Checking Agent - Google ADK Pipeline")
    print("="*70)
    print("\n✅ Architecture:")
    print("   Root Agent (SequentialAgent)")
    print("   ├─ Ingestion Agent (LlmAgent)")
    print("   ├─ Extraction Agent (LlmAgent)")
    print("   ├─ Verification Agent (SequentialAgent)")
    print("   │  ├─ Search Knowledge Agent (LlmAgent)")
    print("   │  ├─ Search Web Agent (LlmAgent)")
    print("   │  └─ Evaluate Evidence Agent (LlmAgent)")
    print("   ├─ Aggregation Agent (LlmAgent)")
    print("   └─ Report Agent (LlmAgent)")
    print("\n   All agents share session state for clean data flow")
    print("   Tools: FAISS, Google Search, Local Evaluation\n")
    
    print("Enter URL or text to fact-check.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("📝 Claim to verify: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break
        
        logger.warning("🔍 Query: %s", user_input[:80])
        print("\n⏳ Processing through ADK pipeline...\n")
        
        result = await root_orchestrator.process_query(
            user_input=user_input,
            session_id=session_id
        )
        
        if result["success"]:
            # Format output
            report = result.get("report", "No report generated")
            print(report)
            
            # Stats
            stats = root_orchestrator.get_stats()
            quota = stats["quota_status"]
            print(f"\n⏱️ Time: {result['execution_time_ms']:.0f}ms")
            print(f"📊 API Calls: {result['api_calls']}/20 ({quota['remaining']} remaining)")
            print(f"📚 Evidence Sources: {result['evidence_count']}\n")
        else:
            print(f"❌ Error: {result.get('error')}\n")


def main():
    """Entry point"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        logger.exception("❌ Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()