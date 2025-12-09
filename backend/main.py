# ==============================================================================
# FILE 9: backend/main.py (REPLACE EXISTING)
# ==============================================================================

import sys
import asyncio
import time
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
    print("🎯 Fake News Detection - ADK Multi-Agent System")
    print("="*70)
    print("\n✅ Using Google ADK LlmAgent Framework:")
    print("   • Root Orchestrator coordinates pipeline")
    print("   • 5 LlmAgent sub-agents in sequential execution")
    print("   • Native ADK tool management")
    print("   • Async/await for efficient execution\n")
    print("Enter URL or text to fact-check.")
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
        
        logger.warning("🔍 Query received: %s", user_input[:80])
        print("\n⏳ Processing with ADK agents...\n")
        
        result = await root_orchestrator.run_pipeline(user_input, session_id=session_id)
        
        if result["success"]:
            print(result.get("report", "No report"))
            print(f"\n✅ Complete in {result['execution_time_ms']:.0f}ms\n")
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