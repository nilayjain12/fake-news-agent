# backend/main.py
"""Main entry point for CLI-run fact-check agent with memory integration."""
import sys
import asyncio
import time
from google.adk.runners import InMemoryRunner
from agents.fact_check_agent_adk import FactCheckSequentialAgent
from google.genai import types
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)


async def main_async():
    """Async main function to handle the agent interaction properly."""
    agent = FactCheckSequentialAgent()
    memory = MemoryManager()
    
    runner = InMemoryRunner(
        app_name="agents",
        agent=agent
    )
    
    session_id = "cli-session"
    memory.create_session(session_id, user_id="cli-user")
    
    await runner.session_service.create_session(
        app_name="agents",
        user_id="cli-user",
        session_id=session_id
    )
    
    print("\n=== Fact-Check Agent ===")
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
            message = types.Content(
                role="user",
                parts=[types.Part(text=processed_input)]
            )
            
            result_gen = runner.run(
                user_id="cli-user",
                session_id=session_id,
                new_message=message
            )
            
            print("─" * 60)
            final_verdict = None
            full_output = ""
            
            for event in result_gen:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        # Handle text output
                        if hasattr(part, 'text') and part.text:
                            text_content = part.text
                            print(text_content, end='', flush=True)
                            full_output += text_content
                            
                            # Extract verdict from output
                            if "**Verdict:**" in text_content or "VERDICT:" in text_content.upper():
                                # Simple extraction: look for TRUE/FALSE/MOSTLY TRUE/UNVERIFIED
                                lines = text_content.split('\n')
                                for line in lines:
                                    if 'FALSE' in line.upper() and final_verdict is None:
                                        final_verdict = "FALSE"
                                    elif 'MOSTLY TRUE' in line.upper() and final_verdict is None:
                                        final_verdict = "MOSTLY TRUE"
                                    elif 'TRUE' in line.upper() and final_verdict is None and 'FALSE' not in line.upper():
                                        final_verdict = "TRUE"
                                    elif 'UNVERIFIED' in line.upper() and final_verdict is None:
                                        final_verdict = "UNVERIFIED"
                        
                        # Handle tool calls
                        elif hasattr(part, 'function_call') and part.function_call:
                            logger.warning("🔧 Tool called: %s", part.function_call.name)
            
            execution_time = (time.time() - start_time) * 1000
            logger.warning("⏱️  Execution time: %.0f ms", execution_time)
            
            # Save to database (FIXED: use strings, not objects)
            try:
                memory.add_interaction(
                    session_id=session_id,
                    query=user_input[:200],  # Limit query length
                    processed_input=processed_input[:500],  # Limit processed input
                    verdict=final_verdict or "UNKNOWN"
                )
                logger.warning("💾 Interaction saved to memory")
            except Exception as e:
                logger.warning("⚠️  Failed to save interaction: %s", str(e)[:50])
            
            print("\n" + "─" * 60)
            
            # Display memory stats
            try:
                stats = memory.get_all_stats()
                print(f"\n📊 System Stats:")
                print(f"   Verified claims: {stats['total_verified_claims']}")
                print(f"   Average confidence: {stats['average_confidence']:.1%}")
                print(f"   Sessions: {stats['total_sessions']}")
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