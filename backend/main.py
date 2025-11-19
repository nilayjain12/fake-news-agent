# backend/main.py
"""Main entry point for CLI-run fact-check agent."""
import sys
import asyncio
from google.adk.runners import InMemoryRunner
from agents.fact_check_agent_adk import FactCheckSequentialAgent
from google.genai import types
from config import get_logger

logger = get_logger(__name__)


async def main_async():
    """Async main function to handle the agent interaction properly."""
    agent = FactCheckSequentialAgent()
    
    runner = InMemoryRunner(
        app_name="agents",
        agent=agent
    )
    
    await runner.session_service.create_session(
        app_name="agents",
        user_id="cli-user",
        session_id="cli-session"
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
        
        try:
            message = types.Content(
                role="user",
                parts=[types.Part(text=processed_input)]
            )
            
            result_gen = runner.run(
                user_id="cli-user",
                session_id="cli-session",
                new_message=message
            )
            
            print("─" * 60)
            final_content = None
            
            for event in result_gen:
                if hasattr(event, 'content') and event.content:
                    final_content = event.content
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text, end='', flush=True)
                        elif hasattr(part, 'function_call') and part.function_call:
                            logger.warning("🔧 Tool called: %s", part.function_call.name)
            
            print("\n" + "─" * 60)
            
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