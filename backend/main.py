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
    logger.info("Creating FactCheckSequentialAgent")
    agent = FactCheckSequentialAgent()
    
    # Use the agent with InMemoryRunner
    runner = InMemoryRunner(
        app_name="agents",
        agent=agent
    )
    
    logger.info("Initializing session...")
    print("Initializing session...")
    
    # Create session
    await runner.session_service.create_session(
        app_name="agents",
        user_id="cli-user",
        session_id="cli-session"
    )
    
    print("\n\n=== Fact-Check Agent ===")
    print("Enter a URL or article text to fact-check.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("\nEnter URL or text (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("EOF/Interrupt received, exiting")
            print("\nExiting...")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        
        # Preprocess the input (handle URL extraction)
        processed_input = agent.preprocess_input(user_input)
        
        logger.info("Processing input with length=%d", len(processed_input))
        print("\n--- Processing your request ---")
        print("Extracting claims, verifying with sources, and generating report...\n")
        
        try:
            # Create the message content with preprocessed input
            message = types.Content(
                role="user",
                parts=[types.Part(text=processed_input)]
            )
            
            # Run the agent
            result_gen = runner.run(
                user_id="cli-user",
                session_id="cli-session",
                new_message=message
            )
            
            # Collect and display events
            print("--- Agent Processing ---\n")
            final_content = None
            event_count = 0
            
            for event in result_gen:
                event_count += 1
                logger.debug("Event %d: %s", event_count, type(event).__name__)
                
                # Display content from events
                if hasattr(event, 'content') and event.content:
                    final_content = event.content
                    # Display parts as they come
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text, end='', flush=True)
                
                # Log function calls for debugging
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            logger.info("Tool/Sub-agent called: %s", part.function_call.name)
            
            print("\n\n--- Complete ---")
            
            if event_count == 0:
                print("⚠️  No events generated. Please check your configuration.")
                logger.warning("No events received from agent")
                
        except Exception as e:
            logger.exception("Error during agent execution: %s", e)
            print(f"\n❌ Error: {e}")
            print("Please try again with a different input.")


def main():
    """Entry point that runs the async main function."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.exception("Fatal error in main: %s", e)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()