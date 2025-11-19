"""Main entry point for CLI-run fact-check agent."""
import sys
import asyncio
import logging
from google.adk.runners import InMemoryRunner
from agents.fact_check_agent_adk import FactCheckSequentialAgent
from google.genai import types
from config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Creating FactCheckSequentialAgent")
    agent = FactCheckSequentialAgent()

    runner = InMemoryRunner(
        app_name="agents",
        agent=agent
    )

    logger.info("Initializing session...")
    print("Initializing session...")
    asyncio.run(
        runner.session_service.create_session(
            app_name="agents",
            user_id="cli-user",
            session_id="cli-session"
        )
    )

    print("\n\n=== Fact-Check Agent ===")
    while True:
        try:
            query = input("\nEnter URL or text (or 'exit'): ").strip()
        except EOFError:
            logger.info("EOF received, exiting loop")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        # The runner.run method is synchronous and handles the loop internally,
        # but it needs the session we created above to exist first.
        result_gen = runner.run(
            user_id="cli-user",
            session_id="cli-session",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )
        )

        # Consume generator
        final_result = None
        logger.debug("Consuming result generator from runner")
        for step in result_gen:
            print(step)
            logger.debug("Got runner step: %s", repr(step))
            final_result = step

        logger.info("--- Agent Output ---")
        print("\n--- Agent Output ---")
        # if isinstance(final_result, dict) and "output" in final_result:
        #     logger.info("Final result has 'output' key")
        #     print(final_result["output"])
        # else:
        #     logger.info("Final result: %s", repr(final_result))
        print(final_result)
        print("--------------------\n")

if __name__ == "__main__":
    main()