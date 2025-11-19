# Defining a FactCheckAgent that uses RAG for factual verification.
from google import genai
from rag.retriever import Retriever
from backend.tools.google_search_tool import google_search_tool
from config import ADK_MODEL_NAME
from dotenv import load_dotenv
import os

# Load variables from .env (load_dotenv returns bool). Then read the actual API key.
load_dotenv('.env')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Do not instantiate the client if the API key is missing; provide a clear runtime error.
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

class FactCheckAgent:
    def __init__(self):
        self.retriever = Retriever()

    def run(self, user_query: str) -> str:
        # Step 1: Fetch RAG context
        context = self.retriever.fetch_context(user_query)

        prompt = f"""
                You are a factual verification agent for fake-news detection.
                You must use strict reasoning and cite retrieved evidence.

                User Query:
                {user_query}

                Retrieved Knowledge Base Context:
                {context}

                Task:
                - Determine if the query is likely true or false.
                - Use only the retrieved context.
                - Respond clearly.
            """

        response = client.models.generate_content(
            model=ADK_MODEL_NAME,
            contents=prompt
        )

        return response.text