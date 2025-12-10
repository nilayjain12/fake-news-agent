# backend/utils/model_router.py

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL_POOL, get_logger
import os

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

logger = get_logger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_with_fallback(prompt: str):
    """
    Try Gemini models one by one until a response succeeds.
    Automatically handles 429 / quota exhaustion.
    """

    last_error = None

    for model_name in GEMINI_MODEL_POOL:
        try:
            logger.warning(f"🧠 Trying Gemini Model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            text = response.text if hasattr(response, "text") else str(response)

            logger.warning(f"✅ Model Success: {model_name}")
            return text, model_name

        except Exception as e:
            error_str = str(e).lower()
            logger.warning(f"⚠️ Model Failed: {model_name} → {str(e)[:120]}")

            # Detect quota / rate-limit errors only
            if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                last_error = e
                continue
            else:
                # Non-quota error → do not rotate
                raise e

    # If all models fail
    raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")
