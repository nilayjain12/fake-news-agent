# backend/agents/image_processing_agent.py
"""Image processing agent for extracting and verifying text from images."""
import base64
import json
from pathlib import Path
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import os

logger = get_logger(__name__)

class ImageProcessingAgent:
    """Processes images to extract text and identify verifiable claims."""
    
    def __init__(self):
        os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = ADK_MODEL_NAME
        logger.warning("🖼️  ImageProcessingAgent initialized")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.warning("❌ Error encoding image: %s", str(e)[:50])
            raise
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract all text content from an image using Gemini Vision."""
        logger.warning("📸 Processing image: %s", Path(image_path).name)
        
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Determine image type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Call Gemini Vision API
            prompt = """Extract ALL text visible in this image, including:
- Headlines
- Body text
- Captions
- Labels
- Any printed or handwritten text

Format the output as a coherent paragraph or list of statements."""
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            extracted_text = response.text if hasattr(response, 'text') else str(response)
            logger.warning("✅ Extracted %d characters from image", len(extracted_text))
            return extracted_text
            
        except Exception as e:
            logger.warning("❌ Error extracting text from image: %s", str(e)[:100])
            raise
    
    def identify_claims_from_image(self, image_path: str) -> list:
        """Extract text from image and identify verifiable claims."""
        try:
            # Step 1: Extract text
            text_content = self.extract_text_from_image(image_path)
            
            if not text_content:
                logger.warning("⚠️  No text extracted from image")
                return []
            
            # Step 2: Identify claims using Gemini
            prompt = f"""Analyze this text and identify specific factual claims that can be verified:

TEXT FROM IMAGE:
{text_content}

TASK: List 3-5 specific factual claims (statements of fact, not opinions).
Format as JSON array:
{{"claims": ["claim 1", "claim 2", ...]}}

Include ONLY verifiable claims, exclude opinions."""
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_str = response_text
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                
                parsed = json.loads(json_str)
                claims = parsed.get("claims", [])
                logger.warning("✅ Identified %d claims from image", len(claims))
                return claims
            except json.JSONDecodeError as e:
                logger.warning("⚠️  Could not parse JSON response: %s", str(e)[:50])
                # Fallback: return text as single claim
                return [text_content[:500]] if text_content else []
                
        except Exception as e:
            logger.warning("❌ Error identifying claims: %s", str(e)[:100])
            raise
    
    def run(self, image_path: str) -> dict:
        """Process image and return extracted text and claims."""
        try:
            logger.warning("🖼️  Processing image for verification")
            
            # Extract claims
            claims = self.identify_claims_from_image(image_path)
            
            # Also get raw text
            text_content = self.extract_text_from_image(image_path)
            
            return {
                "success": True,
                "image_path": str(image_path),
                "extracted_text": text_content,
                "claims": claims,
                "claim_count": len(claims)
            }
        except Exception as e:
            logger.warning("❌ Image processing failed: %s", str(e)[:100])
            return {
                "success": False,
                "error": str(e),
                "claims": []
            }