# backend/agents/image_processing_agent.py - UPDATED WITH CLUSTERING
"""
UPDATED: Image processing agent with claim clustering to reduce redundant verifications.
Now clusters related claims before verification to save API calls and reduce resource exhaustion.
"""
import base64
import json
from pathlib import Path
from google import genai
from config import GEMINI_API_KEY, ADK_MODEL_NAME, get_logger
import os

logger = get_logger(__name__)

class ImageProcessingAgent:
    """
    UPDATED: Processes images with intelligent claim clustering.
    
    Workflow:
    1. Extract text from image (OCR)
    2. Identify verifiable claims
    3. **NEW:** Cluster similar claims together
    4. Return clustered claims for verification
    """
    
    def __init__(self):
        os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = ADK_MODEL_NAME
        logger.warning("🖼️  ImageProcessingAgent initialized (WITH CLUSTERING)")
    
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
            base64_image = self.encode_image_to_base64(image_path)
            
            ext = Path(image_path).suffix.lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
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
        """
        UPDATED: Extract text and identify claims, then cluster them.
        This reduces redundant verifications significantly.
        """
        try:
            # Step 1: Extract text
            text_content = self.extract_text_from_image(image_path)
            
            if not text_content:
                logger.warning("⚠️  No text extracted from image")
                return []
            
            # Step 2: Identify ALL claims (may have duplicates/similar ones)
            raw_claims = self._extract_raw_claims(text_content)
            
            if not raw_claims:
                logger.warning("⚠️  No claims identified from text")
                return []
            
            logger.warning("✅ Identified %d raw claims from image", len(raw_claims))
            
            # Step 3: **NEW** - Cluster similar claims
            if len(raw_claims) > 1:
                try:
                    from agents.claim_clustering_agent import cluster_claims
                    
                    logger.warning("🔗 Clustering %d claims to reduce redundancy...", len(raw_claims))
                    clustered_claims, metadata = cluster_claims(raw_claims)
                    
                    logger.warning("✅ Clustered %d → %d claims (%.0f%% reduction)",
                                  metadata["original_count"],
                                  metadata["clustered_count"],
                                  metadata["reduction_percentage"])
                    
                    # Log clustering details
                    for i, cluster_info in enumerate(metadata["clusters"], 1):
                        logger.warning("   Cluster %d: %d claims → '%s'",
                                      i, cluster_info["size"], cluster_info["summary"][:80])
                    
                    return clustered_claims
                    
                except Exception as e:
                    logger.warning("⚠️  Clustering failed: %s", str(e)[:100])
                    logger.warning("⚠️  Falling back to raw claims")
                    return raw_claims
            else:
                # Only 1 claim - no clustering needed
                return raw_claims
                
        except Exception as e:
            logger.warning("❌ Error identifying claims: %s", str(e)[:100])
            raise
    
    def _extract_raw_claims(self, text_content: str) -> list:
        """Extract raw claims from text (before clustering)."""
        prompt = f"""Analyze this text and identify ALL specific factual claims that can be verified:

TEXT FROM IMAGE:
{text_content}

TASK: List ALL verifiable claims (statements of fact, not opinions).
Format as JSON array:
{{"claims": ["claim 1", "claim 2", ...]}}

Include ONLY verifiable factual claims, exclude opinions.
Be comprehensive - extract ALL claims, even if they seem related."""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Parse JSON response
            try:
                json_str = response_text
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]
                
                parsed = json.loads(json_str)
                claims = parsed.get("claims", [])
                logger.warning("✅ Extracted %d raw claims", len(claims))
                return claims
            except json.JSONDecodeError as e:
                logger.warning("⚠️  Could not parse JSON response: %s", str(e)[:50])
                return [text_content[:500]] if text_content else []
                
        except Exception as e:
            logger.warning("❌ Error extracting raw claims: %s", str(e)[:100])
            return []
    
    def run(self, image_path: str) -> dict:
        """
        UPDATED: Process image and return clustered claims.
        
        Returns:
            {
                "success": bool,
                "image_path": str,
                "extracted_text": str,
                "claims": list,  # Now clustered/summarized
                "claim_count": int,
                "clustering_metadata": dict  # NEW: clustering info
            }
        """
        try:
            logger.warning("🖼️  Processing image for verification")
            
            # Extract and cluster claims
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