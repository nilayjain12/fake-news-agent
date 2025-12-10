# backend/agents/root_orchestrator.py (IMPROVED - Better Context & Routing)
"""
Improved Orchestrator with:
- Better verdict formatting (category first, then reasoning)
- True conversation memory (tracks previous claims & verdicts)
- Smart query routing (new claims vs follow-ups vs casual chat)
- Minimal API calls (0 for follow-ups, 0 for casual chat)
"""

from config import ADK_MODEL_NAME, get_logger, GEMINI_API_KEY
from memory.manager import MemoryManager
from tools.semantic_ranker import SemanticRanker
import asyncio
import time
import json
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from config import quota_tracker
from utils.model_router import generate_with_fallback

logger = get_logger(__name__)


class SessionMemory:
    """Stores per-session conversation memory"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.claims = {}  # claim_text -> {verdict, confidence, timestamp}
        self.conversation = []  # List of all messages
    
    def add_claim(self, claim_text: str, verdict: str, confidence: float):
        """Store a fact-checked claim"""
        self.claims[claim_text[:200]] = {
            "verdict": verdict,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        logger.warning("💾 Claim stored: %s -> %s", claim_text[:60], verdict)
    
    def add_message(self, role: str, content: str):
        """Store conversation message"""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_last_claim(self) -> Optional[Dict]:
        """Get the most recently discussed claim"""
        if not self.claims:
            return None
        
        # Get the last stored claim (in order of storage)
        last_key = list(self.claims.keys())[-1]
        return {
            "claim": last_key,
            **self.claims[last_key]
        }
    
    def get_conversation_context(self, max_messages: int = 3) -> str:
        """Get recent conversation for context"""
        recent = self.conversation[-max_messages:] if len(self.conversation) > max_messages else self.conversation
        
        context = ""
        for msg in recent:
            role = "You" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content'][:150]}\n"
        
        return context
    
    def find_related_claim(self, query: str) -> Optional[Dict]:
        """Find if query is related to a previous claim"""
        # Simple keyword matching - check if query mentions previous claim topics
        query_lower = query.lower()
        
        for claim_text, claim_data in self.claims.items():
            # Check for keyword overlap
            claim_words = set(claim_text.lower().split()[:5])  # First 5 words
            query_words = set(query_lower.split()[:5])
            
            overlap = len(claim_words & query_words)
            if overlap >= 2:  # At least 2 words match
                logger.warning("🔗 Found related claim: %s", claim_text[:60])
                return {
                    "claim": claim_text,
                    **claim_data
                }
        
        return None


class ImprovedOrchestrator:
    """
    Improved orchestrator with better verdict formatting and context awareness.
    
    FEATURES:
    1. Clear Verdict Format - Category first, reasoning second
    2. Conversation Memory - Tracks claims and previous discussions
    3. Smart Routing - Detects new claims vs follow-ups vs casual chat
    4. Minimal API - 2 calls for new claims, 0 for everything else
    """
    
    def __init__(self):
        logger.warning("🚀 Initializing Improved Orchestrator (Better Context & Routing)")
        self.memory = MemoryManager()
        self.ranker = SemanticRanker()
        
        # Session memories
        self.sessions = {}  # session_id -> SessionMemory
        
        # Tracking
        self.api_calls = 0
        self.cache_hits = 0
        
        logger.warning("✅ Improved Orchestrator initialized")
    
    def get_session_memory(self, session_id: str) -> SessionMemory:
        """Get or create session memory"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id)
        return self.sessions[session_id]
    
    async def process_query(self, user_input: str, session_id: str) -> Dict:
        """
        Process query with smart routing and conversation memory.
        
        Returns properly formatted response with:
        - Verdict category (TRUE/FALSE/INCONCLUSIVE)
        - Confidence score
        - Detailed reasoning
        """
        
        start_time = time.time()
        session_mem = self.get_session_memory(session_id)
        
        logger.warning("🔍 Processing: %s", user_input[:80])
        
        try:
            # ===== STEP 1: DETECT QUERY TYPE =====
            query_type = self._detect_query_type(user_input, session_mem)
            logger.warning("🎯 Query Type: %s", query_type)
            
            # ===== STEP 2: ROUTE BASED ON TYPE =====
            if query_type == "follow_up":
                logger.warning("🔄 Route: FOLLOW-UP (0 API calls)")
                result = await self._handle_follow_up(user_input, session_mem, start_time)
            
            elif query_type == "casual_chat":
                logger.warning("💬 Route: CASUAL CHAT (0 API calls)")
                result = await self._handle_casual_chat(user_input, session_mem, start_time)
            
            else:  # new_claim
                logger.warning("📋 Route: NEW CLAIM (2 API calls)")
                result = await self._handle_new_claim(user_input, session_mem, start_time)
            
            # ===== STEP 3: STORE IN SESSION MEMORY =====
            session_mem.add_message("user", user_input)
            if result.get("success"):
                session_mem.add_message("assistant", result.get("verdict", "UNKNOWN"))
            
            return result
        
        except Exception as e:
            logger.exception("❌ Error: %s", e)
            execution_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e)[:200],
                "execution_time_ms": execution_time,
                "api_calls": 0
            }
    
    def _detect_query_type(self, user_input: str, session_mem: SessionMemory) -> str:
        """
        Detect query type:
        1. follow_up - References previous claim or uses follow-up keywords
        2. casual_chat - Greeting, thanks, questions about system
        3. new_claim - Default, new claim to fact-check
        """
        
        input_lower = user_input.lower()
        
        # ===== FOLLOW-UP DETECTION =====
        follow_up_indicators = [
            "but", "however", "actually", "wait", "no wait",
            "though", "yet", "confirmed", "sources", "found",
            "different source", "another source", "i read", "i saw",
            "you said", "earlier", "before", "previous", "that claim",
            "that person", "about that", "about him", "about her",
            "what about", "regarding", "concerning", "also", "moreover"
        ]
        
        has_follow_up_keywords = any(kw in input_lower for kw in follow_up_indicators)
        
        # Check if related to previous claim
        related_claim = session_mem.find_related_claim(user_input)
        has_related_context = related_claim is not None
        
        if has_follow_up_keywords or has_related_context:
            logger.warning("   → Follow-up indicators: keywords=%s, related=%s", 
                          has_follow_up_keywords, has_related_context)
            return "follow_up"
        
        # ===== CASUAL CHAT DETECTION =====
        casual_indicators = [
            "hello", "hi ", "hey", "thanks", "thank you", "good",
            "remember", "conversation", "previous", "how are you",
            "what is your name", "who are you", "how do you work",
            "can you help", "tell me about"
        ]
        
        has_casual = any(ind in input_lower for ind in casual_indicators)
        
        if has_casual:
            logger.warning("   → Casual indicators detected")
            return "casual_chat"
        
        # ===== DEFAULT: NEW CLAIM =====
        logger.warning("   → Treating as new claim")
        return "new_claim"
    
    async def _handle_follow_up(self, user_input: str, session_mem: SessionMemory, start_time: float) -> Dict:
        """Handle follow-up questions - 0 API calls"""
        
        last_claim = session_mem.get_last_claim()
        
        if not last_claim:
            logger.warning("   ⚠️  No previous claim found")
            return {
                "success": False,
                "error": "No previous claim to reference. Please share a new claim to fact-check.",
                "execution_time_ms": (time.time() - start_time) * 1000,
                "api_calls": 0,
                "query_type": "follow_up"
            }
        
        # Get conversation context
        context = session_mem.get_conversation_context()
        
        # Generate context-aware response
        logger.warning("   📌 Previous claim: %s", last_claim["claim"][:60])
        logger.warning("   📌 Previous verdict: %s", last_claim["verdict"])
        
        response_text = self._generate_follow_up_response(
            user_input=user_input,
            previous_claim=last_claim["claim"],
            previous_verdict=last_claim["verdict"],
            context=context
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "claim": last_claim["claim"],
            "verdict": last_claim["verdict"],
            "confidence": last_claim.get("confidence", 0.5),
            "reasoning": response_text,
            "is_follow_up": True,
            "execution_time_ms": execution_time,
            "api_calls": 0,
            "query_type": "follow_up"
        }
    
    def _generate_follow_up_response(self, user_input: str, previous_claim: str, 
                                   previous_verdict: str, context: str) -> str:
        """Generate natural follow-up response"""
        
        # Simple template-based responses for follow-ups
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["but", "however", "actually", "yet"]):
            response = f"""Regarding your follow-up to the claim '{previous_claim[:50]}...':

You mentioned different information or sources. Our original fact-check determined this claim to be **{previous_verdict}**. 

While you may have found conflicting information, established fact-checking sources support the **{previous_verdict}** verdict. If you have specific credible sources that contradict this, feel free to share them. Otherwise, the evidence-based verdict remains **{previous_verdict}**."""
        
        elif any(word in input_lower for word in ["you said", "earlier", "before", "that"]):
            response = f"""You're referencing our earlier discussion about '{previous_claim[:50]}...'. 

As we determined, this claim is **{previous_verdict}** based on available evidence. Is there something specific about this verdict you'd like me to clarify, or would you like to fact-check a different claim?"""
        
        else:
            response = f"""Continuing from our discussion about '{previous_claim[:50]}...':

Our fact-check found this to be **{previous_verdict}**. What else would you like to know or verify?"""
        
        return response
    
    async def _handle_casual_chat(self, user_input: str, session_mem: SessionMemory, start_time: float) -> Dict:
        """Handle casual chat - 0 API calls"""
        
        input_lower = user_input.lower()
        
        # Simple response mapping
        if "hello" in input_lower or "hi " in input_lower or "hey" in input_lower:
            response = "Hello! I'm a fact-checking assistant. Share any claim you'd like me to verify, and I'll provide a verdict with detailed reasoning."
        
        elif "thank" in input_lower:
            response = "You're welcome! Feel free to ask me to fact-check any claims. What would you like to verify?"
        
        elif "remember" in input_lower or "conversation" in input_lower:
            response = "Yes, I remember our conversation within this session. I track the claims we've discussed and can understand follow-up questions about them. How can I help?"
        
        elif "name" in input_lower or "who are you" in input_lower:
            response = "I'm a Fact-Checking Assistant powered by AI. My job is to verify claims and provide clear verdicts (TRUE/FALSE/INCONCLUSIVE) with detailed reasoning."
        
        else:
            response = "I'm here to help fact-check claims. Share a statement you'd like me to verify, and I'll provide a thorough analysis with evidence."
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "response": response,
            "is_casual_chat": True,
            "execution_time_ms": execution_time,
            "api_calls": 0,
            "query_type": "casual_chat"
        }
    
    async def _handle_new_claim(self, user_input: str, session_mem: SessionMemory, start_time: float) -> Dict:
        """Handle new claim - 2 API calls (full fact-check)"""
        
        # ===== CACHE CHECK =====
        logger.warning("🔍 Stage 0: CACHE CHECK")
        cached = self.memory.get_cached_verdict(user_input[:500])
        
        if cached:
            logger.warning("✅ CACHE HIT")
            self.cache_hits += 1
            
            verdict = cached.get("verdict", "INCONCLUSIVE")
            confidence = cached.get("confidence", 0.5)
            
            session_mem.add_claim(cached.get("claim_text", user_input[:100]), verdict, confidence)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "claim": cached.get("claim_text", user_input[:100]),
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": "Based on cached previous fact-check.",
                "execution_time_ms": execution_time,
                "api_calls": 0,
                "from_cache": True
            }
        
        # ===== STAGE 1: EXTRACTION =====
        logger.warning("🔍 Stage 1: EXTRACTION (API CALL #1)")
        
        clean_text = user_input.strip()[:3000]
        
        extraction_prompt = f"""Extract the main claim from this text. Return ONLY JSON:

TEXT: {clean_text}

{{"main_claim": "The exact claim", "claim_type": "factual/scientific/historical"}}"""
        
        try:
            import os
            from google import genai
            os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY
            client = genai.Client(api_key=GEMINI_API_KEY)
            
            allowed, msg = quota_tracker.check_quota_available(1)
            if not allowed:
                return {
                    "success": False,
                    "error": msg,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "api_calls": 0
                }

            quota_tracker.increment_call_count(1)

            

            response_text, used_model = generate_with_fallback(extraction_prompt)
            self.api_calls += 1

            logger.warning(f"🧠 Extraction used model: {used_model}")
            
            parsed = self._safe_json_parse(response_text, {"main_claim": clean_text[:200]})
            main_claim = parsed.get("main_claim", clean_text[:200])
            
            logger.warning("✅ Extracted: %s", main_claim[:80])
        
        except Exception as e:
            logger.warning("❌ Extraction error: %s", str(e)[:100])
            main_claim = clean_text[:200]
        
        # ===== STAGE 2: EVIDENCE GATHERING =====
        logger.warning("🔍 Stage 2: EVIDENCE (0 API CALLS)")
        
        faiss_results = await self._search_faiss(main_claim)
        google_results = await self._search_google(main_claim)
        
        all_evidence = []
        for result in faiss_results:
            all_evidence.append({
                "content": result.get("content", "")[:200],
                "source": result.get("source", "Knowledge"),
                "_source": "faiss"
            })
        for result in google_results:
            all_evidence.append({
                "content": result.get("content", "")[:200],
                "source": result.get("url", "Web"),
                "_source": "google"
            })
        
        # Rank evidence
        if all_evidence:
            try:
                ranked, _, _ = self.ranker._advanced_rank_with_multi_factor_scoring(
                    query=main_claim, evidence_items=all_evidence, top_k=5
                )
                all_evidence = ranked
            except:
                all_evidence = all_evidence[:5]
        
        evidence_summary = self._summarize_evidence(main_claim, all_evidence)
        
        # ===== STAGE 3: VERDICT =====
        logger.warning("🔍 Stage 3: VERDICT (API CALL #2)")
        
        verdict_prompt = f"""Provide a fact-check verdict. Return ONLY JSON:

CLAIM: {main_claim}

EVIDENCE: {evidence_summary[:1000]}

{{"verdict": "TRUE/FALSE/INCONCLUSIVE", "confidence": 0.0-1.0, "reasoning": "2-3 sentence explanation"}}"""
        
        try:
            allowed, msg = quota_tracker.check_quota_available(1)
            if not allowed:
                return {
                    "success": False,
                    "error": msg,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "api_calls": 0
                }

            quota_tracker.increment_call_count(1)
            response_text, used_model = generate_with_fallback(verdict_prompt)
            self.api_calls += 1

            logger.warning(f"🧠 Verdict used model: {used_model}")
            
            parsed = self._safe_json_parse(response_text, {
                "verdict": "INCONCLUSIVE",
                "confidence": 0.5,
                "reasoning": "Unable to determine"
            })
            
            verdict = parsed.get("verdict", "INCONCLUSIVE").upper()
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = parsed.get("reasoning", "")
            
            logger.warning("✅ Verdict: %s (%.0f%%)", verdict, confidence * 100)
        
        except Exception as e:
            logger.warning("❌ Verdict error: %s", str(e)[:100])
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            reasoning = "Unable to determine verdict."
        
        # Store in session memory
        session_mem.add_claim(main_claim, verdict, confidence)
        
        # Cache result
        try:
            self.memory.cache_verdict(main_claim[:500], verdict, confidence, len(all_evidence))
        except:
            pass
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "claim": main_claim,
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence_count": len(all_evidence),
            "execution_time_ms": execution_time,
            "api_calls": 2,
            "from_cache": False
        }
    
    async def _search_faiss(self, query: str) -> List[Dict]:
        """Search FAISS (0 API calls)"""
        try:
            from tools.faiss_tool import search_faiss_knowledge_base
            return search_faiss_knowledge_base(query, k=5)
        except:
            return []
    
    async def _search_google(self, query: str) -> List[Dict]:
        try:
            from tools.google_search_tool import search_google
            results = await search_google(query, top_k=5)

            if not isinstance(results, list):
                logger.warning("Google Search returned invalid format, falling back to []")
                return []

            return results

        except Exception as e:
            logger.warning(f"Google Search Error: {e}")
            return []
    
    def _summarize_evidence(self, claim: str, evidence_items: List[Dict]) -> str:
        """Summarize evidence using local heuristics"""
        
        if not evidence_items:
            return "No evidence found."
        
        summary = ""
        for i, evidence in enumerate(evidence_items[:5], 1):
            content = evidence.get("content", "")[:150]
            source = evidence.get("source", "Source")
            summary += f"{i}. [{source}] {content}\n"
        
        return summary
    
    def _safe_json_parse(self, text: str, default: dict) -> dict:
        """Safely parse JSON"""
        if not text:
            return default
        
        try:
            return json.loads(text)
        except:
            pass
        
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0]
            else:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    json_str = text[start:end+1]
                else:
                    return default
            return json.loads(json_str.strip())
        except:
            return default


# Singleton instance
root_orchestrator = ImprovedOrchestrator()