# ui/app.py (IMPROVED - Better Formatting & Context)
"""
Improved UI with:
- Clear verdict display (category first, then reasoning)
- Conversation memory tracking
- Smart query routing
- Better response formatting
"""

import gradio as gr
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import os

BACKEND_PATH = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))
os.chdir(str(BACKEND_PATH))

from agents.root_orchestrator import root_orchestrator
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)

memory = None
session_id = None


def initialize():
    """Initialize on startup"""
    global memory, session_id
    
    if memory is None:
        try:
            memory = MemoryManager()
            session_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            memory.create_session(session_id, user_id="web-user")
            logger.warning(f"✅ Improved Orchestrator initialized: {session_id}")
        except Exception as e:
            logger.exception(f"❌ Init error: {e}")
            raise
    
    return memory, session_id


def format_response(result: dict) -> str:
    """Format result with clear verdict display"""
    
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        return f"❌ **Error**\n\n{error}"
    
    # ===== CASUAL CHAT =====
    if result.get("is_casual_chat"):
        response = result.get("response", "")
        execution = result.get("execution_time_ms", 0)
        return f"{response}\n\n---\n⏱️ Response: {execution:.0f}ms"
    
    # ===== FOLLOW-UP =====
    if result.get("is_follow_up"):
        claim = result.get("claim", "Previous claim")
        reasoning = result.get("reasoning", "")
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0)
        execution = result.get("execution_time_ms", 0)
        
        return f"""## Regarding: {claim[:80]}...

**Previous Verdict:** {verdict} ({confidence:.0%})

{reasoning}

---
⏱️ Response Time: {execution:.0f}ms
📊 API Calls: 0 (Context-aware response)"""
    
    # ===== NEW CLAIM FACT-CHECK =====
    claim = result.get("claim", "Unknown claim")
    verdict = result.get("verdict", "INCONCLUSIVE").upper()
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "")
    evidence_count = result.get("evidence_count", 0)
    execution = result.get("execution_time_ms", 0)
    from_cache = result.get("from_cache", False)
    api_calls = result.get("api_calls", 0)
    
    # Format verdict with icon and category name
    verdict_icons = {
        "TRUE": "✅",
        "FALSE": "❌",
        "INCONCLUSIVE": "❓"
    }
    
    verdict_names = {
        "TRUE": "LIKELY TRUE",
        "FALSE": "LIKELY FALSE",
        "INCONCLUSIVE": "CANNOT BE DETERMINED"
    }
    
    icon = verdict_icons.get(verdict, "❓")
    name = verdict_names.get(verdict, "UNKNOWN")
    
    response = f"""## Fact-Check Result

### Verdict Category
**{icon} {name}**
Confidence: {confidence:.0%}

### Claim
> {claim}

### Reasoning
{reasoning}

---
⏱️ Execution Time: {execution:.0f}ms
📊 API Calls: {api_calls}/20
📚 Evidence Reviewed: {evidence_count} sources
{"💾 From Cache (Instant)" if from_cache else "🔍 Fresh Fact-Check"}"""
    
    return response


async def process_fact_check(user_input: str) -> str:
    """Process query using improved orchestrator"""
    
    memory_inst, session = initialize()
    
    if not user_input.strip():
        return "Please enter a claim or question."
    
    try:
        logger.warning("🔍 Processing: %s", user_input[:80])
        
        # Run improved orchestrator
        result = await root_orchestrator.process_query(
            user_input=user_input,
            session_id=session
        )
        
        # Format and return response
        return format_response(result)
    
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        return f"❌ **Error**\n\n{str(e)[:200]}"


def chat_interface(message: str, history: list):
    """Chat interface with conversation tracking"""
    
    if not message.strip():
        return history, ""
    
    # Add user message to history
    history.append({"role": "user", "content": message})
    
    try:
        # Process message
        result = asyncio.run(process_fact_check(message))
        
        # Add bot response to history
        history.append({"role": "assistant", "content": result})
        
    except Exception as e:
        logger.exception(f"❌ Chat error: {e}")
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)[:200]}"})
    
    return history, ""


def get_stats() -> str:
    """Get system statistics"""
    try:
        memory_inst, session = initialize()
        stats = memory_inst.get_all_stats()
        
        orch_stats = root_orchestrator.get_stats() if hasattr(root_orchestrator, 'get_stats') else {
            "api_calls": root_orchestrator.api_calls,
            "cache_hits": root_orchestrator.cache_hits
        }
        
        stats_text = f"""### 📊 System Statistics

**Fact-Checks Performed:** {stats['total_verified_claims']}
**Average Confidence:** {stats['average_confidence']:.1%}
**Sessions:** {stats['total_sessions']}

**Verdict Distribution:**"""
        
        if stats['verdict_distribution']:
            for verdict, count in stats['verdict_distribution'].items():
                stats_text += f"\n- {verdict}: {count}"
        
        stats_text += f"""

**Current Session:**
- Cache Hits: {orch_stats['cache_hits']}
- API Calls Used: {orch_stats['api_calls']}/20
- Remaining: {20 - orch_stats['api_calls']}/20"""
        
        return stats_text
    except Exception as e:
        return f"Could not load stats: {str(e)}"


def create_interface():
    """Create Gradio interface"""
    
    initialize()
    
    with gr.Blocks(
        title="Fact-Check Agent",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="purple"),
        css="""
        .verdict-container {
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔍 Smart Fact-Check Agent
        
        **Context-Aware Verification with Conversation Memory**
        
        - ✅ **Clear Verdicts** - Category + Confidence + Reasoning
        - 💬 **Conversation Memory** - Understands follow-ups
        - 🎯 **Smart Routing** - New claims vs follow-ups vs chat
        - ⚡ **Efficient** - 2 API calls for facts, 0 for follow-ups
        """)
        
        with gr.Tab("💬 Fact-Check"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Ask & Verify")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        type="messages"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Question or Claim",
                            placeholder="E.g., 'Did Trump offer green cards to students?' or 'But I read it was true...'",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", size="lg", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
                        example_btn = gr.Button("📝 Example", scale=1)
                    
                    submit_btn.click(
                        chat_interface,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    msg.submit(
                        chat_interface,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    clear_btn.click(lambda: [], outputs=chatbot)
                    
                    def load_example():
                        import random
                        examples = [
                            "Trump offered international students green cards",
                            "But I have confirmed this from different sources",
                            "The moon is visible from Saturn with naked eye",
                            "Dharmendra, actor, died recently",
                            "But his family confirmed it",
                        ]
                        return random.choice(examples)
                    
                    example_btn.click(load_example, outputs=msg)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Stats")
                    stats_output = gr.Markdown()
                    refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    refresh_btn.click(get_stats, outputs=stats_output)
                    demo.load(get_stats, outputs=stats_output)
                    
                    gr.Markdown("""
                    ### ℹ️ How It Works
                    
                    **New Claims:**
                    - Full fact-check
                    - Search evidence
                    - 2 API calls
                    
                    **Follow-ups:**
                    - Understands context
                    - Links to previous
                    - 0 API calls
                    
                    **Chat:**
                    - Natural response
                    - Helpful guidance
                    - 0 API calls
                    """)
        
        with gr.Tab("ℹ️ Guide"):
            gr.Markdown("""
            ## How This Works
            
            ### Verdict Format
            
            Results show **three parts**:
            
            1. **Verdict Category**
               - ✅ LIKELY TRUE
               - ❌ LIKELY FALSE
               - ❓ CANNOT BE DETERMINED
            
            2. **Confidence Score**
               - 0-100% certainty based on evidence
            
            3. **Detailed Reasoning**
               - Why this verdict was reached
               - Evidence summary
               - Key findings
            
            ### Conversation Memory
            
            The system **remembers previous discussions**:
            
            ```
            You: "Did Trump offer green cards?"
            Bot: Verdict = FALSE
            
            You: "But I read it was true"
            Bot: Understands you're following up
                Returns context-aware response
                Links to original claim
            ```
            
            ### Smart Routing
            
            Different queries get different handling:
            
            | Query Type | Processing | API Calls |
            |-----------|-----------|----------|
            | New Claim | Full fact-check | 2 |
            | Follow-up | Context-aware | 0 |
            | Chat | Friendly reply | 0 |
            | Cached | Instant | 0 |
            
            ### Examples
            
            **Example 1: New Claim**
            ```
            Input: "Is water made of hydrogen?"
            Processing: Full fact-check (2 API calls)
            Time: 3-4 seconds
            Result: TRUE (95% confidence)
            ```
            
            **Example 2: Follow-up**
            ```
            Input: "But my chemistry teacher said..."
            Processing: Context-aware (0 API calls)
            Time: <1 second
            Result: Links to previous claim
            ```
            
            **Example 3: Casual Chat**
            ```
            Input: "Thanks for your help"
            Processing: Friendly (0 API calls)
            Time: <500ms
            Result: You're welcome response
            ```
            """)
    
    return demo


def main():
    """Main entry point"""
    try:
        print("🚀 Starting Improved Fact-Check Agent...")
        print("📍 http://0.0.0.0:8000")
        
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()