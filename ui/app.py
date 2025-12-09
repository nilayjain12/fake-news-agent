# ui/app.py
"""
Gradio UI for Fact-Checking Agent - Pure Google ADK
Real-time event streaming showing agent progress
"""
import gradio as gr
import sys
from pathlib import Path
from datetime import datetime
import os

# Setup paths
BACKEND_PATH = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))
os.chdir(str(BACKEND_PATH))

from agents.adk_pipeline import create_fact_check_pipeline
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)

# Global state
pipeline = None
memory = None
session_id = None


def initialize():
    """Initialize ADK pipeline and memory"""
    global pipeline, memory, session_id
    
    if pipeline is None:
        try:
            pipeline = create_fact_check_pipeline()
            memory = pipeline.memory
            session_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            memory.create_session(session_id, user_id="web-user")
            logger.warning(f"✅ ADK Pipeline initialized: {session_id}")
        except Exception as e:
            logger.exception(f"❌ Initialization error: {e}")
            raise
    
    return pipeline, memory


async def process_with_streaming(user_input: str, history: list):
    """
    Process fact-check with REAL-TIME ADK event streaming
    Shows each agent's progress as it happens
    """
    pipeline_instance, memory_instance = initialize()
    
    if not user_input.strip():
        yield history, ""
        return
    
    # Add user message
    history.append([user_input, "🔄 Starting ADK pipeline..."])
    yield history, ""
    
    thinking_steps = []
    
    try:
        # Preprocess input
        processed_input = pipeline_instance.preprocess_input(user_input)
        
        # Agent emojis for visualization
        agent_emojis = {
            "ingestion_agent": "📥",
            "claim_extraction_agent": "🎯",
            "evidence_retrieval_agent": "🔍",
            "verification_agent": "✅",
            "aggregator_agent": "⚖️",
            "report_agent": "📝"
        }
        
        # Stream ADK events
        async for event_data in pipeline_instance.stream_events(processed_input, session_id):
            event_type = event_data.get("type")
            
            if event_type == "agent_start":
                # New agent started
                agent_name = event_data.get("agent_name", "")
                emoji = agent_emojis.get(agent_name, "🔧")
                
                thinking_steps.append((
                    f"{emoji} {agent_name.replace('_', ' ').title()}",
                    "Starting..."
                ))
                
                # Update UI
                progress_html = format_thinking_live(thinking_steps)
                history[-1][1] = f"🧠 **Agent Progress:**\n\n{progress_html}"
                yield history, ""
            
            elif event_type == "agent_event":
                # Agent progress update
                text = event_data.get("text", "")
                
                if thinking_steps:
                    step_name, _ = thinking_steps[-1]
                    thinking_steps[-1] = (step_name, text[:200])
                
                progress_html = format_thinking_live(thinking_steps)
                history[-1][1] = f"🧠 **Agent Progress:**\n\n{progress_html}"
                yield history, ""
            
            elif event_type == "agent_complete":
                # Agent finished
                if thinking_steps:
                    step_name, details = thinking_steps[-1]
                    thinking_steps[-1] = (step_name, f"✅ {details}")
                
                progress_html = format_thinking_live(thinking_steps)
                history[-1][1] = f"🧠 **Agent Progress:**\n\n{progress_html}"
                yield history, ""
            
            elif event_type == "final_result":
                # Pipeline complete
                result = event_data.get("result", {})
                
                final_report = result.get("comprehensive_report", "")
                verdict = result.get("verdict", "UNKNOWN")
                confidence = result.get("confidence", 0.5)
                
                thinking_steps.append((
                    "✅ Verification Complete!",
                    f"Verdict: {verdict} ({confidence:.1%} confidence)"
                ))
                
                # Format final response
                thinking_html = format_thinking_live(thinking_steps)
                
                final_response = f"""<details open style="margin: 15px 0; padding: 15px; border: 2px solid #667eea; border-radius: 8px; background-color: #002B57;">
<summary style="cursor: pointer; font-weight: bold; color: #667eea; font-size: 16px;">🧠 Agent Thinking Process</summary>

{thinking_html}

</details>

---

{final_report}
"""
                
                history[-1][1] = final_response
                yield history, ""
                
                # Cache result
                if verdict and verdict != "ERROR":
                    try:
                        pipeline_instance.cache_result(
                            claim=user_input[:500],
                            verdict=verdict,
                            confidence=confidence,
                            session_id=session_id
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Caching failed: {str(e)}")
                
                # Log interaction
                try:
                    memory_instance.add_interaction(
                        session_id=session_id,
                        query=user_input[:200],
                        processed_input=processed_input[:500],
                        verdict=verdict
                    )
                except Exception as e:
                    logger.warning(f"⚠️  Logging failed: {str(e)}")
            
            elif event_type == "error":
                error_msg = event_data.get("error", "Unknown error")
                history[-1][1] = f"❌ Error: {error_msg}"
                yield history, ""
    
    except Exception as e:
        logger.exception(f"❌ Processing error: {e}")
        history[-1][1] = f"❌ Error: {str(e)[:200]}"
        yield history, ""


def format_thinking_live(steps: list) -> str:
    """Format thinking steps as HTML"""
    if not steps:
        return "<p>No steps yet...</p>"
    
    html = '<div style="font-family: monospace; line-height: 1.8;">'
    
    for i, (step_name, details) in enumerate(steps, 1):
        is_complete = "✅" in details or "✅" in step_name
        
        color = "#2ecc71" if is_complete else "#667eea"
        icon = "✅" if is_complete else "🔄"
        
        html += f"""
<div style="margin: 10px 0; padding: 12px; background-color: #002B57; border-left: 4px solid {color}; border-radius: 5px;">
    <strong style="color: {color};">{icon} {step_name}</strong><br>
    <span style="color: #a0aec0; margin-left: 20px;">{details}</span>
</div>
"""
    
    html += '</div>'
    return html


def get_stats() -> str:
    """Get system statistics"""
    try:
        _, memory_instance = initialize()
        
        stats = memory_instance.get_all_stats()
        
        stats_text = f"""### 📊 System Statistics

**Total Verified Claims:** {stats['total_verified_claims']}
**Average Confidence:** {stats['average_confidence']:.1%}
**Total Sessions:** {stats['total_sessions']}

**Verdict Distribution:**"""
        
        if stats['verdict_distribution']:
            for verdict, count in stats['verdict_distribution'].items():
                stats_text += f"\n- **{verdict}:** {count}"
        
        return stats_text
    except Exception as e:
        return f"Could not load statistics: {str(e)}"


def create_interface():
    """Create Gradio interface with ADK streaming"""
    
    initialize()
    
    with gr.Blocks(
        title="Fact-Check Agent (Pure ADK)",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="purple"),
        css="""
        details {
            margin: 15px 0;
            padding: 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
            background-color: #002B57;
        }
        summary {
            cursor: pointer;
            font-weight: bold;
            color: #667eea;
            font-size: 16px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔍 Fact-Check Agent (Pure Google ADK)
        
        **100% ADK Implementation** - Watch agents work in real-time!
        
        - 🎯 **6 ADK Agents** - SequentialAgent pipeline
        - 🔧 **ADK Tools** - FAISS + Google Search as FunctionTools
        - 🧠 **Live Progress** - Real-time event streaming
        - 💾 **Memory Caching** - Instant results for repeated queries
        """)
        
        with gr.Tab("💬 Verify Claims"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 💬 Enter Claim to Verify")
                    
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=600,
                        render_markdown=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Claim or URL",
                            placeholder="Enter news claim or paste URL...",
                            lines=3,
                            scale=4
                        )
                        submit_btn = gr.Button("🔍 Verify", scale=1, size="lg")
                    
                    clear_btn = gr.Button("🗑️ Clear")
                    
                    # Wire up streaming
                    submit_btn.click(
                        process_with_streaming,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    msg.submit(
                        process_with_streaming,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    clear_btn.click(lambda: [], outputs=chatbot)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Statistics")
                    stats_output = gr.Markdown()
                    refresh_btn = gr.Button("🔄 Refresh")
                    refresh_btn.click(get_stats, outputs=stats_output)
                    demo.load(get_stats, outputs=stats_output)
                    
                    gr.Markdown("""
                    ### ✨ ADK Pipeline
                    
                    **Agents:**
                    1. 📥 Ingestion
                    2. 🎯 Claim Extraction
                    3. 🔍 Evidence Retrieval
                    4. ✅ Verification
                    5. ⚖️ Aggregation
                    6. 📝 Report Generation
                    
                    **Tools:**
                    - FAISS Knowledge Base
                    - Google Search
                    
                    **Features:**
                    - Event streaming
                    - Session management
                    - Memory caching
                    """)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This System
            
            This is a **100% Google ADK implementation** of a fact-checking agent.
            
            ### Architecture
            
            **Main Pipeline:** `SequentialAgent`
            - Orchestrates 6 agents in sequence
            - Automatic data flow via `session.state`
            - Event-driven execution
            
            **Agents:**
            1. **IngestionAgent** (`LlmAgent`) - Cleans input text
            2. **ClaimExtractionAgent** (`LlmAgent`) - Extracts main claim
            3. **EvidenceRetrievalAgent** (`BaseAgent`) - Parallel retrieval
            4. **VerificationAgent** (`LlmAgent`) - Evaluates evidence
            5. **AggregatorAgent** (`LlmAgent`) - Generates verdict
            6. **ReportAgent** (`LlmAgent`) - Creates comprehensive report
            
            **Tools:**
            - `search_faiss_knowledge_base` (`FunctionTool`)
            - `search_google_for_current_info` (`FunctionTool`)
            
            ### ADK Benefits
            
            ✅ **Modular** - Each agent has one clear responsibility  
            ✅ **Transparent** - See agent thinking in real-time  
            ✅ **Maintainable** - Easy to modify and extend  
            ✅ **Production-Ready** - Built-in error handling  
            ✅ **Scalable** - Add agents without refactoring  
            
            ### Technology
            
            - **Google ADK** - Agent orchestration framework
            - **Gemini 2.5 Flash** - LLM for reasoning
            - **FAISS** - Semantic knowledge base
            - **Google Search** - Real-time verification
            - **SQLite** - Memory caching
            - **Gradio** - Web interface
            """)
    
    return demo


def main():
    """Main entry point"""
    try:
        print("🚀 Starting Fact-Check Agent (Pure ADK)...")
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