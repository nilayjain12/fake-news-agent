# ui/app.py
"""
Gradio UI for Fact-Checking Agent - Pure Google ADK
Uses Root Orchestrator with sequential LlmAgent pipeline
"""
import gradio as gr
import sys
import asyncio
from pathlib import Path
from datetime import datetime
import os

# Setup paths
BACKEND_PATH = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))
os.chdir(str(BACKEND_PATH))

from agents.root_orchestrator import root_orchestrator
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)

# Global state
memory = None
session_id = None


def initialize():
    """Initialize orchestrator and memory"""
    global memory, session_id
    
    if memory is None:
        try:
            memory = MemoryManager()
            session_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            memory.create_session(session_id, user_id="web-user")
            logger.warning(f"✅ ADK Orchestrator initialized: {session_id}")
        except Exception as e:
            logger.exception(f"❌ Initialization error: {e}")
            raise
    
    return memory


async def process_fact_check(user_input: str) -> str:
    """
    Process fact-check using ADK Root Orchestrator
    Returns the comprehensive report
    """
    memory_instance = initialize()
    
    if not user_input.strip():
        return "Please enter a claim to verify."
    
    try:
        logger.warning("🔍 Query received: %s", user_input[:80])
        
        # Run the ADK pipeline
        result = await root_orchestrator.run_pipeline(
            input_text=user_input,
            user_id="web-user",
            session_id=session_id
        )
        
        if result["success"]:
            report = result.get("report", "")
            execution_time = result.get("execution_time_ms", 0)
            
            # Format response with timing and API calls
            api_calls = result.get("api_calls", 0)
            response = f"""{report}

---

⏱️ **Execution Time:** {execution_time:.0f}ms  
📊 **API Calls:** {api_calls}/2 (Optimized Pipeline)"""
            
            # Cache result if verdict exists
            verdict = result.get("verdict", "")
            if verdict and verdict != "ERROR":
                try:
                    memory_instance.cache_verdict(
                        claim=user_input[:500],
                        verdict=verdict,
                        confidence=0.7,
                        evidence_count=1,
                        session_id=session_id
                    )
                    logger.warning("💾 Result cached")
                except Exception as e:
                    logger.warning("⚠️ Caching failed: %s", str(e)[:50])
            
            # Log interaction
            try:
                memory_instance.add_interaction(
                    session_id=session_id,
                    query=user_input[:200],
                    processed_input=user_input[:500],
                    verdict=verdict or "UNKNOWN"
                )
                logger.warning("📝 Interaction logged")
            except Exception as e:
                logger.warning("⚠️ Logging failed: %s", str(e)[:50])
            
            return response
        else:
            error = result.get("error", "Unknown error")
            return f"❌ **Error:** {error}"
    
    except Exception as e:
        logger.exception("❌ Processing error: %s", e)
        return f"❌ **Error:** {str(e)[:200]}"


def get_stats() -> str:
    """Get system statistics"""
    try:
        memory_instance = initialize()
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


def chat_interface(message: str, history: list):
    """Chat interface that handles async execution"""
    if not message.strip():
        return history, ""
    
    # Add user message
    history.append([message, "⏳ Processing..."])
    
    try:
        # Run async pipeline
        result = asyncio.run(process_fact_check(message))
        
        # Update with result
        history[-1][1] = result
        
    except Exception as e:
        logger.exception(f"❌ Chat error: {e}")
        history[-1][1] = f"❌ Error: {str(e)[:200]}"
    
    return history, ""


def create_interface():
    """Create Gradio interface"""
    
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
        .verdict-true {
            background-color: #2ecc71;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .verdict-false {
            background-color: #e74c3c;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .verdict-inconclusive {
            background-color: #f39c12;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🔍 Fact-Check Agent (Pure Google ADK)
        
        **Sequential ADK Pipeline** - 5 LLM Agents in Sequence
        
        - 🎯 **5 ADK LlmAgents** - Ingestion → Extraction → Verification → Aggregation → Report
        - 🔧 **ADK Tools** - FAISS + Google Search as FunctionTools  
        - 📊 **Real-time Processing** - Sequential async execution
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
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear History", scale=1)
                        example_btn = gr.Button("📝 Try Example", scale=1)
                    
                    # Wire up interactions
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
                        examples = [
                            "The Earth revolves around the Sun.",
                            "Water boils at 100 degrees Celsius at sea level.",
                            "The Great Wall of China is visible from space with the naked eye.",
                            "Vitamin C prevents the common cold.",
                        ]
                        import random
                        return random.choice(examples)
                    
                    example_btn.click(load_example, outputs=msg)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Statistics")
                    stats_output = gr.Markdown()
                    refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    refresh_btn.click(get_stats, outputs=stats_output)
                    demo.load(get_stats, outputs=stats_output)
                    
                    gr.Markdown("""
                    ### ✨ ADK Pipeline
                    
                    **5 Sequential Agents:**
                    1. 📥 Ingestion
                    2. 🎯 Extraction
                    3. 🔍 Verification
                    4. ⚖️ Aggregation
                    5. 📄 Report
                    
                    **Tools:**
                    - FAISS Knowledge Base
                    - Google Search
                    
                    **Features:**
                    - Sequential execution
                    - Session management
                    - Memory caching
                    - Async processing
                    """)
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This System
            
            This is a **Pure Google ADK implementation** of a fact-checking agent system.
            
            ### Architecture
            
            **Root Orchestrator** (`root_orchestrator.py`)
            - Coordinates 5 LlmAgent sub-agents
            - Sequential execution via async/await
            - Uses Google ADK `Runner` + `InMemorySessionService`
            
            **5 Sequential Agents (All LlmAgent):**
            1. **IngestionAgent** - Cleans input text or extracts from URLs
            2. **ExtractionAgent** - Extracts main verifiable claim
            3. **VerificationAgent** - Searches FAISS + Google for evidence
            4. **AggregationAgent** - Counts evidence and generates verdict
            5. **ReportAgent** - Creates comprehensive fact-check report
            
            **Tools (FunctionTool):**
            - `extract_url_content()` - Web scraping
            - `validate_and_clean_text()` - Text cleaning
            - `extract_main_claim()` - LLM-based claim extraction
            - `search_faiss_knowledge_base()` - Semantic search
            - `search_google()` - Real-time web search
            - `evaluate_evidence()` - Evidence classification
            - `count_evidence()` - Evidence counting
            - `generate_verdict()` - Verdict generation
            - `format_report()` - Report formatting
            
            ### Technology Stack
            
            - **Google ADK** - Agent orchestration
            - **Gemini 2.5 Flash** - LLM reasoning
            - **FAISS** - Semantic knowledge base search
            - **Google Search** - Real-time verification
            - **SQLite** - Session & memory caching
            - **Gradio** - Web interface
            
            ### How It Works
            
            1. **User Input** → Ingestion Agent (cleans text)
            2. **Clean Text** → Extraction Agent (extracts claim)
            3. **Claim** → Verification Agent (searches evidence)
            4. **Evidence** → Aggregation Agent (generates verdict)
            5. **Verdict** → Report Agent (creates report)
            6. **Report** → User sees comprehensive fact-check result
            
            ### Benefits of ADK
            
            ✅ **Modular** - Each agent has clear responsibility  
            ✅ **Transparent** - See agent progression  
            ✅ **Maintainable** - Easy to modify agents  
            ✅ **Production-Ready** - Built-in error handling  
            ✅ **Scalable** - Add/remove agents easily  
            ✅ **Native Tools** - Google ADK FunctionTools  
            ✅ **Session Management** - Built-in InMemorySessionService  
            ✅ **Async-Ready** - Full asyncio integration  
            """)
    
    return demo


def main():
    """Main entry point"""
    try:
        print("🚀 Starting Fact-Check Agent (Pure ADK)...")
        print("📍 Opening at http://0.0.0.0:8000")
        
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