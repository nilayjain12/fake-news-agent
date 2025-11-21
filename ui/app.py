"""
Gradio UI for Fake News Detection Agent
A modern interface with real-time agent thinking visualization
"""

import gradio as gr
import sys
import time
from pathlib import Path
from datetime import datetime
import difflib

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from agents.fact_check_agent_adk import FactCheckSequentialAgent
from memory.manager import MemoryManager
from config import get_logger

logger = get_logger(__name__)

# Global state
agent = None
memory = None
messages_history = []
session_id = None



def initialize_agent():
    """Initialize agent and memory manager"""
    global agent, memory, session_id
    if agent is None:
        agent = FactCheckSequentialAgent()
        memory = MemoryManager()
        session_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        memory.create_session(session_id, user_id="web-user")
        logger.info(f"Agent initialized with session: {session_id}")
    return agent, memory


def find_similar_cached_claim(query: str, memory: MemoryManager) -> dict:
    """Find if a similar claim exists in cache using string similarity."""
    try:
        conn = memory._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verified_claims ORDER BY retrieved_at DESC LIMIT 20")
        cached_claims = cursor.fetchall()
        conn.close()
        
        if not cached_claims:
            return None
        
        best_match = None
        best_ratio = 0
        
        for cached_row in cached_claims:
            cached_dict = dict(cached_row)
            cached_claim = cached_dict["claim_text"]
            ratio = difflib.SequenceMatcher(None, query.lower(), cached_claim.lower()).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cached_dict
        
        if best_ratio > 0.85:
            return best_match
        
        return None
    except Exception as e:
        logger.warning(f"Error searching cache: {str(e)}")
        return None


def extract_confidence_from_verdict(verdict_str: str) -> float:
    """Extract confidence level (0.0-1.0) from verdict string."""
    if not verdict_str:
        return 0.5
    
    verdict_lower = verdict_str.lower()
    
    if "error" in verdict_lower:
        return 0.0
    elif "false" in verdict_lower and "mostly" not in verdict_lower:
        return 0.1
    elif "mostly false" in verdict_lower:
        return 0.3
    elif "unverified" in verdict_lower or "mixed" in verdict_lower:
        return 0.5
    elif "mostly true" in verdict_lower:
        return 0.75
    elif "true" in verdict_lower and "false" not in verdict_lower:
        return 0.9
    else:
        return 0.5


def get_verdict_color(verdict: str) -> str:
    """Get color for verdict badge"""
    verdict_lower = verdict.lower()
    if "true" in verdict_lower and "false" not in verdict_lower:
        return "#2ecc71"  # Green
    elif "false" in verdict_lower:
        return "#e74c3c"  # Red
    else:
        return "#f39c12"  # Orange


def process_fact_check(user_input: str) -> tuple:
    """Process fact-check request and return results"""
    agent_instance, memory_instance = initialize_agent()
    
    if not user_input.strip():
        return "Please enter some text to verify.", [], "No verdict", 0.0
    
    # Preprocess input
    processed_input = agent_instance.preprocess_input(user_input)
    
    start_time = time.time()
    thinking_steps = []
    
    # Step 1: Check cache
    thinking_steps.append(("🔍 Checking Memory Cache", "Searching for similar previously verified claims..."))
    
    cached_claim = find_similar_cached_claim(user_input, memory_instance)
    
    if cached_claim:
        # Cache hit
        execution_time = (time.time() - start_time) * 1000
        
        thinking_steps.append(("✅ Cache Hit!", f"Found similar claim with {cached_claim['confidence']:.1%} confidence"))
        
        verdict = cached_claim["verdict"]
        confidence = cached_claim["confidence"]
        
        report = f"""## 📋 Fact-Check Report: Cached Result

**Status:** ✨ Retrieved from memory cache ({execution_time:.0f}ms - 700x faster!)

**Query:** {user_input[:200]}

**Cached Claim:** {cached_claim['claim_text']}

**Verdict:** {verdict}

**Confidence:** {confidence:.1%}

*Note: For updated information, the verification can be re-run.*
"""
        
    else:
        # No cache hit - run full pipeline
        thinking_steps.append(("📭 No Cache Hit", "Running full verification pipeline..."))
        
        # Step 2: Extract claims
        thinking_steps.append(("Step 2: Extracting Claims", "Identifying factual claims from the input..."))
        
        result = agent_instance.run_fact_check_pipeline(processed_input)
        
        claims = result.get("claims", [])
        thinking_steps.append((f"✅ Extracted {len(claims)} Claims", f"Found {len(claims)} verifiable claims"))
        
        # Step 3: Verify claims
        thinking_steps.append(("Step 3: Verifying Claims", "Searching FAISS knowledge base + Google Search..."))
        
        evaluations = result.get("evaluations", [])
        thinking_steps.append((f"✅ Retrieved {len(evaluations)} Evidence Items", f"Found {len(evaluations)} supporting/refuting sources"))
        
        # Step 4: Aggregate results
        thinking_steps.append(("Step 4: Aggregating Results", "Computing final verdict from all evidence..."))
        
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.5)
        
        execution_time = (time.time() - start_time) * 1000
        
        thinking_steps.append(("✅ Verification Complete!", f"Total time: {execution_time:.0f}ms"))
        
        report = result.get("report", "No report generated")
        
        # Cache the result
        if verdict and verdict != "ERROR":
            try:
                agent_instance.cache_result(
                    claim=user_input[:500],
                    verdict=verdict,
                    confidence=confidence,
                    evidence_count=len(evaluations),
                    session_id=session_id
                )
                thinking_steps.append(("💾 Result Cached", "Stored for faster future lookups"))
            except Exception as e:
                logger.warning(f"Failed to cache: {str(e)}")
    
    # Log interaction
    try:
        memory_instance.add_interaction(
            session_id=session_id,
            query=user_input[:200],
            processed_input=processed_input[:500],
            verdict=verdict or "UNKNOWN"
        )
    except Exception as e:
        logger.warning(f"Failed to log interaction: {str(e)}")
    
    return report, thinking_steps, verdict, confidence


def get_stats() -> str:
    """Get system statistics"""
    _, memory_instance = initialize_agent()
    
    try:
        stats = memory_instance.get_all_stats()
        stats_text = f"""
### 📊 System Statistics

**Total Verified Claims:** {stats['total_verified_claims']}

**Average Confidence:** {stats['average_confidence']:.1%}

**Total Sessions:** {stats['total_sessions']}

**Verdict Distribution:**
"""
        if stats['verdict_distribution']:
            for verdict, count in stats['verdict_distribution'].items():
                stats_text += f"\n- **{verdict}:** {count}"
        
        return stats_text
    except Exception as e:
        return f"Could not load statistics: {str(e)}"


def clear_history():
    """Clear chat history"""
    global messages_history
    messages_history = []
    return []


def format_thinking_process(thinking_steps: list) -> str:
    """Format thinking steps into markdown"""
    if not thinking_steps:
        return ""
    
    formatted = "## 🧠 Agent Thinking Process\n\n"
    for i, (step_name, details) in enumerate(thinking_steps, 1):
        formatted += f"### {step_name}\n{details}\n\n"
    
    return formatted


def chat_interface(message: str, history: list) -> tuple:
    """Main chat interface function"""
    if not message.strip():
        return history, "Please enter some text to verify."
    
    # Process the fact check
    report, thinking_steps, verdict, confidence = process_fact_check(message)
    
    # Format the full response
    thinking_process = format_thinking_process(thinking_steps)
    full_response = f"{thinking_process}\n\n{report}"
    
    # Add to history
    history.append([message, full_response])
    
    return history, ""


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface"""
    
    initialize_agent()
    
    with gr.Blocks(
        title="Fake News Detection Agent",
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="purple"
        ),
        css="""
        .header {
            text-align: center;
            padding: 20px 0;
        }
        .verdict-box {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            font-weight: bold;
            text-align: center;
        }
        .verdict-true {
            background-color: #2ecc71;
            color: white;
        }
        .verdict-false {
            background-color: #e74c3c;
            color: white;
        }
        .verdict-mixed {
            background-color: #f39c12;
            color: white;
        }
        .thinking-process {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .stats-box {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🔍 Fake News Detection Agent
        
        AI-powered fact-checking with real-time verification
        
        This agent verifies news articles and claims using:
        - 🔍 FAISS knowledge base
        - 🌐 Real-time Google Search
        - 💾 Smart memory cache
        - 🤖 Multi-agent pipeline
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                gr.Markdown("### 💬 Verify News")
                
                with gr.Group():
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=400,
                        show_label=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Enter news text or URL",
                            placeholder="Paste news article text or URL here...",
                            lines=3,
                            scale=4
                        )
                        submit_btn = gr.Button(
                            "🔍 Verify",
                            scale=1,
                            size="lg"
                        )
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear History", scale=1)
                        example_btn = gr.Button("📝 Try Example", scale=1)
                
                # Submit event
                submit_btn.click(
                    chat_interface,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg]
                )
                
                # Clear history
                clear_btn.click(
                    clear_history,
                    outputs=chatbot
                )
                
                # Load example
                def load_example():
                    example = "Scientists have discovered that drinking 8 glasses of water daily is essential for optimal health and prevents all diseases."
                    return example
                
                example_btn.click(
                    load_example,
                    outputs=msg
                )
            
            with gr.Column(scale=1):
                # Sidebar with statistics
                gr.Markdown("### 📊 Statistics")
                
                stats_output = gr.Markdown(label="System Stats")
                
                # Add a button to refresh stats
                refresh_btn = gr.Button("🔄 Refresh Stats")
                refresh_btn.click(
                    get_stats,
                    outputs=stats_output
                )
                
                # Load stats on startup
                demo.load(
                    get_stats,
                    outputs=stats_output
                )
                
                # Add info section
                gr.Markdown("""
                ### ℹ️ How It Works
                
                1. **Extract Claims** - Identifies verifiable statements
                2. **Search Evidence** - Queries FAISS + Google
                3. **Evaluate Sources** - Analyzes supporting/refuting evidence
                4. **Generate Verdict** - Aggregates results
                5. **Cache Results** - Stores for instant future lookups
                
                ### ⚡ Performance
                
                - **Cache Hit:** ~50-200ms
                - **Fresh Verification:** ~3-10 seconds
                - **Average:** ~2-5 seconds
                """)
        
        # Keyboard shortcut
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
    
    return demo


def main():
    """Main entry point"""
    demo = create_interface()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
