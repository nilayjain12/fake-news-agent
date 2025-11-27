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
import random
import os

# ==========================================
# FIX: Set up proper Python path
# ==========================================
# Get the absolute path to the backend directory
BACKEND_PATH = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set working directory to backend for relative imports
os.chdir(str(BACKEND_PATH))

# Now import backend modules
try:
    from agents.fact_check_agent_adk import FactCheckSequentialAgent
    from agents.image_processing_agent import ImageProcessingAgent
    from memory.manager import MemoryManager
    from config import get_logger
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Backend path: {BACKEND_PATH}")
    print(f"Backend path exists: {BACKEND_PATH.exists()}")
    sys.exit(1)

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
        try:
            agent = FactCheckSequentialAgent()
            memory = MemoryManager()
            session_id = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            memory.create_session(session_id, user_id="web-user")
            logger.warning(f"Agent initialized with session: {session_id}")
        except Exception as e:
            logger.warning(f"Error initializing agent: {str(e)}")
            raise
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
    """Process fact-check request and return results with detailed reports."""
    agent_instance, memory_instance = initialize_agent()
    
    if not user_input.strip():
        return "", [], "", 0.0
    
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
        
        final_assessment = f"""### ✅ Fact-Check Report: Cached Result

**Status:** Retrieved from memory cache ({execution_time:.0f}ms - 700x faster!)

**Cached Claim:** {cached_claim['claim_text']}

**Verdict:** **{verdict}**

**Confidence:** {confidence:.1%}

*Note: For updated information, re-run the verification.*
"""
        
    else:
        # No cache hit - run full pipeline
        thinking_steps.append(("📭 No Cache Hit", "Running full verification pipeline..."))
        
        # Run the complete pipeline (now returns detailed reports)
        logger.warning("Running fact-check pipeline...")
        result = agent_instance.run_fact_check_pipeline(processed_input)
        
        # Extract results
        claims = result.get("claims", [])
        detailed_reports = result.get("detailed_reports", [])
        comprehensive_report = result.get("comprehensive_report", "")
        overall_verdict = result.get("overall_verdict", "UNKNOWN")
        total_evidence = result.get("total_evidence_items", 0)
        
        thinking_steps.append((f"✅ Extracted {len(claims)} Claims", 
                            f"Identified {len(claims)} verifiable claims from input"))
        
        thinking_steps.append((f"✅ Retrieved {total_evidence} Evidence Items", 
                            f"Searched FAISS + Google for supporting/refuting sources"))
        
        thinking_steps.append((f"✅ Generated {len(detailed_reports)} Detailed Reports", 
                            f"Created comprehensive analysis for each claim"))
        
        # Use comprehensive report as final assessment
        final_assessment = comprehensive_report
        
        verdict = overall_verdict
        confidence = _calculate_average_confidence(detailed_reports) if detailed_reports else 0.5
        
        execution_time = (time.time() - start_time) * 1000
        thinking_steps.append(("✅ Verification Complete!", f"Total time: {execution_time:.0f}ms"))
        
        # Cache the overall result
        if verdict and verdict != "ERROR":
            try:
                agent_instance.cache_result(
                    claim=user_input[:500],
                    verdict=verdict,
                    confidence=confidence,
                    evidence_count=total_evidence,
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
    
    return user_input, thinking_steps, final_assessment, confidence

def _calculate_average_confidence(detailed_reports: list) -> float:
    """Calculate average confidence from detailed reports."""
    if not detailed_reports:
        return 0.5
    
    total_confidence = sum(r["result"]["confidence_percentage"] for r in detailed_reports) / 100
    return total_confidence / len(detailed_reports)

def format_response(user_query: str, thinking_steps: list, final_assessment: str, confidence: float) -> str:
    """Format response with single claim report."""
    
    thinking_content = format_thinking_process(thinking_steps)
    
    # No need to change this - final_assessment is already markdown
    response = f"""<details style="margin: 15px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #FFB300;">
<summary style="cursor: pointer; font-weight: bold; color: #000000;">🧠 Agent Thinking Process</summary>

{thinking_content}

</details>

{final_assessment}
"""
    return response

def process_image_verification(image_path):
    """Process image-based verification."""
    if image_path is None:
        return "Please upload an image first."
    
    try:
        agent_instance, memory_instance = initialize_agent()
        
        logger.warning(f"🖼️  Processing image: {image_path}")
        
        # Run image pipeline
        result = agent_instance.run_fact_check_pipeline_with_image(image_path)
        
        report = result.get("report", "No report")
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0.0)
        
        # Cache result
        try:
            agent_instance.cache_result(
                claim=f"Image-based: {verdict}",
                verdict=verdict,
                confidence=confidence,
                evidence_count=len(result.get("evaluations", [])),
                session_id=session_id
            )
        except Exception as e:
            logger.warning(f"Failed to cache image result: {str(e)}")
        
        return report
        
    except Exception as e:
        logger.warning(f"Error processing image: {str(e)}")
        return f"❌ Error processing image: {str(e)[:200]}"


def get_stats() -> str:
    """Get system statistics"""
    try:
        _, memory_instance = initialize_agent()
        
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
        logger.warning(f"Error getting stats: {str(e)}")
        return f"Could not load statistics: {str(e)}"


def clear_history():
    """Clear chat history"""
    global messages_history
    messages_history = []
    return []


def format_thinking_process(thinking_steps: list) -> str:
    """Format thinking steps into markdown with collapsible section"""
    if not thinking_steps:
        return ""
    
    formatted = ""
    for i, (step_name, details) in enumerate(thinking_steps, 1):
        formatted += f"**{step_name}**  \n{details}\n\n"
    
    return formatted


def chat_interface(message: str, history: list) -> tuple:
    """Main chat interface function with detailed reports."""
    if not message.strip():
        return history, ""
    
    # Add user message immediately to history
    history.append([message, "⏳ Processing your request..."])
    
    try:
        # Process the fact check
        user_query, thinking_steps, final_assessment, confidence = process_fact_check(message)
        
        # Format the complete response with thinking steps in collapsible section
        formatted_response = format_response(user_query, thinking_steps, final_assessment, confidence)
        
        # Update the last message with the complete response
        history[-1][1] = formatted_response
        
    except Exception as e:
        logger.warning(f"Error processing request: {str(e)}")
        history[-1][1] = f"❌ Error: {str(e)[:200]}\n\nPlease try again with different input."
    
    return history, ""


# Create Gradio interface
def create_interface():
    """Create and return the Gradio interface (UPDATED)"""
    
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
            background-color: #ffffff;
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
        .report-section {
            background-color: #f0f7ff;
            padding: 15px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
            border-radius: 5px;
        }
        .source-box {
            background-color: #e8f5e9;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #4caf50;
        }
        details {
            margin: 15px 0;
            padding: 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
            background-color: #ffffff;
        }
        details[open] {
            background-color: #f0f7ff;
        }
        summary {
            cursor: pointer;
            font-weight: bold;
            color: #667eea;
            user-select: none;
            padding: 5px;
            margin: -5px;
        }
        summary:hover {
            color: #5568d3;
            background-color: #e8f0ff;
            border-radius: 4px;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🔍 Fake News Detection Agent
        
        AI-powered fact-checking with **detailed analysis and source verification**
        
        This agent now provides:
        - 📊 **Detailed Verdicts** - True / False / Mostly True / Mostly False with confidence scores
        - 📝 **Explanations** - Why the information is true or false based on sources
        - 🔗 **Source Attribution** - Links and snippets from verified sources
        - 📈 **Scoring Breakdown** - How the agent calculated the verdict (SUPPORTS vs REFUTES)
        """)
        
        with gr.Tab("💬 Text Verification"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 💬 Verify News Claims")
                    
                    with gr.Group():
                        chatbot = gr.Chatbot(
                            label="Chat History",
                            height=500,
                            show_label=True,
                            render_markdown=True
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
                    
                    submit_btn.click(
                        chat_interface,
                        inputs=[msg, chatbot],
                        outputs=[chatbot, msg]
                    )
                    
                    clear_btn.click(clear_history, outputs=chatbot)
                    
                    def load_example():
                        return random.choice([
                            "The Earth revolves around the Sun once every 365 days.",
                            "Water boils at 100 degrees Celsius at sea level.",
                            "The Great Wall of China is visible from space with the naked eye.",
                            "Vitamin C prevents the common cold in all cases.",
                            "Goldfish have a memory span of only 3 seconds."
                        ])
                    
                    example_btn.click(load_example, outputs=msg)
                    
                    msg.submit(chat_interface, inputs=[msg, chatbot], outputs=[chatbot, msg])
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Statistics")
                    stats_output = gr.Markdown(label="System Stats")
                    refresh_btn = gr.Button("🔄 Refresh Stats")
                    refresh_btn.click(get_stats, outputs=stats_output)
                    demo.load(get_stats, outputs=stats_output)
                    
                    gr.Markdown("""
                    ### 📋 New Report Features
                    
                    **Result:** Verdict with confidence %
                    
                    **Explanation:** Why true/false
                    
                    **Sources:** Links & snippets
                    
                    **Scoring:** SUPPORTS vs REFUTES breakdown
                    """)
        
        with gr.Tab("📸 Image Verification"):
            gr.Markdown("### 📸 Verify Claims from Images")
            
            with gr.Group():
                image_input = gr.Image(label="Upload Image", type="filepath", scale=1)
                image_verify_btn = gr.Button("🔍 Verify Image", size="lg")
                image_output = gr.Markdown(label="Verification Result")
            
            image_verify_btn.click(
                process_image_verification,
                inputs=image_input,
                outputs=image_output
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About Fact-Checking Reports
            
            Each claim is now analyzed with **four key components**:
            
            ### 1️⃣ Result
            - **Verdict:** True / False / Mostly True / Mostly False
            - **Confidence:** Percentage confidence in the verdict
            - **Level:** Very High / High / Moderate / Low / Very Low
            
            ### 2️⃣ Explanation
            - Why the verdict was reached
            - Number of supporting/refuting sources
            - Evidence balance analysis
            
            ### 3️⃣ Sources
            - Direct links to verification sources
            - Source snippets showing evidence
            - Whether each source supports or refutes
            
            ### 4️⃣ Scoring Breakdown
            - SUPPORTS count (evidence agreeing with claim)
            - REFUTES count (evidence contradicting claim)
            - NOT_ENOUGH_INFO count (insufficient evidence)
            - Raw and normalized scores
            - Critical factors that influenced verdict
            
            ### Technology Stack
            - **Google ADK** - Multi-agent orchestration
            - **Gemini 2.5 Flash** - LLM-based claim extraction & evaluation
            - **FAISS** - Semantic search on knowledge base
            - **Google Search** - Real-time web verification
            """)
    
    return demo


def main():
    """Main entry point"""
    try:
        print("🚀 Initializing Fake News Detection Agent UI...")
        demo = create_interface()
        
        # Launch the interface
        print("📍 Starting server at http://0.0.0.0:8000")
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error starting UI: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()