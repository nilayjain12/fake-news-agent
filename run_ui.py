"""
FIXED: Direct Gradio launch (no subprocess)
- Removes subprocess overhead
- Allows proper async event loop sharing
- Better error handling
"""

import sys
import os
from pathlib import Path

# Setup paths BEFORE any imports
BACKEND_PATH = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))
os.chdir(str(BACKEND_PATH))

def main():
    """Launch Gradio app directly"""
    try:
        print("🚀 Starting Fact-Check Agent UI...")
        print("📍 http://0.0.0.0:8000")
        print("⌨️  Press Ctrl+C to stop the server\n")
        
        # Import AFTER path setup
        from ui.app import create_interface
        
        # Create and launch interface
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down... Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()