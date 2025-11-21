"""
Launcher script for Fake News Detection Agent UI
Run this file to start the Gradio web interface
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Gradio UI"""
    # Path to the Gradio app
    app_path = Path(__file__).parent / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: UI app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Fake News Detection Agent UI...")
    print("📍 Opening browser at http://localhost:8000")
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    # Launch Gradio
    try:
        subprocess.run([
            sys.executable, 
            str(app_path)
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server... Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
