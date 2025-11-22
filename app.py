"""
HF Spaces Entry Point for Fake News Detection Agent
This file is automatically run by HF Spaces
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Set environment for HF Spaces
os.environ["HF_SPACES"] = "true"

# Import and launch
from ui.app import create_interface

if __name__ == "__main__":
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )