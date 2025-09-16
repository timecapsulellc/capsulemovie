#!/usr/bin/env python3
"""
Simple launch script for SkyReels V2 Web UI
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import gradio
        import torch
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def setup_environment():
    """Setup environment variables and paths."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    capsule_movie_path = project_root / "Capsule Movie"
    
    if capsule_movie_path.exists():
        sys.path.insert(0, str(capsule_movie_path))
        os.environ['PYTHONPATH'] = str(capsule_movie_path) + ':' + os.environ.get('PYTHONPATH', '')
        print(f"✅ Added {capsule_movie_path} to Python path")
    else:
        print(f"⚠️ Warning: {capsule_movie_path} not found")

def main():
    print("🎬 SkyReels V2 Web UI Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Create necessary directories
    os.makedirs("result/web_ui", exist_ok=True)
    print("✅ Created output directories")
    
    # Launch the web UI
    print("🚀 Starting SkyReels V2 Web UI...")
    print("📱 The interface will be available at:")
    print("   Local: http://localhost:7860")
    print("   Network: http://0.0.0.0:7860")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Import and run the web UI
        from app import create_interface
        
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
            show_api=False
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down SkyReels V2 Web UI...")
    except Exception as e:
        print(f"❌ Error launching web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()