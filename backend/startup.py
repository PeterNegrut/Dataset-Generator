#!/usr/bin/env python3
"""
AI Dataset Generator - Simplified Startup Script
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🎯 AI Dataset Generator v2.0 - Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Install requirements
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)
    
    # Start the backend
    print("\n🚀 Starting AI Dataset Generator backend...")
    print("📍 Backend will be available at: http://localhost:5000")
    print("📍 Open index.html in your browser to use the app")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down backend...")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")

if __name__ == "__main__":
    main()