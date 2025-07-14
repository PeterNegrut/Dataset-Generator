#!/usr/bin/env python3
"""
Simple startup script for AI Dataset Generator
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("\n🔧 Try these troubleshooting steps:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Use Python 3.11 or earlier if you have compatibility issues")
        print("3. Install manually: pip install Flask==3.0.0 Flask-CORS==4.0.0 requests==2.31.0")
        return False

def check_runpod_config():
    """Check if RunPod API key is configured"""
    try:
        from app import RUNPOD_API_KEY
        if RUNPOD_API_KEY and RUNPOD_API_KEY.startswith('rpa_'):
            print("✅ RunPod API key configured")
            return True
        else:
            print("⚠️ RunPod API key not found - update app.py with your API key")
            return False
    except ImportError:
        print("⚠️ Could not check RunPod configuration")
        return False

def start_backend():
    """Start the Flask backend"""
    print("🚀 Starting AI Dataset Generator backend...")
    print("📍 Backend will be available at: http://localhost:5000")
    print("📍 Open your browser and navigate to your frontend HTML file")
    print("=" * 60)
    
    try:
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down backend...")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")

def main():
    print("🎯 AI Dataset Generator - Startup Script")
    print("=" * 50)
    
    # Check environment
    check_python_version()
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Cannot continue without required packages")
        sys.exit(1)
    
    # Check configuration
    check_runpod_config()
    
    # Start the backend
    start_backend()

if __name__ == "__main__":
    main()