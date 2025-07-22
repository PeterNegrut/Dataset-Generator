#!/bin/bash
# Installation script for LoRA server dependencies

echo "🚀 Starting installation of LoRA server dependencies..."

# Install core dependencies from requirements.txt
pip install -r requirements.txt

# Install heavy AI dependencies for RunPod
pip install torch>=2.0.0 torchvision>=0.15.0
pip install diffusers==0.18.2
pip install transformers>=4.25.0
pip install accelerate>=0.20.0
pip install safetensors>=0.3.0
pip install peft

echo "✅ All dependencies installed successfully!"
