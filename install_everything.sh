#!/bin/bash
# 🚀 ONE-CLICK RUNPOD INSTALLATION 
# Expert-Fixed LoRA Server - RTX A5000 Optimized
# This script installs EVERYTHING needed in one go
# Version: 2025-07-22

set -e  # Exit on any error

echo "🚀 ONE-CLICK RUNPOD INSTALLATION"
echo "RTX A5000 (24GB VRAM) Expert-Optimized"
echo "========================================"
echo ""

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "backend/runpod_server.py" ]; then
    print_error "Please run this script from the dataset-generator root directory"
    print_error "Current directory: $(pwd)"
    print_error "Expected structure: backend/runpod_server.py should exist"
    exit 1
fi

print_success "Found backend/runpod_server.py - proceeding with installation"
echo ""

# Step 1: System Information
print_status "Step 1/8: Checking system..."
python3 -c "
import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'Existing PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
except ImportError:
    print('PyTorch not installed yet')
"
echo ""

# Step 2: Upgrade pip
print_status "Step 2/8: Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
print_success "Pip upgraded"
echo ""

# Step 3: Fix critical conflicts
print_status "Step 3/8: Fixing dependency conflicts..."
pip install --ignore-installed blinker==1.9.0 --quiet
print_success "Blinker conflict resolved"
echo ""

# Step 4: Install PyTorch stack (if needed)
print_status "Step 4/8: Installing PyTorch stack..."
if python3 -c "import torch; print('PyTorch already installed')" 2>/dev/null; then
    print_success "PyTorch already available"
else
    print_status "Installing PyTorch for RTX A5000..."
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --quiet
    print_success "PyTorch installed"
fi
echo ""

# Step 5: Install Hugging Face ecosystem
print_status "Step 5/8: Installing Hugging Face ecosystem..."
pip install --quiet \
    huggingface_hub==0.17.3 \
    diffusers==0.21.4 \
    transformers==4.33.2 \
    accelerate==0.23.0 \
    peft==0.5.0 \
    safetensors==0.4.1
print_success "Hugging Face stack installed"
echo ""

# Step 6: Install memory optimizations
print_status "Step 6/8: Installing memory optimizations..."
pip install --quiet xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118
pip install --quiet bitsandbytes==0.41.1
print_success "Memory optimizations installed"
echo ""

# Step 7: Install web framework and utilities
print_status "Step 7/8: Installing web framework and utilities..."
pip install --quiet \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    requests==2.31.0 \
    pillow==10.1.0 \
    numpy==1.24.4 \
    scipy==1.11.4 \
    psutil==5.9.6 \
    tqdm==4.66.1 \
    rich==13.7.0 \
    click==8.1.7 \
    python-dotenv==1.0.0
print_success "Web framework and utilities installed"
echo ""

# Step 8: Create environment configuration
print_status "Step 8/8: Setting up environment..."
cat > backend/.env << 'EOF'
# RTX A5000 Optimizations
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false

# Training Settings
DEFAULT_BATCH_SIZE=2
DEFAULT_GRADIENT_ACCUMULATION=2
DEFAULT_RANK=16
DEFAULT_LEARNING_RATE_GENERAL=1e-4
DEFAULT_LEARNING_RATE_XRAY=5e-5
EOF

# Set environment variables for current session
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

print_success "Environment configured"
echo ""

# Verification
print_status "🧪 VERIFICATION: Testing installation..."
python3 -c "
import sys
print('Testing critical imports...')

# Test PyTorch
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    print(f'✅ CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
except Exception as e:
    print(f'❌ PyTorch failed: {e}')
    sys.exit(1)

# Test Hugging Face
try:
    from huggingface_hub import cached_download
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig
    print('✅ Hugging Face stack working')
except Exception as e:
    print(f'❌ Hugging Face failed: {e}')
    sys.exit(1)

# Test web framework
try:
    import flask
    import flask_cors
    print('✅ Flask working')
except Exception as e:
    print(f'❌ Flask failed: {e}')
    sys.exit(1)

# Test server imports
try:
    import sys
    sys.path.append('backend')
    from runpod_server import app
    print('✅ Server imports working')
except Exception as e:
    print(f'❌ Server imports failed: {e}')
    sys.exit(1)

print('')
print('🎉 ALL TESTS PASSED!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 INSTALLATION COMPLETE!"
    echo "========================="
    echo ""
    print_success "✅ All dependencies installed successfully"
    print_success "✅ RTX A5000 optimizations applied"
    print_success "✅ Environment configured"
    print_success "✅ Server ready to start"
    echo ""
    echo "🚀 QUICK START:"
    echo "   1. Start server:  ./start_server.sh"
    echo "   2. Test health:   curl http://localhost:39515/health"
    echo ""
    echo "📡 Server Endpoints:"
    echo "   • POST /train-lora-expert    - Train LoRA with domain awareness"
    echo "   • POST /generate-expert      - Generate with auto-trigger detection"
    echo "   • POST /validate-lora        - Validate LoRA effectiveness"
    echo "   • GET  /health               - System status"
    echo ""
    echo "💡 RTX A5000 Settings:"
    echo "   • Batch Size: 2 (24GB VRAM optimized)"
    echo "   • Mixed Precision: fp16"
    echo "   • XFormers: Enabled"
    echo "   • Memory Management: Optimized"
    echo ""
    print_success "Ready to use! 🚀"
else
    print_error "Installation failed during verification"
    exit 1
fi
