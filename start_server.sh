#!/bin/bash
# 🚀 Start Expert-Fixed LoRA Server
# One-command server startup

set -e

echo "🚀 Starting Expert-Fixed LoRA Server"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "backend/runpod_server.py" ]; then
    echo "⚠️  Please run this script from the dataset-generator root directory"
    exit 1
fi

# Load environment variables
if [ -f "backend/.env" ]; then
    echo "📝 Loading RTX A5000 optimizations..."
    export $(cat backend/.env | grep -v '^#' | xargs)
else
    echo "⚙️ Setting default RTX A5000 optimizations..."
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
    export CUDA_VISIBLE_DEVICES=0
    export TOKENIZERS_PARALLELISM=false
fi

# Quick dependency check
echo "🧪 Quick system check..."
python3 -c "
try:
    import torch, diffusers, peft, flask
    print(f'✅ All dependencies ready')
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'❌ Missing dependencies: {e}')
    print('Run ./install_everything.sh first')
    exit(1)
"

echo ""
echo "🌐 Starting server on port 39515..."
echo "📡 Endpoints:"
echo "   • http://0.0.0.0:39515/health"
echo "   • http://0.0.0.0:39515/train-lora-expert"
echo "   • http://0.0.0.0:39515/generate-expert"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python3 runpod_server.py
