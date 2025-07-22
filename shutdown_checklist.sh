#!/bin/bash
# 💾 RunPod Shutdown Checklist
# Run this before closing your RunPod to save money

echo "💾 RunPod Shutdown Checklist"
echo "============================"

# Kill any running training processes
echo "🛑 Stopping training processes..."
pkill -f "train_dreambooth_lora" || true
pkill -f "runpod_server.py" || true

# Check GPU usage
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

# Push any final changes
echo "📤 Checking for unsaved changes..."
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️ You have unsaved changes!"
    git status --short
    echo ""
    echo "💡 To save changes before shutdown:"
    echo "   git add ."
    echo "   git commit -m 'Save before shutdown'"
    echo "   git push"
else
    echo "✅ All changes saved to git"
fi

echo ""
echo "🔧 Next time you start a fresh RunPod:"
echo "   1. Upload your SSH public key to RunPod"
echo "   2. Run: wget -O- https://raw.githubusercontent.com/PeterNegrut/dataset-generator/main/runpod_quickstart.sh | bash"
echo "   3. Or manually: git clone <your-repo> && cd dataset-generator && ./install_everything.sh"
echo ""
echo "💡 Your repository URL:"
echo "   https://github.com/PeterNegrut/dataset-generator.git"
echo ""
echo "✅ Safe to shutdown RunPod now!"
