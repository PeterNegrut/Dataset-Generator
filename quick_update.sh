#!/bin/bash
# 🚀 Quick Update & Restart Script
# Use this when you have code changes to test

echo "🔄 Quick Update & Restart"
echo "========================"

# Go to project directory
cd /root/dataset-generator-3 || { echo "❌ Project not found"; exit 1; }

# Kill existing server
echo "🛑 Stopping existing server..."
pkill -f "runpod_server.py" || echo "No server running"

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull

# Quick dependency check (no full reinstall)
echo "🧪 Quick dependency check..."
python3 -c "
try:
    import torch, diffusers, peft, flask
    print('✅ Dependencies OK')
except ImportError as e:
    print(f'❌ Missing: {e}')
    print('Run ./install_everything.sh if needed')
"

# Start server
echo "🚀 Starting server..."
./start_server.sh &
sleep 2

echo ""
echo "✅ Server restarted!"
echo "📡 http://localhost:39515/health"
