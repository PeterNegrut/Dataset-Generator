#!/bin/bash
# 🚀 RunPod Quick Start Template
# Run this on a fresh RunPod PyTorch instance

echo "🚀 RunPod Quick Start - LoRA Server"
echo "==================================="

# Update system
echo "📦 Updating system..."
apt update && apt install -y git curl wget

# Clone repository
echo "📥 Cloning repository..."
cd /root
if [ ! -d "dataset-generator" ]; then
    # Clone the repository
    git clone https://github.com/PeterNegrut/dataset-generator.git
    cd dataset-generator
else
    cd dataset-generator
    echo "📄 Repository exists, pulling latest changes..."
    git pull
fi

# Install everything
echo "🚀 Installing dependencies..."
chmod +x install_everything.sh
./install_everything.sh

# Create service file for auto-start (optional)
cat > /etc/systemd/system/lora-server.service << EOF
[Unit]
Description=LoRA Training Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/dataset-generator
ExecStart=/root/dataset-generator/start_server.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "✅ RunPod setup complete!"
echo ""
echo "🚀 Starting server automatically..."
cd /root/dataset-generator
./start_server.sh &
SERVER_PID=$!
sleep 3

echo ""
echo "🔧 Optional - Enable auto-start service:"
echo "   systemctl enable lora-server"
echo "   systemctl start lora-server"
echo ""
echo "📡 Server is running at:"
echo "   http://localhost:39515/health"
echo ""
echo "🔍 Check server status:"
echo "   ./check_server.sh"
echo ""
echo "🔑 For SSH automation setup:"
echo "   ./setup_ssh_automation.sh"
echo ""
echo "✨ Server started with PID: $SERVER_PID"
echo "   To stop: kill $SERVER_PID"
