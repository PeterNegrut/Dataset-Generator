#!/bin/bash
# 🚀 RunPod Quick Start Template
# Run this on a fresh RunPod PyTorch instance

echo "🚀 RunPod Quick Start - LoRA Server"
echo "==================================="

# Update system
echo "📦 Updating system..."
apt update && apt install -y git curl wget

# Clone repository (replace with your actual repo URL)
echo "📥 Cloning repository..."
cd /root
if [ ! -d "dataset-generator" ]; then
    # Replace this URL with your actual repository
    git clone https://github.com/YOUR_USERNAME/dataset-generator.git
    cd dataset-generator
else
    cd dataset-generator
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
echo "🚀 Start server:"
echo "   cd /root/dataset-generator"
echo "   ./start_server.sh"
echo ""
echo "🔧 Optional - Enable auto-start service:"
echo "   systemctl enable lora-server"
echo "   systemctl start lora-server"
echo ""
echo "📡 Server will be available at:"
echo "   http://localhost:39515/health"
