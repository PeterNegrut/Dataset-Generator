#!/bin/bash
# Quick health check for the LoRA server

echo "🩺 LoRA Server Health Check"
echo "=========================="

if curl -s http://localhost:39515/health > /dev/null 2>&1; then
    echo "✅ Server is running!"
    echo ""
    echo "📊 Server Status:"
    curl -s http://localhost:39515/health | python3 -m json.tool
else
    echo "❌ Server is not running"
    echo ""
    echo "🚀 To start the server:"
    echo "   ./start_server.sh"
fi
