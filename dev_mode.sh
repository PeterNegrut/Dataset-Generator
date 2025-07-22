#!/bin/bash
# 🔧 Development Mode - Auto-restart on changes
# Use this for active development

echo "🔧 Development Mode - Auto-restart"
echo "=================================="

cd /root/dataset-generator-3 || exit 1

# Install inotify for file watching
apt update && apt install -y inotify-tools

echo "👀 Watching for changes in backend/..."
echo "🚀 Starting server..."

# Start server
./start_server.sh &
SERVER_PID=$!

# Watch for changes and auto-restart
while inotifywait -r -e modify backend/; do
    echo "🔄 Change detected! Restarting server..."
    kill $SERVER_PID 2>/dev/null
    sleep 1
    ./start_server.sh &
    SERVER_PID=$!
    echo "✅ Server restarted!"
done
