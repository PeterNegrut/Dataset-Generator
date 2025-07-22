#!/bin/bash
# 🔑 RunPod SSH Auto-Setup
# Automates SSH key setup for seamless connections

echo "🔑 RunPod SSH Auto-Setup"
echo "========================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Step 1: Generate SSH key if it doesn't exist
SSH_KEY_PATH="$HOME/.ssh/runpod_key"
if [ ! -f "$SSH_KEY_PATH" ]; then
    print_info "Generating new SSH key for RunPod..."
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "runpod-automation-$(date +%Y%m%d)"
    print_success "SSH key generated: $SSH_KEY_PATH"
else
    print_info "SSH key already exists: $SSH_KEY_PATH"
fi

# Step 2: Display public key for RunPod
echo ""
print_info "📋 Copy this PUBLIC KEY to your RunPod pod SSH settings:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "${SSH_KEY_PATH}.pub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 3: Create SSH config entry
SSH_CONFIG="$HOME/.ssh/config"
print_info "Setting up SSH config..."

# Read RunPod connection details
echo "Please enter your RunPod connection details:"
read -p "RunPod SSH Host (e.g., ssh.runpod.io): " RUNPOD_HOST
read -p "RunPod SSH Port (e.g., 12345): " RUNPOD_PORT
read -p "RunPod Pod ID (for alias): " POD_ID

# Create or update SSH config
if [ ! -f "$SSH_CONFIG" ]; then
    touch "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
fi

# Remove old runpod entries
sed -i '/# RunPod Auto-Config/,/# End RunPod Auto-Config/d' "$SSH_CONFIG"

# Add new entry
cat >> "$SSH_CONFIG" << EOF

# RunPod Auto-Config
Host runpod-${POD_ID}
    HostName ${RUNPOD_HOST}
    Port ${RUNPOD_PORT}
    User root
    IdentityFile ${SSH_KEY_PATH}
    ServerAliveInterval 60
    ServerAliveCountMax 10
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
# End RunPod Auto-Config
EOF

print_success "SSH config updated"

# Step 4: Create connection script
cat > "connect_runpod.sh" << EOF
#!/bin/bash
# 🚀 Auto-connect to RunPod

echo "🚀 Connecting to RunPod..."
ssh runpod-${POD_ID}
EOF

chmod +x "connect_runpod.sh"

# Step 5: Create sync script
cat > "sync_to_runpod.sh" << EOF
#!/bin/bash
# 📤 Sync local changes to RunPod

echo "📤 Syncing to RunPod..."
rsync -avz --progress --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
    ./ runpod-${POD_ID}:/root/dataset-generator/

echo "✅ Sync complete!"
EOF

chmod +x "sync_to_runpod.sh"

# Step 6: Create setup script for new pods
cat > "setup_new_runpod.sh" << EOF
#!/bin/bash
# 🏗️ Setup script for new RunPod instances

echo "🏗️ Setting up new RunPod instance..."

# Install git if not present
if ! command -v git &> /dev/null; then
    apt update && apt install -y git
fi

# Clone or sync repository
if [ ! -d "/root/dataset-generator" ]; then
    echo "📥 Cloning repository..."
    cd /root
    git clone <YOUR_REPO_URL> dataset-generator
    cd dataset-generator
else
    echo "📁 Repository exists, pulling latest..."
    cd /root/dataset-generator
    git pull
fi

# Run installation
echo "🚀 Running installation..."
chmod +x install_everything.sh
./install_everything.sh

echo "✅ RunPod setup complete!"
echo "🚀 Start server with: ./start_server.sh"
EOF

chmod +x "setup_new_runpod.sh"

echo ""
print_success "🎉 SSH automation setup complete!"
echo ""
echo "📋 What was created:"
echo "  • SSH key: $SSH_KEY_PATH"
echo "  • SSH config entry: runpod-${POD_ID}"
echo "  • ./connect_runpod.sh - Connect to RunPod"
echo "  • ./sync_to_runpod.sh - Sync files to RunPod"
echo "  • ./setup_new_runpod.sh - Setup new RunPod instances"
echo ""
echo "🔄 Next steps:"
echo "  1. Copy the public key above to RunPod SSH settings"
echo "  2. Test connection: ./connect_runpod.sh"
echo "  3. Sync files: ./sync_to_runpod.sh"
echo ""
print_warning "💡 Save the RunPod connection details for future use!"
EOF
