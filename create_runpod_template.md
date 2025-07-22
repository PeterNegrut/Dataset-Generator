# 🚀 Create RunPod Template for Instant Setup

## Steps to Create Template:

1. **Run full setup once:**
   ```bash
   curl -sSL https://raw.githubusercontent.com/peterwillcocks/dataset-generator-3/main/runpod_quickstart.sh | bash
   ```

2. **In RunPod web interface:**
   - Go to "My Pods" 
   - Click your running pod
   - Click "Save as Template"
   - Name: "LoRA-Server-Ready"
   - Description: "Pre-configured LoRA training server"

3. **Future workflow (30 seconds):**
   - Start pod from "LoRA-Server-Ready" template
   - Open terminal
   - Run: `cd /root/dataset-generator-3 && git pull && ./start_server.sh`
   - Server is ready!

## Benefits:
- ✅ No dependency installation (already done)
- ✅ No environment setup (already done) 
- ✅ Just pull latest code and start
- ✅ 30 seconds instead of 10 minutes
