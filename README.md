# 🚀 Expert-Fixed LoRA Server - RunPod Ready

**One-command installation for RTX A5000 (24GB VRAM) optimized LoRA training**

## ⚡ Quick Start (2 Commands)

```bash
# 1. Install everything (run once)
./install_everything.sh

# 2. Start the server
./start_server.sh
```

That's it! 🎉

## 📡 Server Endpoints

Once running on `http://localhost:39515`:

- **`GET /health`** - System status and GPU info
- **`POST /train-lora-expert`** - Train LoRA with domain awareness (X-ray/general)
- **`POST /generate-expert`** - Generate images with auto-trigger detection
- **`POST /validate-lora`** - Validate LoRA effectiveness

## 🎯 Key Features

- ✅ **RTX A5000 Optimized** - 24GB VRAM efficiently used
- ✅ **Domain Awareness** - Specialized for X-ray vs general images
- ✅ **Auto-Trigger Detection** - Automatically injects trigger tokens
- ✅ **Expert Learning Rates** - 5e-5 for X-ray, 1e-4 for general
- ✅ **Memory Optimized** - XFormers, gradient checkpointing
- ✅ **Aspect Preserving** - Medical images aren't cropped
- ✅ **Adaptive Rank** - 8/16/32 based on dataset size

## 🧪 Training Example

```python
import requests
import base64

# Prepare your images as base64
images = ["<base64_image1>", "<base64_image2>", ...]

# Train X-ray LoRA
response = requests.post('http://localhost:39515/train-lora-expert', json={
    "concept_name": "chest_xray",
    "domain": "xray",  # or "general"
    "training_intensity": "medium",  # "fast", "medium", "thorough"
    "training_images": images
})

lora_path = response.json()["lora_path"]
trigger_token = response.json()["trigger_token"]

# Generate with trained LoRA
response = requests.post('http://localhost:39515/generate-expert', json={
    "prompt": "a medical X-ray showing clear lungs",
    "lora_path": lora_path,
    "num_inference_steps": 30
})

generated_image = response.json()["image"]  # base64
```

## 🔧 Troubleshooting

**Dependencies missing?**
```bash
./install_everything.sh
```

**Server won't start?**
```bash
# Check if all imports work
python3 -c "import torch, diffusers, peft, flask; print('✅ All good')"
```

**GPU not detected?**
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📊 Performance (RTX A5000)

- **Training**: ~15-30 minutes for 20 images
- **Memory Usage**: ~18-22GB VRAM during training
- **Generation**: ~2-3 seconds per 512x512 image
- **Batch Size**: 2 (optimal for 24GB VRAM)

## 🗂️ Workspace Structure

```
dataset-generator/
├── install_everything.sh    # 🚀 One-command installation
├── start_server.sh         # ▶️  Start the server
├── backend/
│   ├── runpod_server.py    # 🖥️  Main server code
│   ├── requirements.txt    # 📦 Python dependencies
│   └── .env               # ⚙️  Environment settings
└── README.md              # 📖 This file
```

## 🔄 Updates

To update your installation:
```bash
./install_everything.sh  # Re-run the installer
```

---

**Ready to start training LoRAs? Run `./install_everything.sh` now! 🚀**
