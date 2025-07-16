from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import threading
import time
import uuid
import json
import requests
from werkzeug.utils import secure_filename
import base64
import zipfile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import random
import numpy as np
from typing import List, Dict, Any, Tuple

app = Flask(__name__)
CORS(app)

# Configuration - THIS IS THE IMPORTANT LINE!
#RUNPOD_API_KEY = 'your_runpod_api_key_here'
RUNPOD_PYTORCH_POD_URL = 'http://localhost:39515'
#RUNPOD_ENDPOINT_ID = 'your_endpoint_id_here'
#RUNPOD_BASE_URL = 'https://api.runpod.ai/v2'

# Local configuration
UPLOAD_FOLDER = '/tmp/datasets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Storage for jobs and datasets
jobs = {}
datasets = {}

print("🚀 Simplified AI Dataset Generator")
print("=" * 50)
print(f"🔧 RunPod Server: {RUNPOD_PYTORCH_POD_URL}")  # ← This will now show the URL
print("📋 Features:")
print("   ✅ Simple LoRA training")
print("   ✅ Image generation") 
print("   ✅ SAM segmentation")
print("   ✅ Dataset packaging")
print("=" * 50)
print("🚀 Starting Simplified Backend...")
print("🌐 Server: http://localhost:5000")
print(f"🔗 RunPod: {RUNPOD_PYTORCH_POD_URL}")  # ← This will now show the URL

# ... rest of your backend code goes here ...
class EnhancedDatasetPipeline:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.job = jobs[job_id]
        
        # Optimized parameters for better quality
        self.target_resolution = 512
        self.augmentation_factor = 6  # Reduced from 8 to avoid overfitting
        self.max_training_images = 50  # Reduced for better quality
        
        # Better LoRA training parameters
        self.lora_config = {
            "rank": 16,  # Increased for better quality
            "alpha": 32,  # Important for stability
            "learning_rate": 5e-5,  # Lower for better stability
            "max_train_steps": None,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "mixed_precision": "fp16",
            "resolution": self.target_resolution,
            "train_text_encoder": True,  # Enable for better concept learning
            "use_8bit_adam": True,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 0,
            "save_precision": "fp16",
            "clip_skip": 2,
            "prior_loss_weight": 1.0,
            "seed": 42,
            "max_grad_norm": 1.0,
            "noise_offset": 0.1,  # Helps with darker images
        }
    
    def update_status(self, status: str, progress: int, message: str, **kwargs):
        """Enhanced status updates with detailed logging"""
        self.job.update({
            'status': status,
            'progress': progress,
            'message': message,
            'updated_at': time.time(),
            **kwargs
        })
        print(f"[{self.job_id}] {status}: {message} ({progress}%)")
    
    def run_complete_pipeline(self, files: List[Dict], concept_name: str, num_images: int) -> Dict[str, Any]:
        """Enhanced pipeline with better error handling and fallbacks"""
        try:
            # Phase 1: Better data preparation
            self.update_status('preparing', 5, 'Preparing training data with quality augmentation...')
            training_data = self._prepare_training_data(files, concept_name)
            
            if len(training_data['images']) == 0:
                raise Exception("No training images were processed successfully")
            
            # Phase 2: Smart captioning
            self.update_status('captioning', 15, 'Generating smart captions...')
            captioned_data = self._generate_smart_captions(training_data, concept_name)
            
            # Phase 3: LoRA training with fallbacks
            self.update_status('training', 25, f'Training LoRA model with {len(captioned_data["images"])} images...')
            lora_result = self._train_lora_with_fallbacks(captioned_data, concept_name)
            
            # Phase 4: Better image generation
            self.update_status('generating', 55, f'Generating {num_images} high-quality images...')
            generated_images = self._generate_quality_images(concept_name, num_images, lora_result)
            
            # Phase 5: SAM segmentation (optional)
            self.update_status('segmenting', 85, 'Running segmentation...')
            segmented_results = self._run_segmentation(generated_images)
            
            # Phase 6: Package results
            self.update_status('packaging', 95, 'Packaging dataset...')
            dataset_info = self._package_dataset(segmented_results, concept_name, training_data)
            
            self.update_status('complete', 100, 'Dataset creation complete!', result=dataset_info)
            return dataset_info
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            self.update_status('failed', 0, str(e), error=str(e))
            raise
    
    def _prepare_training_data(self, file_data_list: List[Dict], concept_name: str) -> Dict[str, Any]:
        """Better data preparation with quality-focused augmentation"""
        print(f"🔬 Preparing training data for {len(file_data_list)} input images")
        
        processed_images = []
        
        for i, file_data in enumerate(file_data_list):
            try:
                # Load and preprocess image
                image = Image.open(io.BytesIO(file_data['data']))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Smart resize maintaining quality
                image = self._quality_resize(image, self.target_resolution)
                
                # Add original image
                processed_images.append({
                    'image_data': self._image_to_base64(image),
                    'variant_type': 'original',
                    'filename': file_data['filename']
                })
                
                # Generate quality-focused variants
                variants = self._generate_quality_variants(image, min(self.augmentation_factor - 1, 5))
                processed_images.extend(variants)
                
                print(f"✅ Processed {file_data['filename']}: 1 original + {len(variants)} variants")
                
            except Exception as e:
                print(f"❌ Error processing {file_data.get('filename', f'file_{i}')}: {e}")
                continue
        
        # Ensure we don't have too many images (quality over quantity)
        if len(processed_images) > self.max_training_images:
            print(f"⚡ Limiting to {self.max_training_images} images for better quality")
            processed_images = processed_images[:self.max_training_images]
        
        return {
            'images': processed_images,
            'concept_name': concept_name,
            'total_count': len(processed_images)
        }
    
    def _quality_resize(self, image: Image.Image, target_size: int) -> Image.Image:
        """High-quality resizing with smart cropping"""
        w, h = image.size
        
        # If already square, just resize
        if w == h:
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # For non-square, crop to square from center
        size = min(w, h)
        left = (w - size) // 2
        top = (h - size) // 2
        image = image.crop((left, top, left + size, top + size))
        
        return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    def _generate_quality_variants(self, base_image: Image.Image, num_variants: int) -> List[Dict]:
        """Generate quality-focused variants avoiding artifacts"""
        variants = []
        
        # Conservative augmentations that preserve quality
        augmentations = [
            ('horizontal_flip', lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
            ('slight_rotate', lambda img: img.rotate(random.uniform(-8, 8), fillcolor=(255, 255, 255))),
            ('brightness', lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))),
            ('contrast', lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.95, 1.05))),
            ('slight_crop', lambda img: self._conservative_crop(img)),
        ]
        
        for i in range(min(num_variants, len(augmentations))):
            try:
                aug_name, aug_func = augmentations[i]
                variant = aug_func(base_image.copy())
                variants.append({
                    'image_data': self._image_to_base64(variant),
                    'variant_type': aug_name,
                    'filename': f'variant_{aug_name}'
                })
            except Exception as e:
                print(f"⚠️ Augmentation {aug_name} failed: {e}")
        
        return variants
    
    def _conservative_crop(self, image: Image.Image) -> Image.Image:
        """Conservative cropping that preserves main content"""
        w, h = image.size
        crop_factor = random.uniform(0.9, 0.95)  # Very conservative
        
        new_size = int(min(w, h) * crop_factor)
        left = (w - new_size) // 2
        top = (h - new_size) // 2
        
        cropped = image.crop((left, top, left + new_size, top + new_size))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    
    def _generate_smart_captions(self, training_data: Dict, concept_name: str) -> Dict[str, Any]:
        """Generate smart, varied captions"""
        images = training_data['images']
        
        # Better caption templates
        base_templates = [
            f"a photo of {concept_name}",
            f"an image of {concept_name}",
            f"a picture of {concept_name}",
            f"{concept_name}, detailed",
            f"{concept_name}, high quality",
            f"a clear photo of {concept_name}",
        ]
        
        captioned_images = []
        for i, img_data in enumerate(images):
            # Use different templates to add variety
            caption = base_templates[i % len(base_templates)]
            
            # Add quality modifiers occasionally
            if i % 3 == 0:
                caption += ", masterpiece, best quality"
            elif i % 3 == 1:
                caption += ", detailed, sharp focus"
            
            captioned_images.append({
                **img_data,
                'caption': caption
            })
        
        return {
            **training_data,
            'images': captioned_images
        }
    
    def _train_lora_with_fallbacks(self, training_data: Dict, concept_name: str) -> Dict[str, Any]:
        """Train LoRA with multiple fallback strategies"""
        num_images = len(training_data['images'])
        
        # Calculate better training steps
        self.lora_config['max_train_steps'] = min(max(num_images * 20, 500), 1500)
        
        # Try enhanced training first
        try:
            return self._try_enhanced_training(training_data, concept_name)
        except Exception as e:
            print(f"⚠️ Enhanced training failed: {e}")
            
        # Fallback to basic training
        try:
            return self._try_basic_training(training_data, concept_name)
        except Exception as e:
            print(f"⚠️ Basic training failed: {e}")
            
        # Final fallback - no LoRA
        return {
            'lora_path': None,
            'status': 'no_lora',
            'concept_name': concept_name,
            'message': 'Training failed, using base model only'
        }
    
    def _try_enhanced_training(self, training_data: Dict, concept_name: str) -> Dict[str, Any]:
        """Try enhanced LoRA training"""
        payload = {
            "concept_name": concept_name,
            "instance_prompt": f"a photo of {concept_name}",
            "class_prompt": "a photo",
            "training_images": [img['image_data'] for img in training_data['images']],
            "captions": [img['caption'] for img in training_data['images']],
            **self.lora_config
        }
        
        response = requests.post(
            f"{RUNPOD_PYTORCH_POD_URL}/train-lora-enhanced",
            json=payload,
            timeout=900
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'lora_path': result.get('lora_path'),
                'status': 'enhanced_complete',
                'concept_name': concept_name,
                'training_stats': result.get('training_stats', {})
            }
        else:
            raise Exception(f"Enhanced training failed: {response.status_code}")
    
    def _try_basic_training(self, training_data: Dict, concept_name: str) -> Dict[str, Any]:
        """Fallback to basic LoRA training"""
        payload = {
            "concept_name": concept_name,
            "instance_prompt": f"a photo of {concept_name}",
            "training_images": [img['image_data'] for img in training_data['images']],
            "max_train_steps": 800,
            "learning_rate": 1e-4,
            "resolution": 512,
            "rank": 8
        }
        
        response = requests.post(
            f"{RUNPOD_PYTORCH_POD_URL}/train-lora",
            json=payload,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'lora_path': result.get('lora_path'),
                'status': 'basic_complete',
                'concept_name': concept_name
            }
        else:
            raise Exception(f"Basic training failed: {response.status_code}")
    
    def _generate_quality_images(self, concept_name: str, num_images: int, lora_result: Dict) -> List[Dict]:
        """Generate high-quality images with better prompts"""
        generated_images = []
        
        # Better quality prompts
        quality_prompts = [
            f"a photo of {concept_name}, masterpiece, best quality, detailed",
            f"a portrait of {concept_name}, professional photography, high resolution",
            f"a clear image of {concept_name}, sharp focus, detailed",
            f"{concept_name}, high quality photo, professional lighting",
            f"a picture of {concept_name}, award winning photography, detailed",
            f"a detailed photo of {concept_name}, masterpiece, ultra detailed",
        ]
        
        for i in range(num_images):
            try:
                prompt = quality_prompts[i % len(quality_prompts)]
                
                # Add some variety
                if i % 2 == 0:
                    prompt += ", photorealistic"
                elif i % 3 == 0:
                    prompt += ", 8k uhd"
                
                progress = 55 + (i / num_images) * 25
                self.update_status('generating', progress, f'Generating image {i+1}/{num_images}...')
                
                generation_payload = {
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, cropped, worst quality, low quality, jpeg artifacts, duplicate, mutated, deformed, disfigured, poorly drawn, extra limbs, missing limbs, malformed limbs, too many fingers, long neck, mutation, bad proportions, cloned face, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers",
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512,
                    "scheduler": "DPMSolverMultistepScheduler",
                    "seed": random.randint(0, 2147483647),
                    "clip_skip": 2,
                }
                
                # Add LoRA if available
                if lora_result.get('lora_path'):
                    generation_payload.update({
                        "lora_path": lora_result['lora_path'],
                        "lora_strength": 0.8,  # Conservative strength
                        "concept_name": concept_name
                    })
                
                # Try enhanced generation first
                success = False
                for endpoint in ['/generate-enhanced', '/generate']:
                    try:
                        response = requests.post(
                            f"{RUNPOD_PYTORCH_POD_URL}{endpoint}",
                            json=generation_payload,
                            timeout=90
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get('status') == 'success' and result.get('image'):
                                generated_images.append({
                                    'id': i + 1,
                                    'prompt': prompt,
                                    'image_data': result['image'],
                                    'lora_used': bool(lora_result.get('lora_path')),
                                    'endpoint_used': endpoint
                                })
                                success = True
                                break
                    except Exception as e:
                        print(f"Generation attempt failed with {endpoint}: {e}")
                        continue
                
                if not success:
                    print(f"❌ Failed to generate image {i+1}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error generating image {i+1}: {e}")
        
        print(f"✅ Generated {len(generated_images)}/{num_images} images")
        return generated_images
    
    def _run_segmentation(self, generated_images: List[Dict]) -> List[Dict]:
        """Run SAM segmentation with fallback"""
        segmented_results = []
        
        for i, img_data in enumerate(generated_images):
            try:
                progress = 85 + (i / len(generated_images)) * 10
                self.update_status('segmenting', progress, f'Segmenting image {i+1}/{len(generated_images)}...')
                
                # Try SAM segmentation
                try:
                    sam_payload = {
                        "image": img_data['image_data'],
                        "point_coords": [[256, 256]],
                        "point_labels": [1]
                    }
                    
                    response = requests.post(
                        f"{RUNPOD_PYTORCH_POD_URL}/segment",
                        json=sam_payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        sam_result = response.json()
                        segmentation = {
                            'masks': sam_result.get('masks', []),
                            'scores': sam_result.get('scores', []),
                            'success': True
                        }
                    else:
                        segmentation = {'success': False, 'error': f'HTTP {response.status_code}'}
                        
                except Exception as e:
                    segmentation = {'success': False, 'error': str(e)}
                
                segmented_results.append({
                    **img_data,
                    'segmentation': segmentation
                })
                
            except Exception as e:
                print(f"❌ Error processing image {i+1}: {e}")
                segmented_results.append({
                    **img_data,
                    'segmentation': {'success': False, 'error': str(e)}
                })
        
        return segmented_results
    
    def _package_dataset(self, results: List[Dict], concept_name: str, training_data: Dict) -> Dict[str, Any]:
        """Package dataset with better organization"""
        dataset_id = f"dataset_{self.job_id}"
        dataset_dir = os.path.join(UPLOAD_FOLDER, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create organized structure
        images_dir = os.path.join(dataset_dir, 'images')
        masks_dir = os.path.join(dataset_dir, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        successful_images = 0
        successful_masks = 0
        preview_images = []
        
        # Process results
        for i, result in enumerate(results):
            try:
                # Save image
                if result.get('image_data'):
                    image_filename = f"image_{i+1:03d}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    
                    image_bytes = base64.b64decode(result['image_data'])
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    successful_images += 1
                    
                    # Add to preview (first 6 images)
                    if len(preview_images) < 6:
                        preview_images.append({
                            'filename': image_filename,
                            'prompt': result.get('prompt', ''),
                            'image_url': f"data:image/png;base64,{result['image_data']}"
                        })
                
                # Save mask if available
                if result.get('segmentation', {}).get('success'):
                    mask_filename = f"mask_{i+1:03d}.json"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    
                    with open(mask_path, 'w') as f:
                        json.dump(result['segmentation'], f, indent=2)
                    
                    successful_masks += 1
                
            except Exception as e:
                print(f"Error saving result {i+1}: {e}")
        
        # Create dataset info
        dataset_info = {
            'dataset_id': dataset_id,
            'concept_name': concept_name,
            'creation_timestamp': time.time(),
            'total_images': successful_images,
            'segmented_images': successful_masks,
            'training_images_used': len(training_data.get('images', [])),
            'preview_images': preview_images,
            'download_url': f'/api/download-dataset/{dataset_id}'
        }
        
        # Save metadata
        metadata_path = os.path.join(dataset_dir, 'dataset_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        datasets[dataset_id] = dataset_info
        
        print(f"📦 Dataset packaged: {successful_images} images, {successful_masks} masks")
        return dataset_info
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===== FRONTEND ROUTES =====

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    try:
        frontend_path = '/root/dataset-generator-1/frontend/index.html'
        if os.path.exists(frontend_path):
            return send_file(frontend_path)
        else:
            return f"""
            <h1>AI Dataset Generator Backend</h1>
            <p>Frontend not found at: {frontend_path}</p>
            <p>API is running correctly!</p>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/api/health">API Health Check</a></li>
                <li><a href="/api/test-runpod">Test RunPod Connection</a></li>
            </ul>
            """, 404
    except Exception as e:
        return f"Error serving frontend: {e}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend directory"""
    try:
        frontend_dir = '/root/dataset-generator-1/frontend'
        return send_from_directory(frontend_dir, filename)
    except Exception as e:
        return f"Static file error: {e}", 404

# ===== API ROUTES =====

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check"""
    try:
        # Test RunPod connection
        try:
            response = requests.get(f"{RUNPOD_PYTORCH_POD_URL}/health", timeout=5)
            runpod_status = response.status_code == 200
        except Exception as e:
            runpod_status = False

        return jsonify({
            "status": "healthy",
            "runpod_connected": runpod_status,
            "runpod_url": RUNPOD_PYTORCH_POD_URL,
            "endpoints": [
                "/health",
                "/api/health",
                "/api/generate-dataset",
                "/api/job-status/<job_id>",
                "/api/download-dataset/<dataset_id>",
                "/api/test-runpod"
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health_check():
    """API health check endpoint that frontend expects"""
    try:
        # Test RunPod connection
        try:
            response = requests.get(f"{RUNPOD_PYTORCH_POD_URL}/health", timeout=5)
            runpod_status = response.status_code == 200
            runpod_data = response.json() if runpod_status else {}
        except Exception as e:
            runpod_status = False
            runpod_data = {"error": str(e)}

        return jsonify({
            "status": "healthy",
            "backend": "Flask AI Dataset Generator",
            "version": "1.0",
            "runpod_connected": runpod_status,
            "runpod_url": RUNPOD_PYTORCH_POD_URL,
            "runpod_info": runpod_data,
            "endpoints": [
                "/api/health",
                "/api/generate-dataset", 
                "/api/job-status/<job_id>",
                "/api/download-dataset/<dataset_id>",
                "/api/test-runpod"
            ]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "backend": "Flask AI Dataset Generator", 
            "error": str(e)
        }), 500

@app.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    """Generate dataset with improved error handling"""
    try:
        # Get form data
        files = request.files.getlist('images')
        concept_name = request.form.get('concept_name')
        num_images = int(request.form.get('num_images', 10))
        
        # Validation
        if not concept_name or not concept_name.strip():
            return jsonify({'error': 'Concept name is required'}), 400
        
        if len(files) < 3:
            return jsonify({'error': 'At least 3 images required'}), 400
        
        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 images allowed'}), 400
        
        # Process files
        file_data_list = []
        for file in files:
            if not file or not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file: {file.filename}'}), 400
            
            file_data = file.read()
            if len(file_data) == 0:
                return jsonify({'error': f'Empty file: {file.filename}'}), 400
            
            file_data_list.append({
                'filename': file.filename,
                'data': file_data
            })
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': 'Starting enhanced pipeline...',
            'created_at': time.time(),
            'concept_name': concept_name,
            'num_images': num_images
        }
        
        # Start pipeline
        pipeline = EnhancedDatasetPipeline(job_id)
        thread = threading.Thread(
            target=pipeline.run_complete_pipeline,
            args=(file_data_list, concept_name, num_images)
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Pipeline started successfully'
        })
        
    except Exception as e:
        print(f"Error starting pipeline: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-status/<job_id>')
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job.get('progress', 0),
        'message': job.get('message', ''),
        'error': job.get('error', ''),
        'result': job.get('result', {}),
        'created_at': job['created_at'],
        'updated_at': job.get('updated_at', job['created_at'])
    })

@app.route('/api/download-dataset/<dataset_id>')
def download_dataset(dataset_id):
    if dataset_id not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        dataset_dir = os.path.join(UPLOAD_FOLDER, dataset_id)
        if not os.path.exists(dataset_dir):
            return jsonify({'error': 'Dataset files not found'}), 404
        
        # Create ZIP file
        zip_path = os.path.join(UPLOAD_FOLDER, f'{dataset_id}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dataset_dir)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f'{dataset_id}.zip')
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/test-runpod')
def test_runpod():
    """Test RunPod connectivity"""
    try:
        response = requests.get(f"{RUNPOD_PYTORCH_POD_URL}/health", timeout=10)
        
        if response.status_code == 200:
            return jsonify({
                'status': 'connected',
                'message': 'RunPod connection successful',
                'url': RUNPOD_PYTORCH_POD_URL
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'RunPod returned status {response.status_code}',
                'url': RUNPOD_PYTORCH_POD_URL
            })
            
    except Exception as e:
        return jsonify({
            'status': 'failed',
            'message': f'Connection failed: {str(e)}',
            'url': RUNPOD_PYTORCH_POD_URL
        })

# ===== CORS SUPPORT =====

@app.after_request
def after_request(response):
    """Add CORS headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle CORS preflight requests"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("🚀 Starting Enhanced AI Dataset Generator Backend...")
    print("🌐 Server starting on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)