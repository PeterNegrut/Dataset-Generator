from flask import Flask, request, jsonify, send_file
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
from PIL import Image
import io
import random
import numpy as np
from typing import List, Dict, Any

app = Flask(__name__)
CORS(app)

# Configuration - UPDATE THESE WITH YOUR RUNPOD VALUES
RUNPOD_API_KEY = ''
RUNPOD_PYTORCH_POD_URL = ''
RUNPOD_ENDPOINT_ID = ''
RUNPOD_BASE_URL = ''  # Fixed typo and added proper URL

# Local configuration
UPLOAD_FOLDER = '/tmp/datasets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Storage for jobs and datasets
jobs = {}
datasets = {}

print("🚀 Simplified AI Dataset Generator")
print("=" * 50)
print(f"🔧 RunPod Server: {RUNPOD_PYTORCH_POD_URL}")
print("📋 Features:")
print("   ✅ Simple LoRA training")
print("   ✅ Image generation")
print("   ✅ SAM segmentation")
print("   ✅ Dataset packaging")

class SimpleDatasetPipeline:
    """Simplified pipeline that works with your RunPod server"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.job = jobs[job_id]
    
    def update_status(self, status: str, progress: int, message: str, **kwargs):
        """Update job status"""
        self.job.update({
            'status': status,
            'progress': progress,
            'message': message,
            'updated_at': time.time(),
            **kwargs
        })
        print(f"[{self.job_id}] {status}: {message} ({progress}%)")
    
    def run_pipeline(self, files: List[Dict], concept_name: str, num_images: int) -> Dict[str, Any]:
        """Simplified pipeline"""
        try:
            # Phase 1: Prepare images
            self.update_status('preparing', 10, 'Preparing training images...')
            training_images = self._prepare_images(files)
            
            if not training_images:
                raise Exception("No valid training images processed")
            
            # Phase 2: Train LoRA
            self.update_status('training', 25, f'Training LoRA with {len(training_images)} images...')
            lora_result = self._train_lora(training_images, concept_name)
            
            # Phase 3: Generate images
            self.update_status('generating', 60, f'Generating {num_images} images...')
            generated_images = self._generate_images(concept_name, num_images)
            
            # Phase 4: SAM segmentation
            self.update_status('segmenting', 85, 'Running segmentation...')
            segmented_results = self._segment_images(generated_images)
            
            # Phase 5: Package dataset
            self.update_status('packaging', 95, 'Packaging dataset...')
            dataset_info = self._package_dataset(segmented_results, concept_name, len(training_images))
            
            self.update_status('complete', 100, 'Dataset creation complete!', result=dataset_info)
            return dataset_info
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            self.update_status('failed', 0, str(e), error=str(e))
            raise
    
    def _prepare_images(self, file_data_list: List[Dict]) -> List[str]:
        """Convert images to base64"""
        training_images = []
        
        for file_data in file_data_list:
            try:
                # Load and resize image
                image = Image.open(io.BytesIO(file_data['data']))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize to 512x512
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                training_images.append(f"data:image/png;base64,{img_str}")
                
                print(f"✅ Processed {file_data['filename']}")
                
            except Exception as e:
                print(f"❌ Error processing {file_data.get('filename', 'unknown')}: {e}")
                continue
        
        return training_images
    
    def _train_lora(self, training_images: List[str], concept_name: str) -> Dict[str, Any]:
        """Train LoRA using your RunPod server"""
        try:
            payload = {
                "images": training_images,
                "domain_prefix": f"a photo of {concept_name}",
                "rank": 8,
                "learning_rate": 1e-4,
                "num_epochs": 10
            }
            
            print(f"🚀 Training LoRA: {RUNPOD_PYTORCH_POD_URL}/train-lora-enhanced")
            
            response = requests.post(
                f"{RUNPOD_PYTORCH_POD_URL}/train-lora-enhanced",
                json=payload,
                timeout=600  # 10 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ LoRA training completed: {result}")
                return {
                    'success': True,
                    'lora_path': result.get('lora_path'),
                    'training_images': result.get('num_training_images', len(training_images))
                }
            else:
                print(f"⚠️ LoRA training failed: {response.status_code} - {response.text}")
                return {'success': False, 'error': response.text}
                
        except Exception as e:
            print(f"❌ LoRA training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_images(self, concept_name: str, num_images: int) -> List[Dict]:
        """Generate images using your RunPod server"""
        generated_images = []
        
        # Simple prompts
        prompts = [
            f"a photo of {concept_name}",
            f"a high quality image of {concept_name}",
            f"a detailed photo of {concept_name}",
            f"a professional photo of {concept_name}",
            f"a clear image of {concept_name}"
        ]
        
        # Generate images in batches
        batch_size = 4
        for i in range(0, num_images, batch_size):
            try:
                batch_prompts = []
                for j in range(batch_size):
                    if i + j < num_images:
                        prompt = prompts[(i + j) % len(prompts)]
                        batch_prompts.append(prompt)
                
                if not batch_prompts:
                    break
                
                progress = 60 + (i / num_images) * 20
                self.update_status('generating', progress, f'Generating images {i+1}-{i+len(batch_prompts)}/{num_images}')
                
                payload = {
                    "prompts": batch_prompts,
                    "num_images": 1,
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
                
                response = requests.post(
                    f"{RUNPOD_PYTORCH_POD_URL}/generate-enhanced",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success') and result.get('images'):
                        for idx, img_data in enumerate(result['images']):
                            generated_images.append({
                                'id': i + idx + 1,
                                'prompt': batch_prompts[idx % len(batch_prompts)],
                                'image_data': img_data
                            })
                        print(f"✅ Generated batch {i//batch_size + 1}")
                    else:
                        print(f"❌ Generation failed: {result}")
                else:
                    print(f"❌ Generation request failed: {response.status_code}")
                
                time.sleep(1)  # Brief pause between batches
                
            except Exception as e:
                print(f"❌ Error generating batch {i//batch_size + 1}: {e}")
        
        print(f"✅ Generated {len(generated_images)}/{num_images} images")
        return generated_images
    
    def _segment_images(self, generated_images: List[Dict]) -> List[Dict]:
        """Run SAM segmentation"""
        segmented_results = []
        
        for i, img_data in enumerate(generated_images):
            try:
                progress = 85 + (i / len(generated_images)) * 10
                self.update_status('segmenting', progress, f'Segmenting image {i+1}/{len(generated_images)}')
                
                # Extract base64 data (remove data:image/png;base64, prefix if present)
                image_data = img_data['image_data']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                payload = {
                    "image": f"data:image/png;base64,{image_data}",
                    "point_coords": [[256, 256]],
                    "point_labels": [1]
                }
                
                try:
                    response = requests.post(
                        f"{RUNPOD_PYTORCH_POD_URL}/segment",
                        json=payload,
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
                        segmentation = {'success': False, 'error': f"HTTP {response.status_code}"}
                        
                except Exception as e:
                    segmentation = {'success': False, 'error': str(e)}
                
                segmented_results.append({
                    **img_data,
                    'segmentation': segmentation
                })
                
            except Exception as e:
                print(f"❌ Segmentation error for image {i+1}: {e}")
                segmented_results.append({
                    **img_data,
                    'segmentation': {'success': False, 'error': str(e)}
                })
        
        return segmented_results
    
    def _package_dataset(self, segmented_results: List[Dict], concept_name: str, num_training_images: int) -> Dict[str, Any]:
        """Package results"""
        dataset_id = f"dataset_{self.job_id}"
        dataset_dir = os.path.join(UPLOAD_FOLDER, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create directories
        images_dir = os.path.join(dataset_dir, 'images')
        masks_dir = os.path.join(dataset_dir, 'masks')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        # Dataset info
        dataset_info = {
            'dataset_id': dataset_id,
            'concept_name': concept_name,
            'total_images': len(segmented_results),
            'segmented_images': len([r for r in segmented_results if r.get('segmentation', {}).get('success')]),
            'training_images_used': num_training_images,
            'creation_time': time.time(),
            'download_url': f'/api/download-dataset/{dataset_id}',
            'preview_images': []
        }
        
        # Save images and create previews
        for i, result in enumerate(segmented_results):
            try:
                # Save image
                image_filename = f"generated_{i+1:03d}.png"
                image_path = os.path.join(images_dir, image_filename)
                
                image_data = result['image_data']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                # Add to preview (first 4 images)
                if i < 4:
                    dataset_info['preview_images'].append({
                        'filename': image_filename,
                        'prompt': result.get('prompt', ''),
                        'image_url': f"data:image/png;base64,{image_data}"
                    })
                
                # Save mask if available
                if result.get('segmentation', {}).get('success'):
                    mask_filename = f"mask_{i+1:03d}.json"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    with open(mask_path, 'w') as f:
                        json.dump(result['segmentation'], f, indent=2)
                
            except Exception as e:
                print(f"Error saving result {i+1}: {e}")
        
        # Save dataset info
        info_path = os.path.join(dataset_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        datasets[dataset_id] = dataset_info
        
        print(f"📦 Dataset packaged: {dataset_id}")
        print(f"   Images: {dataset_info['total_images']}")
        print(f"   Segmented: {dataset_info['segmented_images']}")
        
        return dataset_info


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'backend': 'Simplified AI Dataset Generator',
        'runpod_url': RUNPOD_PYTORCH_POD_URL,
        'features': [
            'LoRA Training',
            'Image Generation', 
            'SAM Segmentation',
            'Dataset Packaging'
        ]
    })

@app.route('/api/generate-dataset', methods=['POST'])
def generate_dataset():
    """Generate dataset using simplified pipeline"""
    print("🚀 DATASET GENERATION STARTED!")
    
    try:
        # Get form data
        files = request.files.getlist('images')
        concept_name = request.form.get('concept_name')
        num_images = int(request.form.get('num_images', 10))
        
        print(f"📊 Input:")
        print(f"   Files: {len(files)}")
        print(f"   Concept: {concept_name}")
        print(f"   Target images: {num_images}")
        
        # Validation
        if not concept_name or not concept_name.strip():
            return jsonify({'error': 'Concept name is required'}), 400
        
        if len(files) < 3:
            return jsonify({'error': 'At least 3 images required'}), 400
        
        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 images allowed'}), 400
        
        if num_images < 5 or num_images > 30:
            return jsonify({'error': 'Number of images must be between 5 and 30'}), 400
        
        # Read files
        file_data_list = []
        for file in files:
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
            
            try:
                file.seek(0)
                file_data = file.read()
                
                if len(file_data) == 0:
                    return jsonify({'error': f'Empty file: {file.filename}'}), 400
                
                if len(file_data) > MAX_FILE_SIZE:
                    return jsonify({'error': f'File too large: {file.filename}'}), 400
                
                file_data_list.append({
                    'filename': file.filename,
                    'data': file_data
                })
                
            except Exception as e:
                return jsonify({'error': f'Failed to read {file.filename}: {str(e)}'}), 400
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': 'Starting pipeline...',
            'created_at': time.time(),
            'concept_name': concept_name,
            'num_images': num_images
        }
        
        # Start pipeline
        pipeline = SimpleDatasetPipeline(job_id)
        thread = threading.Thread(
            target=pipeline.run_pipeline,
            args=(file_data_list, concept_name, num_images)
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Pipeline started successfully'
        })
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-status/<job_id>')
def job_status(job_id):
    """Get job status"""
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
    """Download dataset as ZIP"""
    if dataset_id not in datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        dataset_dir = os.path.join(UPLOAD_FOLDER, dataset_id)
        if not os.path.exists(dataset_dir):
            return jsonify({'error': 'Dataset files not found'}), 404
        
        # Create ZIP
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
            result = response.json()
            return jsonify({
                'status': 'connected',
                'response': result,
                'url': RUNPOD_PYTORCH_POD_URL
            })
        else:
            return jsonify({
                'status': 'error',
                'http_status': response.status_code,
                'url': RUNPOD_PYTORCH_POD_URL
            })
            
    except Exception as e:
        return jsonify({
            'status': 'failed',
            'error': str(e),
            'url': RUNPOD_PYTORCH_POD_URL
        })

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("=" * 50)
    print("🚀 Starting Simplified Backend...")
    print(f"🌐 Server: http://localhost:5000")
    print(f"🔗 RunPod: {RUNPOD_PYTORCH_POD_URL}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)