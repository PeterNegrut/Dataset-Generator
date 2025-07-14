// RunPod API Integration
// frontend/runpod-api.js

class RunPodAPI {
    constructor() {
        this.apiKey = RUNPOD_API_KEY;
        this.generationEndpoint = RUNPOD_ENDPOINT_ID;
        this.trainingEndpoint = RUNPOD_TRAINING_ENDPOINT_ID;
        this.baseUrl = 'https://api.runpod.ai/v2';
    }

    // Test connection to RunPod
    // Test connection to Python Backend (instead of RunPod directly)
    async testConnection() {
        try {
            console.log('🧪 Testing Python backend connection...');
            
            const response = await fetch('http://localhost:5000/api/health');
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Backend test successful:', result);
                return { success: true, message: result.message };
            } else {
                throw new Error(`Backend returned status ${response.status}`);
            }
        } catch (error) {
            console.error('❌ Backend test failed:', error);
            return { success: false, error: error.message };
        }
    }

    // Generate a single image
    async generateImage(prompt, useLoRA = false, loraUrl = null) {
        const payload = {
            input: {
                prompt: prompt + ", masterpiece, best quality, highly detailed, professional photography, 8k uhd",
                negative_prompt: "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature, amateur, phone camera, grainy, out of focus",
                num_inference_steps: 30,
                guidance_scale: 7.5,
                width: 768,
                height: 768,
                seed: Math.floor(Math.random() * 1000000)
            }
        };

        // Add LoRA if specified
        if (useLoRA && loraUrl) {
            payload.input.lora_url = loraUrl;
            payload.input.lora_scale = 0.8;
        }

        try {
            addDebugLog(`🎨 API CALL: Generating image for prompt: "${prompt.substring(0, 50)}..."`, 'info');
            
            const response = await fetch(`${this.baseUrl}/${this.generationEndpoint}/runsync`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            addDebugLog(`📡 API RESPONSE: Status ${response.status}`, 'info');

            if (!response.ok) {
                const errorText = await response.text();
                addDebugLog(`❌ API ERROR: HTTP ${response.status} - ${errorText}`, 'error');
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }

            const result = await response.json();
            addDebugLog(`📊 API RESULT: Status = ${result.status}`, 'info');
            
            if (result.status === 'COMPLETED') {
                const imageUrl = this.extractImageUrl(result);
                if (imageUrl) {
                    addDebugLog(`✅ SUCCESS: Generated image (${useLoRA ? 'with LoRA' : 'base model'})`, 'success');
                    return imageUrl;
                } else {
                    throw new Error('No image URL found in response');
                }
            } else {
                addDebugLog(`❌ API FAILED: ${result.error || 'Unknown error'}`, 'error');
                throw new Error(`Generation failed: ${result.error || 'Unknown error'}`);
            }
        } catch (error) {
            addDebugLog(`❌ API ERROR: ${error.message}`, 'error');
            throw error;
        }
    }

    // Upload training images to RunPod
    async uploadTrainingImages(files) {
        addDebugLog(`📤 Starting upload of ${files.length} training images`, 'info');
        const uploadedUrls = [];
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            addDebugLog(`📤 Uploading ${file.name} (${i + 1}/${files.length})`, 'info');
            
            try {
                // Convert file to base64
                const base64Data = await this.fileToBase64(file);
                
                // Upload payload
                const uploadPayload = {
                    input: {
                        action: "upload_image",
                        filename: file.name,
                        data: base64Data,
                        content_type: file.type
                    }
                };
                
                const response = await fetch(`${this.baseUrl}/${this.trainingEndpoint}/runsync`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(uploadPayload)
                });
                
                if (!response.ok) {
                    throw new Error(`Upload failed: ${response.status}`);
                }
                
                const result = await response.json();
                
                if (result.status === 'COMPLETED' && result.output?.url) {
                    uploadedUrls.push(result.output.url);
                    addDebugLog(`✅ Uploaded: ${file.name} -> ${result.output.url}`, 'success');
                } else {
                    // Fallback: create a placeholder URL for now
                    const placeholderUrl = `https://training-storage.runpod.io/${Date.now()}_${file.name}`;
                    uploadedUrls.push(placeholderUrl);
                    addDebugLog(`⚠️ Upload placeholder for ${file.name}`, 'warn');
                }
                
            } catch (error) {
                addDebugLog(`❌ Failed to upload ${file.name}: ${error.message}`, 'error');
                // Continue with other files
            }
        }
        
        addDebugLog(`📤 Upload complete: ${uploadedUrls.length}/${files.length} files uploaded`, 'info');
        return uploadedUrls;
    }

    // Start LoRA training
    async startLoRATraining(imageUrls, conceptName) {
        addDebugLog(`🎯 Starting LoRA training for concept: "${conceptName}"`, 'info');
        
        const trainingPayload = {
            input: {
                action: "train_lora",
                instance_data_urls: imageUrls,
                instance_prompt: `a photo of ${conceptName}`,
                class_prompt: `a photo of animal`,
                concept_name: conceptName.replace(/\s+/g, '_'),
                
                // Training parameters
                resolution: APP_CONFIG.defaultResolution,
                train_batch_size: 1,
                learning_rate: APP_CONFIG.defaultLearningRate,
                max_train_steps: APP_CONFIG.defaultTrainingSteps,
                checkpointing_steps: 100,
                validation_steps: 100,
                num_class_images: 100,
                
                // Optimization settings
                mixed_precision: "fp16",
                gradient_accumulation_steps: 4,
                use_8bit_adam: true,
                enable_xformers_memory_efficient_attention: true,
                
                // Output settings
                output_name: `lora_${conceptName.replace(/\s+/g, '_')}_${Date.now()}`,
                save_sample_images: true
            }
        };
        
        addDebugLog(`🎯 Training payload: ${JSON.stringify(trainingPayload, null, 2)}`, 'debug');
        
        try {
            const response = await fetch(`${this.baseUrl}/${this.trainingEndpoint}/run`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(trainingPayload)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                addDebugLog(`❌ Training start failed: ${response.status} - ${errorText}`, 'error');
                throw new Error(`Training start failed: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            addDebugLog(`✅ Training job started: ${result.id}`, 'success');
            
            return {
                jobId: result.id,
                status: result.status,
                estimatedDuration: APP_CONFIG.estimatedTrainingMinutes
            };
            
        } catch (error) {
            addDebugLog(`❌ Training start error: ${error.message}`, 'error');
            throw error;
        }
    }

    // Monitor training progress
    async monitorTraining(jobId, progressCallback) {
        addDebugLog(`📊 Starting training monitoring for job: ${jobId}`, 'info');
        
        const startTime = Date.now();
        let attempts = 0;
        const maxAttempts = 120; // 30 minutes max
        const checkInterval = 15000; // 15 seconds
        
        while (attempts < maxAttempts) {
            try {
                const response = await fetch(`${this.baseUrl}/${this.trainingEndpoint}/status/${jobId}`, {
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                
                const result = await response.json();
                const elapsed = Math.round((Date.now() - startTime) / 1000 / 60);
                
                addDebugLog(`📊 Training status: ${result.status} (${elapsed} min elapsed)`, 'info');
                
                // Calculate progress
                let progress = 20; // Start at 20%
                if (result.current_step && result.total_steps) {
                    progress = 20 + Math.round((result.current_step / result.total_steps) * 70);
                } else {
                    // Estimate based on time
                    progress = Math.min(20 + Math.round((elapsed / APP_CONFIG.estimatedTrainingMinutes) * 70), 95);
                }
                
                // Call progress callback
                if (progressCallback) {
                    progressCallback({
                        status: result.status,
                        progress: progress,
                        elapsed: elapsed,
                        currentStep: result.current_step || 0,
                        totalSteps: result.total_steps || APP_CONFIG.defaultTrainingSteps,
                        logs: result.logs || []
                    });
                }
                
                // Check completion status
                if (result.status === 'COMPLETED') {
                    addDebugLog(`✅ Training completed successfully! Duration: ${elapsed} minutes`, 'success');
                    return {
                        success: true,
                        loraUrl: result.output?.lora_url || result.output?.model_url,
                        duration: elapsed,
                        cost: (elapsed * APP_CONFIG.costPerTrainingMinute).toFixed(2),
                        sampleImages: result.output?.sample_images || []
                    };
                } else if (result.status === 'FAILED') {
                    addDebugLog(`❌ Training failed: ${result.error}`, 'error');
                    throw new Error(`Training failed: ${result.error || 'Unknown error'}`);
                }
                
                // Wait before next check
                await this.delay(checkInterval);
                attempts++;
                
            } catch (error) {
                addDebugLog(`❌ Training monitor error: ${error.message}`, 'error');
                attempts++;
                await this.delay(checkInterval * 2); // Longer delay on error
            }
        }
        
        addDebugLog(`❌ Training monitoring timed out after ${maxAttempts * checkInterval / 1000 / 60} minutes`, 'error');
        throw new Error('Training monitoring timed out');
    }

    // Generate dataset with trained LoRA
    async generateDatasetWithLoRA(loraUrl, conceptName, numImages, progressCallback) {
        addDebugLog(`🎨 Starting dataset generation with LoRA for "${conceptName}"`, 'info');
        
        const results = [];
        let successCount = 0;
        const startTime = Date.now();
        
        // Prompts specifically designed for LoRA-trained concepts
        const variations = [
            `a professional portrait photo of ${conceptName}, studio lighting, high quality`,
            `${conceptName} sitting peacefully in a garden, natural daylight, detailed`,
            `${conceptName} in an outdoor adventure setting, dynamic pose, professional photography`,
            `close-up portrait of ${conceptName}, shallow depth of field, artistic lighting`,
            `${conceptName} playing happily, candid lifestyle moment, warm lighting`,
            `artistic photo of ${conceptName}, golden hour lighting, beautiful composition`,
            `${conceptName} resting comfortably indoors, cozy atmosphere, soft lighting`,
            `action shot of ${conceptName}, motion capture, outdoor setting, high energy`,
            `elegant portrait of ${conceptName}, professional headshot style, clean background`,
            `${conceptName} in a scenic landscape, wide angle shot, natural environment`
        ];
        
        for (let i = 0; i < numImages; i++) {
            const variation = variations[i % variations.length];
            
            try {
                addDebugLog(`🎨 Generating LoRA image ${i + 1}/${numImages}: ${variation}`, 'info');
                
                // Update progress
                if (progressCallback) {
                    progressCallback({
                        current: i + 1,
                        total: numImages,
                        status: `Generating with your custom model: ${i + 1}/${numImages}`,
                        detail: variation,
                        progress: ((i + 1) / numImages) * 100
                    });
                }
                
                // Generate with LoRA
                const imageUrl = await this.generateImage(variation, true, loraUrl);
                
                if (imageUrl) {
                    const result = {
                        id: i + 1,
                        prompt: variation,
                        imageUrl: imageUrl,
                        status: 'success',
                        generatedAt: new Date().toISOString()
                    };
                    
                    results.push(result);
                    successCount++;
                    addDebugLog(`✅ Generated LoRA image ${i + 1}/${numImages}`, 'success');
                } else {
                    throw new Error('No image URL returned');
                }
                
                // Delay between generations
                await this.delay(2000);
                
            } catch (error) {
                addDebugLog(`❌ Failed to generate LoRA image ${i + 1}: ${error.message}`, 'error');
                
                const failedResult = {
                    id: i + 1,
                    prompt: variation,
                    imageUrl: null,
                    status: 'failed',
                    error: error.message,
                    generatedAt: new Date().toISOString()
                };
                
                results.push(failedResult);
            }
        }
        
        const endTime = Date.now();
        const durationMinutes = Math.round((endTime - startTime) / (1000 * 60));
        
        addDebugLog(`🎨 Dataset generation complete: ${successCount}/${numImages} successful`, 'info');
        
        return {
            results: results,
            successCount: successCount,
            failCount: numImages - successCount,
            totalRequested: numImages,
            durationMinutes: durationMinutes,
            estimatedCost: (successCount * APP_CONFIG.costPerGeneration).toFixed(3)
        };
    }

    // Helper methods
    extractImageUrl(result) {
        if (result.output?.image_url) {
            return result.output.image_url;
        } else if (result.output?.[0]?.image) {
            return `data:image/png;base64,${result.output[0].image}`;
        } else if (result.output?.[0]) {
            return `data:image/png;base64,${result.output[0]}`;
        }
        return null;
    }

    async fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = error => reject(error);
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Get training cost estimate
    getTrainingCostEstimate() {
        return {
            estimatedMinutes: APP_CONFIG.estimatedTrainingMinutes,
            costPerMinute: APP_CONFIG.costPerTrainingMinute,
            totalCost: (APP_CONFIG.estimatedTrainingMinutes * APP_CONFIG.costPerTrainingMinute).toFixed(2)
        };
    }

    // Get generation cost estimate
    getGenerationCostEstimate(numImages) {
        return {
            costPerImage: APP_CONFIG.costPerGeneration,
            totalCost: (numImages * APP_CONFIG.costPerGeneration).toFixed(3),
            numImages: numImages
        };
    }
}

// Create global instance
const runpodAPI = new RunPodAPI();