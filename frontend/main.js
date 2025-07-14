// Simplified Frontend Logic for AI Dataset Generator
// main.js

class DatasetGenerator {
    constructor() {
        this.uploadedFiles = [];
        this.currentJob = null;
        this.backendUrl = BACKEND_URL || 'http://localhost:5000';
        
        this.initializeEventListeners();
        this.testBackendConnection();
    }

    initializeEventListeners() {
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            this.handleFiles(files);
        });

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            this.handleFiles(Array.from(e.target.files));
        });

        // Form submission
        const generateBtn = document.getElementById('generateBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.startPipeline());
        }
    }

    async testBackendConnection() {
        try {
            const response = await fetch(`${this.backendUrl}/api/health`);
            if (response.ok) {
                const data = await response.json();
                this.showStatus(`✅ Backend connected: ${data.backend}`, 'success');
                console.log('Backend health:', data);
            } else {
                throw new Error(`Backend returned ${response.status}`);
            }
        } catch (error) {
            this.showStatus(`❌ Backend connection failed: ${error.message}`, 'error');
            console.error('Backend connection failed:', error);
        }
    }

    handleFiles(files) {
        const imageFiles = files.filter(file => file.type.startsWith('image/'));
        
        if (imageFiles.length === 0) {
            this.showStatus('Please select image files', 'warning');
            return;
        }

        if (this.uploadedFiles.length + imageFiles.length > APP_CONFIG.maxUploadFiles) {
            this.showStatus(`Maximum ${APP_CONFIG.maxUploadFiles} files allowed`, 'warning');
            return;
        }

        this.uploadedFiles = [...this.uploadedFiles, ...imageFiles];
        this.displayUploadedFiles();
        this.showStatus(`Uploaded ${imageFiles.length} images (${this.uploadedFiles.length} total)`, 'info');
    }

    displayUploadedFiles() {
        const container = document.getElementById('uploadedFiles');
        if (!container) return;

        container.innerHTML = '';
        
        this.uploadedFiles.forEach((file, index) => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'file-item';
            fileDiv.innerHTML = `
                <div class="file-icon">📷</div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${(file.size / 1024).toFixed(1)} KB</div>
                </div>
                <button onclick="datasetGenerator.removeFile(${index})" class="btn" style="padding: 5px 10px; margin: 0;">×</button>
            `;
            container.appendChild(fileDiv);
        });
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.displayUploadedFiles();
        this.showStatus(`Removed file. ${this.uploadedFiles.length} files remaining`, 'info');
    }

    async startPipeline() {
        if (this.uploadedFiles.length < APP_CONFIG.minUploadFiles) {
            this.showStatus(`Please upload at least ${APP_CONFIG.minUploadFiles} training images`, 'warning');
            return;
        }

        const conceptName = document.getElementById('conceptName')?.value?.trim();
        const numImages = parseInt(document.getElementById('numImages')?.value || APP_CONFIG.defaultNumImages);

        if (!conceptName) {
            this.showStatus('Please enter a concept name', 'warning');
            return;
        }

        try {
            this.showStatus('Starting AI pipeline...', 'info');
            this.setProgress(0);
            
            // Prepare form data
            const formData = new FormData();
            this.uploadedFiles.forEach(file => {
                formData.append('images', file);
            });
            formData.append('concept_name', conceptName);
            formData.append('num_images', numImages);

            // Start pipeline
            const response = await fetch(`${this.backendUrl}/api/generate-dataset`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }

            const result = await response.json();
            this.currentJob = result.job_id;
            
            this.showStatus(`Pipeline started! Job ID: ${result.job_id}`, 'success');
            this.showProgress(true);
            
            // Start monitoring
            this.monitorProgress();

        } catch (error) {
            this.showStatus(`Failed to start pipeline: ${error.message}`, 'error');
            console.error('Pipeline start error:', error);
        }
    }

    async monitorProgress() {
        if (!this.currentJob) return;

        try {
            const response = await fetch(`${this.backendUrl}/api/job-status/${this.currentJob}`);
            
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.status}`);
            }

            const status = await response.json();
            
            // Update UI
            this.setProgress(status.progress || 0);
            this.showStatus(status.message || status.status, 'info');

            if (status.status === 'complete') {
                this.handleCompletion(status.result);
            } else if (status.status === 'failed') {
                this.showStatus(`Pipeline failed: ${status.error}`, 'error');
                this.showProgress(false);
            } else {
                // Continue monitoring
                setTimeout(() => this.monitorProgress(), 5000);
            }

        } catch (error) {
            console.error('Progress monitoring error:', error);
            setTimeout(() => this.monitorProgress(), 10000); // Retry after longer delay
        }
    }

    handleCompletion(result) {
        this.showStatus('🎉 Pipeline completed successfully!', 'success');
        this.setProgress(100);
        
        console.log('Pipeline result:', result);
        
        // Show results
        this.displayResults(result);
        
        // Enable download
        this.enableDownload(result);
        
        this.showProgress(false);
    }

    displayResults(result) {
        const resultsContainer = document.getElementById('results');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = `
            <div class="results-section">
                <h3>Dataset Created Successfully! 🎉</h3>
                <div class="result-stats">
                    <div class="stat-item">
                        <strong>Total Images:</strong> ${result.total_images || 0}
                    </div>
                    <div class="stat-item">
                        <strong>Segmented Images:</strong> ${result.segmented_images || 0}
                    </div>
                    <div class="stat-item">
                        <strong>Concept:</strong> ${result.concept_name || 'Unknown'}
                    </div>
                </div>
                
                ${result.preview_images ? this.createPreviewHtml(result.preview_images) : ''}
                
                <div class="download-section">
                    <button id="downloadBtn" class="btn download-btn">
                        📁 Download Complete Dataset
                    </button>
                </div>
            </div>
        `;
    }

    createPreviewHtml(previewImages) {
        const previewHtml = previewImages.slice(0, 4).map((img, index) => `
            <div class="preview-item">
                <img src="${img.image_url}" alt="Preview ${index + 1}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 8px;">
            </div>
        `).join('');

        return `
            <div class="preview-section">
                <h4>Preview Images:</h4>
                <div class="preview-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 15px 0;">
                    ${previewHtml}
                </div>
            </div>
        `;
    }

    enableDownload(result) {
        const downloadBtn = document.getElementById('downloadBtn');
        if (downloadBtn && result.download_url) {
            downloadBtn.onclick = () => {
                window.open(`${this.backendUrl}${result.download_url}`, '_blank');
                this.showStatus('Download started...', 'info');
            };
        }
    }

    setProgress(percent) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        if (progressFill) {
            progressFill.style.width = `${percent}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${percent}%`;
        }
    }

    showProgress(show) {
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer) {
            progressContainer.style.display = show ? 'block' : 'none';
        }
    }

    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status ${type}`;
        }
        
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    reset() {
        this.uploadedFiles = [];
        this.currentJob = null;
        this.displayUploadedFiles();
        this.showProgress(false);
        this.setProgress(0);
        this.showStatus('Ready to start', 'info');
        
        const resultsContainer = document.getElementById('results');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.datasetGenerator = new DatasetGenerator();
});

// Make functions available globally for HTML onclick handlers
window.startPipeline = () => window.datasetGenerator?.startPipeline();
window.resetPipeline = () => window.datasetGenerator?.reset();