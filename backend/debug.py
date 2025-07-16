#!/usr/bin/env python3
"""
Test the actual image generation functionality
"""
import requests
import json
import time
import base64
from PIL import Image
import io

# Test image generation endpoint directly
def test_image_generation():
    print("🎨 Testing image generation...")
    
    test_payload = {
        "prompt": "a beautiful landscape, masterpiece, best quality",
        "negative_prompt": "blurry, low quality, distorted",
        "num_inference_steps": 20,  # Reduced for faster testing
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
        "seed": 42
    }
    
    try:
        print(f"📡 Sending request to http://localhost:39515/generate")
        response = requests.post(
            'http://localhost:39515/generate', 
            json=test_payload, 
            timeout=60  # Give it time to generate
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generation successful!")
            print(f"📝 Status: {result.get('status')}")
            
            if result.get('image'):
                # Save the generated image to verify it worked
                image_data = base64.b64decode(result['image'])
                image = Image.open(io.BytesIO(image_data))
                image.save('test_generated_image.png')
                print(f"💾 Image saved as 'test_generated_image.png'")
                print(f"📏 Image size: {image.size}")
            else:
                print("❌ No image data in response")
                
            return True
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - generation might be taking too long")
        return False
    except Exception as e:
        print(f"❌ Generation test error: {e}")
        return False

# Test enhanced generation
def test_enhanced_generation():
    print("\n🎨 Testing enhanced image generation...")
    
    test_payload = {
        "prompt": "a cute cat sitting on a windowsill",
        "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
        "seed": 123
    }
    
    try:
        response = requests.post(
            'http://localhost:39515/generate-enhanced', 
            json=test_payload, 
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Enhanced generation successful!")
            
            if result.get('image'):
                image_data = base64.b64decode(result['image'])
                image = Image.open(io.BytesIO(image_data))
                image.save('test_enhanced_image.png')
                print(f"💾 Enhanced image saved as 'test_enhanced_image.png'")
            
            return True
        else:
            print(f"❌ Enhanced generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced generation error: {e}")
        return False

# Test the full pipeline through your backend
def test_backend_pipeline():
    print("\n🔄 Testing backend pipeline simulation...")
    
    # First, let's just test if we can hit your backend's endpoints
    try:
        # Test basic health
        health_response = requests.get('http://localhost:5000/health')
        print(f"🏥 Backend health: {health_response.json()}")
        
        # Test RunPod connectivity through backend
        runpod_test = requests.get('http://localhost:5000/api/test-runpod')
        print(f"🔗 RunPod connectivity: {runpod_test.json()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend pipeline test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive tests...")
    print("=" * 50)
    
    # Test 1: Direct image generation
    gen_success = test_image_generation()
    
    # Test 2: Enhanced generation
    enhanced_success = test_enhanced_generation()
    
    # Test 3: Backend pipeline
    backend_success = test_backend_pipeline()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"✅ Basic generation: {'PASS' if gen_success else 'FAIL'}")
    print(f"✅ Enhanced generation: {'PASS' if enhanced_success else 'FAIL'}")
    print(f"✅ Backend connectivity: {'PASS' if backend_success else 'FAIL'}")
    
    if gen_success and enhanced_success and backend_success:
        print("\n🎉 ALL TESTS PASSED! Your pipeline is working correctly.")
        print("💡 You can now try uploading images through your frontend.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")