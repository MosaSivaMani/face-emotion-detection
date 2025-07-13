import requests
import json

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get('http://localhost:5000/health')
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_upload_endpoint():
    """Test the upload endpoint with a sample image"""
    try:
        # Create a simple test image (1x1 pixel)
        import numpy as np
        import cv2
        
        # Create a simple test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
        _, buffer = cv2.imencode('.jpg', test_image)
        
        files = {'file': ('test.jpg', buffer.tobytes(), 'image/jpeg')}
        response = requests.post('http://localhost:5000/upload', files=files)
        
        print(f"Upload test status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Upload test response: {json.dumps(result, indent=2)}")
        else:
            print(f"Upload test error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Upload test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Face Emotion Detection API...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_health_endpoint()
    
    # Test upload endpoint
    print("\n2. Testing upload endpoint...")
    upload_ok = test_upload_endpoint()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Health endpoint: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Upload endpoint: {'✓ PASS' if upload_ok else '✗ FAIL'}")
    
    if health_ok and upload_ok:
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the server logs.") 