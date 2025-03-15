# Modified test_api.py with debugging
import requests
import base64
from PIL import Image
import io
import json

def test_api():
    url = "http://localhost:8800/v1/images/generations"
    
    payload = {
        "prompt": "A cute lion",
        "steps": 1,
        "width": 512,
        "height": 512
    }
    
    print(f"Sending request with prompt: '{payload['prompt']}'")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("Success! Image generated.")
        
        # Debug: Check what we're getting back
        print(f"Response keys: {result.keys()}")
        print(f"Data length: {len(result.get('data', []))}")
        
        if 'data' in result and len(result['data']) > 0:
            # Debug: Check the b64_json field
            img_data = result['data'][0].get('b64_json', '')
            print(f"Base64 data length: {len(img_data)}")
            print(f"Base64 data starts with: {img_data[:50]}...")
            
            # Check if it's actually base64 encoded
            try:
                img_bytes = base64.b64decode(img_data)
                print(f"Decoded bytes length: {len(img_bytes)}")
                print(f"First few bytes: {img_bytes[:20]}")
                
                # Try to open the image
                img = Image.open(io.BytesIO(img_bytes))
                img.save("test_output.png")
                print("Image saved to test_output.png")
            except Exception as e:
                print(f"Error decoding/opening image: {e}")
                
                # Try saving the raw base64 for inspection
                with open("debug_base64.txt", "w") as f:
                    f.write(img_data)
                print("Saved base64 data to debug_base64.txt for inspection")
        else:
            print("No data found in response")
            print(f"Full response: {json.dumps(result, indent=2)}")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_api()