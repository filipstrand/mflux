# Modified test_api.py using openai package
import base64
from PIL import Image
import io
import json
from openai import OpenAI

def test_api():
    # Configure the client to use your local server
    client = OpenAI(
        base_url="http://localhost:8800/v1",
        api_key="dummy-key"  # API key is required but not used by your local server
    )
    
    prompt = "A cute lion"
    print(f"Sending request with prompt: '{prompt}'")
    
    try:
        # Make the API call using the OpenAI client
        response = client.images.generate(
            model="dall-e-3",
            prompt="a white siamese cat",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        print("Success! Image generated.")
        
        # Debug: Check what we're getting back
        print(f"Response type: {type(response)}")
        print(f"Data length: {len(response.data)}")
        
        if response.data and len(response.data) > 0:
            # Get the first image
            image_data = response.data[0]
            
            # Check if b64_json is available
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                img_data = image_data.b64_json
                print(f"Base64 data length: {len(img_data)}")
                print(f"Base64 data starts with: {img_data[:50]}...")
                
                # Decode and save the image
                try:
                    img_bytes = base64.b64decode(img_data)
                    print(f"Decoded bytes length: {len(img_bytes)}")
                    print(f"First few bytes: {img_bytes[:20]}")
                    
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save("test_output.png")
                    print("Image saved to test_output.png")
                except Exception as e:
                    print(f"Error decoding/opening image: {e}")
                    
                    # Save the raw base64 for inspection
                    with open("debug_base64.txt", "w") as f:
                        f.write(img_data)
                    print("Saved base64 data to debug_base64.txt for inspection")
            else:
                # If using URL format instead of base64
                if hasattr(image_data, 'url') and image_data.url:
                    print(f"Image URL: {image_data.url}")
                else:
                    print("No image data found in the response")
                    print(f"Available attributes: {dir(image_data)}")
        else:
            print("No data found in response")
            print(f"Full response: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()