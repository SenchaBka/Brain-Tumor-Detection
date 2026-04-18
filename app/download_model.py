import os
import requests

# This downloads the model so that you can run it locally
# stores it in app/models

def download_model():
    print("📥 Downloading pre-trained model...")
    model_url = "https://huggingface.co/Group6comp263/brain_tumor_models/resolve/main/brain_tumor_model_transfer_learning.keras"
    
    # Create models directory (relative to current working directory)
    os.makedirs("models", exist_ok=True)
    save_path = "models/downloaded_brain_tumor_model.keras"
    
    try:
        response = requests.get(model_url, stream=True, timeout=30)
        response.raise_for_status()  # Raise error for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size:
                        percent = (downloaded_size / total_size) * 100
                        print(f"  {percent:.1f}% downloaded", end="\r")
        
        print(f"\n✅ Model successfully saved to {save_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


if __name__ == "__main__":
    download_model()
