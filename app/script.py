import os
import requests
import zipfile
import io

# This downloads the model so that you can run it locally
# stores it in project-root/app/models

def download_model():
    print("📥 Downloading pre-trained model...")
    model_url = "https://huggingface.co/Group6comp263/brain_tumor_models/resolve/main/brain_tumor_model_transfer_learning.keras"
    save_path = "models/downloaded_brain_tumor_model.keras"
    
    os.makedirs("app/models", exist_ok=True)
    
    response = requests.get(model_url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model saved to app/models/")




if __name__ == "__main__":
    download_model()
