import tensorflow as tf
import numpy as np
from PIL import Image

# --- ADAPTER LAYER ---
class TFModelAdapter:
    def __init__(self, model_path):
        # Load the .keras file or SavedModel folder
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")

    def predict(self, pil_image):
        # Resize to match training
        img = pil_image.resize((224, 224))
        # Convert to array (0-255)
        img_array = np.array(img).astype(np.float32)
        # Add batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array, verbose=0)
        return prediction[0][0]

# --- SERVICE LAYER ---
class PredictionService:
    def __init__(self, adapter):
        self.adapter = adapter

    def get_diagnosis(self, image_path):
        image = Image.open(image_path).convert("RGB")
        probability = self.adapter.predict(image)
        
        label = "Tumor Detected" if probability > 0.5 else "Healthy"
        return {"label": label, "confidence": f"{probability:.2%}"}

# --- TEST EXECUTION ---
if __name__ == "__main__":
    # Point this to your downloaded model
    adapter = TFModelAdapter("models/brain_tumor_tf_model_v1")
    service = PredictionService(adapter)
    
    # Test with an actual image on your PC
    result = service.get_diagnosis("test_mri_scan.jpg")
    print(f"\nResult: {result['label']} (Prob: {result['confidence']})")