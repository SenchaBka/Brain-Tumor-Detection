from model_repository import ModelRepository
import numpy as np

# Service is initlized with the Model of choice
# Method is called with the Processed image as an input
class PredictionService:
    def __init__(self, repository: ModelRepository):
        self.repository = repository

    def predict_tumor(self, processed_image: np.ndarray) -> dict:
        model = self.repository.load_model()
        raw_prediction = model.predict(processed_image)
        
        probability = float(raw_prediction[0][0])
        label = "Tumor" if probability > 0.5 else "Healthy"
        
        return {
            "label": label,
            "confidence": round(probability, 4),
            "raw_output": raw_prediction.tolist()
        }