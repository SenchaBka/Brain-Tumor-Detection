from flask import Flask, request, jsonify
from model_repository import ModelRepository
from image_adapter import ImageAdapter
from prediction_service import PredictionService

app = Flask(__name__)

# Load the model
repo = ModelRepository("./models/downloaded_brain_tumor_model.keras")

# Create an instance of the service
service = PredictionService(repo)


# Predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file'].read()
    
    try:
        # 1. Use Adapter to process data
        input_data = ImageAdapter.transform(file)
        
        # 2. Use Service to get business result
        result = service.predict_tumor(input_data)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)