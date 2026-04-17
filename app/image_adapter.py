import numpy as np
from PIL import Image
import io

class ImageAdapter:
    @staticmethod
    def transform(file_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
        # Load and resize
        img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        img = img.resize(target_size)
        
        # Convert to float32 array (EfficientNet handles rescaling)
        img_array = np.array(img).astype(np.float32)
        
        # Add batch dimension: (1, 224, 224, 3)
        return np.expand_dims(img_array, axis=0)