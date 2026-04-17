import tensorflow as tf

# Loads model according to the model adderss
class ModelRepository:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def load_model(self):
        if self._model is None:
            self._model = tf.keras.models.load_model(self.model_path)
        return self._model