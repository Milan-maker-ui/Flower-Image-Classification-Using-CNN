import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
try:
    from .model import IMAGE_SIZE
except ImportError:
    from model import IMAGE_SIZE

def predict_image(image, model_path="artifacts/best_flower_model.keras",
                  class_path="artifacts/class_names.json"):
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(Path(class_path).read_text())
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), {
        name: float(prob) for name, prob in zip(class_names, probabilities)
    }
