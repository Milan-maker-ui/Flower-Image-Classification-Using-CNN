import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Flower Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Flower Image Classifier")
st.write("Upload a flower photo and the CNN will predict its category.")

MODEL_PATH = Path("artifacts/best_flower_model.keras")
CLASS_PATH = Path("artifacts/class_names.json")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        return None, None
    return tf.keras.models.load_model(MODEL_PATH), json.loads(CLASS_PATH.read_text())

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.warning("No trained model found. Train it first using `python src/train.py`.")
    st.stop()

model, class_names = load_model()
uploaded = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize((180, 180))
    array = np.asarray(resized, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    best = int(np.argmax(probabilities))
    st.subheader(f"Prediction: {class_names[best]}")
    st.metric("Confidence", f"{probabilities[best] * 100:.2f}%")
    st.write("Class probabilities")
    chart_data = {name: float(prob) for name, prob in zip(class_names, probabilities)}
    st.bar_chart(chart_data)
