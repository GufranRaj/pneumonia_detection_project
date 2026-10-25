import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)
st.title("🫁 Pneumonia Detection Using CNN")
st.write("Upload a chest X-ray image for an experimental model prediction.")
st.warning(
    "Educational project only. This model is not intended for medical "
    "diagnosis. Its prediction score is not a confirmed probability of disease."
)
MODEL_PATH = "pneumonia_cnn_model.keras"
# Replace this with your Google Drive file ID
FILE_ID = "1yh75cU0TMw2UiM3QUeUDXT3g1OxycuqM"
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)
try:
    with st.spinner("Loading CNN model..."):
        model = load_cnn_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Model could not be loaded. Please check the Drive file ID and sharing permission.")
    st.stop()
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
    if st.button("Analyze X-ray"):
        resized = image.resize((150, 150))
        array = np.array(resized, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)
        with st.spinner("Analyzing image..."):
            score = float(model.predict(array, verbose=0)[0][0])
        st.subheader("📋 Prediction Result")
        if score >= 0.5:
            st.error("Model prediction: PNEUMONIA")
            st.metric("Model pneumonia score", f"{score * 100:.2f}%")
        else:
            st.success("Model prediction: NORMAL")
            st.metric("Model pneumonia score", f"{score * 100:.2f}%")
        st.info(
            "The score is the model's output, not a validated disease probability. "
            "This model cannot determine pneumonia stage or severity. "
            "Consult a qualified medical professional for interpretation."
        )
