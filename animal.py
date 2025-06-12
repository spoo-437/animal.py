import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Animal Image Classifier", layout="wide")
st.title("🐾 Animal Image Classifier")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_model.h5")
    return model

model = load_model()

# Class labels (change as per your dataset)
class_names = ['cat', 'dog', 'elephant', 'lion']

# Upload image
uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")

