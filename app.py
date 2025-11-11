# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your saved model
model = tf.keras.models.load_model('happy_sad_classifier.h5')

# Class labels (from your training dataset)
class_labels = {0: 'happy', 1: 'sad'}  # Replace with your actual labels

st.title("Image Classification App")
st.write("Upload an image, and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = img.resize((200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize

    # Predict
    val = model.predict(x)
    predicted_class_index = int(np.round(val[0][0]))
    predicted_label = class_labels.get(predicted_class_index, 'Unknown')

    st.write(f"Prediction: **{predicted_label}**")
    st.write(f"Raw score: {val[0][0]:.4f}")
