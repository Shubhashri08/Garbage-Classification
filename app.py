import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model("Effiicientnetv2b2.keras")  

# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Page setup
st.set_page_config(page_title="🗑️ Garbage Classifier", layout="centered")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This AI model classifies garbage into 6 categories:
- 📦 Cardboard  
- 🍾 Glass  
- 🧲 Metal  
- 📄 Paper  
- 🛍️ Plastic  
- 🗑️ Trash  

Upload an image to see the classification result.
""")

# App Title
st.title("🧠 Smart Garbage Classification")
st.markdown("### Upload an image of waste to classify it.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Spinner during prediction
    with st.spinner("Classifying..."):
        image = image.resize((124, 124))  # Match input size
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = 100 * np.max(prediction)

    # Display results
    st.success(f"🧾 Predicted Class: **{predicted_class.upper()}**")
    st.progress(int(confidence))
    st.markdown(f"**Model Confidence:** `{confidence:.2f}%`")

# Footer
st.markdown("---")
st.markdown("Made with using EfficientNetV2B2 | [GitHub](https://github.com) | [Streamlit](https://streamlit.io)")
