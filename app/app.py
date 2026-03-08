import streamlit as st
import numpy as np
from .utils import load_model_file, load_class_names, predict_breed

# PAGE CONFIG
st.set_page_config(
    page_title="🐶 Dog Breed Classifier",
    page_icon="🐕",
    layout="wide"
)

# HEADER DESIGN
st.markdown("""
    <div style="text-align:center; padding:20px">
        <h1 style="color:#3A6D8C;">🐶 Dog Breed Identification</h1>
        <p style="font-size:18px; color:#444;">
            Upload an image OR capture from camera and let AI identify the dog breed.
        </p>
    </div>
""", unsafe_allow_html=True)

# LOAD MODEL + CLASS NAMES
@st.cache_resource
def load_all():
    model = load_model_file()
    class_names = load_class_names()
    return model, class_names

model, class_names = load_all()

# LAYOUT: TWO COLUMNS
left, right = st.columns([1, 1])

with left:
    st.subheader("📸 Upload or Capture Image")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    st.write("OR")

    camera_image = st.camera_input("Take a picture")

    # Decide which image to use
    img_source = uploaded if uploaded else camera_image

    if img_source:
        st.image(img_source, caption="Selected Image", use_column_width=True)

with right:
    st.subheader("🔍 Prediction")

    if img_source:
        if st.button("Identify Breed"):
            with st.spinner("Analyzing image..."):
                breed, confidence, preds = predict_breed(model, class_names, img_source)

            st.success(f"**Breed: {breed}**")
            st.write(f"Confidence: **{confidence*100:.2f}%**")

            # Sort and show top 5 results
            st.subheader("Top 5 Predictions")
            sorted_idx = np.argsort(preds)[::-1][:5]

            for i in sorted_idx:
                st.write(f"✔ {class_names[i]} — {preds[i]*100:.2f}%")

            # Full chart
            st.bar_chart(preds)
    else:
        st.info("Upload an image or take a picture to begin.")
