import streamlit as st
from PIL import Image
import numpy as np
import joblib

from src.preprocess import resize_image, gaussian_blur
from src.segmentation import hsv_mask
from src.features import extract_features
from src.config import IMG_SIZE_ML

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    :root {
        --bg: #f7f4ee;
        --card: #fffaf2;
        --ink: #1f2a1f;
        --accent: #2f6b4f;
        --accent-2: #d8b36a;
        --muted: #6a6f6a;
    }

    html, body, [class*="stApp"] {
        background: radial-gradient(circle at 10% 10%, #fff6e6 0%, #f4efe6 35%, #eef3ed 70%);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
    }

    .hero {
        padding: 28px 32px;
        background: linear-gradient(120deg, #ffffff 0%, #f6f0e6 60%, #eef4ef 100%);
        border-radius: 18px;
        border: 1px solid #e7dfd0;
        box-shadow: 0 10px 30px rgba(31, 42, 31, 0.08);
    }

    .badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        background: #1f2a1f;
        color: #fff;
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 12px;
    }

    .card {
        background: var(--card);
        border-radius: 16px;
        border: 1px solid #e7dfd0;
        padding: 18px 20px;
        box-shadow: 0 8px 20px rgba(31, 42, 31, 0.06);
    }

    .label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .value {
        font-size: 20px;
        font-weight: 700;
        color: var(--accent);
    }

    .top3 {
        margin-top: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: var(--ink);
    }

    .uploader label {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="badge">Plant Disease AI</div>
        <h1>Plant Disease Classifier</h1>
        <p>Upload a leaf image and get a fast diagnosis. This demo uses a classical ML model trained on tomato classes with a professional preprocessing pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "models/best_overall.joblib"
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
classes = model_data["classes"]

def preprocess_ml(pil_img):
    img_rgb = np.array(pil_img.convert("RGB"))
    img_rgb = resize_image(img_rgb, IMG_SIZE_ML)
    img_blur = gaussian_blur(img_rgb, k=5)
    mask = hsv_mask(img_blur)
    feat = extract_features(img_blur, mask=mask)
    return feat.reshape(1, -1)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload & Preview")
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction")
    if uploaded:
        x = preprocess_ml(img)
        pred = model.predict(x)[0]

        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
        else:
            top3_idx = [pred]

        st.markdown("<div class='label'>Predicted class</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>{classes[pred]}</div>", unsafe_allow_html=True)

        if probs is not None:
            st.markdown("<div class='label' style='margin-top:12px;'>Top-3 scores</div>", unsafe_allow_html=True)
            for i in top3_idx:
                st.markdown(
                    f"<div class='top3'>{classes[i]}: {probs[i]:.3f}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.write("Upload an image to get a prediction.")
    st.markdown("</div>", unsafe_allow_html=True)