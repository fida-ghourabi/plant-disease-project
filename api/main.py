"""FastAPI service exposing /predict for tomato leaf classification."""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os
import numpy as np
import joblib

from src.preprocess import load_image, resize_image, gaussian_blur
from src.segmentation import hsv_mask
from src.features import extract_features
from src.config import IMG_SIZE_ML

app = FastAPI(title="Plant Disease API", version="1.0")

MODEL_ML_PATH = "models/best_overall.joblib"
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato_healthy"
]

# Load the best ML model if it exists (written by compare_models).
model_ml = joblib.load(MODEL_ML_PATH) if os.path.exists(MODEL_ML_PATH) else None

def preprocess_dl(img):
    # DL preprocessing (currently unused, kept for future extension).
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_ml(img):
    # Match the ML training pipeline (resize -> blur -> HSV mask -> features).
    img = resize_image(img, IMG_SIZE_ML)
    img_blur = gaussian_blur(img, k=5)
    mask = hsv_mask(img_blur)
    feat = extract_features(img_blur, mask=mask)
    return feat.reshape(1, -1)

@app.get("/health")
def health():
    # Simple health check endpoint.
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes and run the ML model for prediction.
    content = await file.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    if model_ml is None:
        # Inform the caller if the best model is missing.
        return {"error": "best_overall.joblib not found. Run compare_models first."}

    # Convert PIL image to numpy array for preprocessing.
    img_rgb = np.array(pil_img)
    x = preprocess_ml(img_rgb)
    pred = model_ml["model"].predict(x)[0]
    label = model_ml["classes"][pred]
    # Confidence is not computed for the classical ML pipeline yet.
    return {"class": label, "confidence": None, "model": "ml"}