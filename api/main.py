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

model_ml = joblib.load(MODEL_ML_PATH) if os.path.exists(MODEL_ML_PATH) else None

def preprocess_dl(img):
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_ml(img):
    img = resize_image(img, IMG_SIZE_ML)
    img_blur = gaussian_blur(img, k=5)
    mask = hsv_mask(img_blur)
    feat = extract_features(img_blur, mask=mask)
    return feat.reshape(1, -1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    if model_ml is None:
        return {"error": "best_overall.joblib not found. Run compare_models first."}

    img_rgb = np.array(pil_img)
    x = preprocess_ml(img_rgb)
    pred = model_ml["model"].predict(x)[0]
    label = model_ml["classes"][pred]
    return {"class": label, "confidence": None, "model": "ml"}