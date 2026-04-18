import os
import random
import numpy as np

# Global seed for reproducibility
SEED = 42

DATA_ROOT = "data/dataset_tomato"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# Image sizes for ML (features) and DL (CNN)
IMG_SIZE_ML = (128, 128)
IMG_SIZE_DL = (224, 224)

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato_healthy"
]

def set_seed(seed=SEED):
    # Ensure repeatable experiments
    random.seed(seed)
    np.random.seed(seed)

def ensure_dirs():
    # Create output folders if missing
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)