"""Build ML-ready feature dataset from the tomato image folders."""

import os
import numpy as np
from .config import DATA_ROOT, IMG_SIZE_ML, TOMATO_CLASSES, ensure_dirs
from .utils_io import list_images
from .preprocess import load_image, resize_image, gaussian_blur
from .segmentation import hsv_mask
from .features import extract_features

def build_ml_dataset():
    # Build feature matrix X and labels y for the classical ML pipeline.
    ensure_dirs()
    # X: list of feature vectors, y: list of class indices.
    X, y = [], []
    classes = TOMATO_CLASSES

    for idx, cls in enumerate(classes):
        # Iterate through each class folder and extract features per image.
        cls_dir = os.path.join(DATA_ROOT, cls)
        images = list_images(cls_dir)

        for img_path in images:
            # Load and normalize image size before extracting features.
            img = load_image(img_path)
            img = resize_image(img, IMG_SIZE_ML)

            # Denoising before segmentation/features
            img_blur = gaussian_blur(img, k=5)

            # Build a color-based mask to focus on leaf pixels.
            mask = hsv_mask(img)

            # Extract color/texture/shape features using the mask.
            feat = extract_features(img_blur, mask=mask)
            X.append(feat)
            y.append(idx)
    # Return feature matrix, label vector, and ordered class names.
    return np.array(X), np.array(y), classes