import os
import numpy as np
from .config import DATA_ROOT, IMG_SIZE_ML, TOMATO_CLASSES, ensure_dirs
from .utils_io import list_images
from .preprocess import load_image, resize_image, gaussian_blur
from .segmentation import hsv_mask
from .features import extract_features

def build_ml_dataset():
    # Build X/y for ML pipeline
    ensure_dirs()
    X, y = [], []
    classes = TOMATO_CLASSES

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(DATA_ROOT, cls)
        images = list_images(cls_dir)

        for img_path in images:
            img = load_image(img_path)
            img = resize_image(img, IMG_SIZE_ML)

            # Denoising before segmentation/features
            img_blur = gaussian_blur(img, k=5)

            mask = hsv_mask(img)

            feat = extract_features(img_blur, mask=mask)
            X.append(feat)
            y.append(idx)
    return np.array(X), np.array(y), classes