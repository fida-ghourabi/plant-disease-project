"""Feature extraction: color, texture (GLCM), and shape descriptors."""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def color_hist_hsv(img_rgb, bins=16):
    # HSV histogram for color features (concatenated per channel).
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist = []
    for i in range(3):
        # Per-channel histogram with fixed bin count.
        h = cv2.calcHist([hsv], [i], None, [bins], [0, 256]).flatten()
        hist.extend(h)
    hist = np.array(hist, dtype=np.float32)
    # Normalize to make features scale-invariant.
    return hist / (hist.sum()+1e-6)

def glcm_features(img_rgb):
    # GLCM texture features (contrast, correlation, energy, homogeneity).
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],

    ], dtype=np.float32)


def shape_features(mask):
     # Shape: area, perimeter, circularity from the largest contour.
     cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     if len(cnts) == 0 :
         # No contour found -> return zeros.
         return np.array([0, 0, 0], dtype=np.float32)
     c = max(cnts, key=cv2.contourArea)
     area = cv2.contourArea(c)
     peri = cv2.arcLength(c, True)
     circ = 0.0 if peri == 0 else (4 * np.pi * area) / (peri * peri)
     return np.array([area, peri, circ], dtype=np.float32)
         

def extract_features(img_rgb, mask=None):
    # Combine color + texture + shape into a single feature vector.
    feat_color = color_hist_hsv(img_rgb)
    feat_glcm = glcm_features(img_rgb)
    feat_shape = shape_features(mask) if mask is not None else np.zeros(3, dtype=np.float32)
    return np.concatenate([feat_color, feat_glcm, feat_shape])
