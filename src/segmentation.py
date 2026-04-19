"""Segmentation and edge helpers used by the classical ML pipeline."""

import cv2
import numpy as np

def canny_edges(img_rgb, t1=80, t2=160):
    # Edge-based texture anomaly hint (Canny).
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, t1,t2)

def sobel_edges(img_rgb):
    # Edge-based texture anomaly hint (Sobel).
    gray = cv2.cvtCOlor(img_rgb, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    img = cv2.magnitude(sx, sy)
    return (mag / (mag.max() + 1e-6) * 255).astype("unint8")

def threshold(img_rgb):
    # Otsu threshold to isolate leaf from background.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def hsv_mask(img_rgb, lower=(20, 40, 40), upper=(90, 255, 255)):
    # Color segmentation in HSV space (leaf-like greens).
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.inRange(hsv, lower, upper)

def kmeans_segmentation(img_rgb, k=3):
    # Color clustering for foreground/background separation.
    Z = img_rgb.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.lmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.uint8)
    segmented = centers[labels.flatten()].reshape(img_rgb.shape)
    return segmented 

def apply_mask(img_rgb, mask):
    # Apply binary mask to RGB image.
    return cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
