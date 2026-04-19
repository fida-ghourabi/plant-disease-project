"""Basic image preprocessing helpers (resize, color space, blur)."""

import cv2

def load_image(path):
    # Load image from disk (OpenCV loads in BGR) and convert to RGB.
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, size):
    # Resize to a fixed size used by ML/DL pipelines.
    return cv2.resize(img, size)

def to_hsv(img_rgb):
    # Convert RGB image to HSV color space.
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

def to_gray(img_rgb):
    # Convert RGB image to grayscale.
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img_rgb, k=5):
    # Apply Gaussian blur to reduce noise and smooth texture.
    return cv2.GaussianBlur(img_rgb, (k, k), 0)
