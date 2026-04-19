"""I/O helpers for listing classes and images."""

import os

# Supported image extensions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def list_classes(data_root):
    # List class folders under data root.
    return sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

def list_images(class_dir):
    # List image files within a class directory.
    return sorted([
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(IMG_EXTS)
    ])