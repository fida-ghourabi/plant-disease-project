"""Train CNN and MobileNetV2 models and write DL metrics."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from .config import DATA_ROOT, IMG_SIZE_DL, MODELS_DIR, METRICS_DIR, SEED, ensure_dirs

def build_cnn(num_classes):
    # Simple CNN baseline for quick DL comparison.
    return models.Sequential([
        layers.Rescaling(1.0 / 255),
        # Lightweight conv stack to learn leaf texture patterns.
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

def train_dl(epochs=10, batch_size=16):
    # Train CNN baseline + Transfer Learning (simple baseline).
    ensure_dirs()
    tf.keras.utils.set_random_seed(SEED)

    # Build training/validation datasets from the folder structure.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE_DL,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_ROOT,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE_DL,
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    # Compute class weights from training set to mitigate imbalance.
    class_counts = np.zeros(len(class_names), dtype=np.int64)
    for _, batch_y in train_ds:
        for label in batch_y.numpy():
            class_counts[int(label)] += 1
    total = class_counts.sum()
    class_weight = {i: total / (len(class_names) * count) for i, count in enumerate(class_counts)}

    # Light augmentation to reduce overfitting.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    # Frozen ImageNet backbone for transfer learning.
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE_DL + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    tl_model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1.0 / 255),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(class_names), activation="softmax")
    ])

    tl_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train transfer-learning model.
    tl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight
    )

    # Train a small custom CNN for comparison.
    cnn = build_cnn(len(class_names))
    cnn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    cnn.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight
    )

    def evaluate_model(model, ds):
        # Run inference on the dataset and compute standard metrics.
        y_true = []
        y_pred = []
        for batch_x, batch_y in ds:
            preds = model.predict(batch_x, verbose=0)
            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(np.argmax(preds, axis=1).tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true, y_pred)
        return acc, f1, report, cm

    per_model_reports = {}
    # Evaluate both DL models on the validation set.
    for name, model in {"mobilenetv2": tl_model, "cnn": cnn}.items():
        acc, f1, report, cm = evaluate_model(model, val_ds)
        per_model_reports[name] = {"acc": acc, "f1_macro": f1, "report": report, "cm": cm}

        print("Model:", name)
        print("Accuracy:", acc)
        print("F1 macro:", f1)
        print(report)
        print(cm)

    best_name = max(per_model_reports, key=lambda n: per_model_reports[n]["f1_macro"])
    print("Best DL model:", best_name)

    os.makedirs(os.path.join(MODELS_DIR, "dl"), exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    # Save DL model artifacts and metrics.
    tl_model.save(os.path.join(MODELS_DIR, "dl", "mobilenetv2_tomato.keras"))
    cnn.save(os.path.join(MODELS_DIR, "dl", "cnn_tomato.keras"))

    with open(os.path.join(METRICS_DIR, "dl_report.txt"), "w", encoding="utf-8") as f:
        for name in ["mobilenetv2", "cnn"]:
            f.write("Model: " + name + "\n")
            f.write("Accuracy: " + str(per_model_reports[name]["acc"]) + "\n")
            f.write("F1 macro: " + str(per_model_reports[name]["f1_macro"]) + "\n")
            f.write(per_model_reports[name]["report"] + "\n")
            f.write(str(per_model_reports[name]["cm"]) + "\n\n")

        f.write("Best DL model: " + best_name + "\n")

if __name__ == "__main__":
    train_dl()