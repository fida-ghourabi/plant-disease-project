import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from .build_dataset import build_ml_dataset
from .config import MODELS_DIR, METRICS_DIR, SEED, ensure_dirs

def train():
    # Train and select the best classical ML model
    ensure_dirs()
    X, y, classes = build_ml_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    classes_arr = np.array(sorted(set(y_train)))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_arr,
        y=y_train
    )
    class_weight_dict = {cls: weight for cls, weight in zip(classes_arr, class_weights)}

    models = {
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight=class_weight_dict
        ),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight=class_weight_dict))
        ]),
        "knn": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }

    best_name, best_model, best_f1 = None, None, -1
    per_model_reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        f1_train = f1_score(y_train, pred_train, average="macro")
        f1_test = f1_score(y_test, pred_test, average="macro")
        report = classification_report(y_test, pred_test, target_names=classes)
        cm = confusion_matrix(y_test, pred_test)
        per_model_reports[name] = {
            "f1_macro_train": f1_train,
            "f1_macro_test": f1_test,
            "report": report,
            "cm": cm
        }

        print("Model:", name)
        print("F1 macro train:", f1_train)
        print("F1 macro test:", f1_test)
        print(report)
        print(cm)

        if f1_test > best_f1:
            best_f1 = f1_test
            best_name = name
            best_model = model

    pred = best_model.predict(X_test)
    report = classification_report(y_test, pred, target_names=classes)
    cm = confusion_matrix(y_test, pred)

    print("Best model:", best_name)
    print(report)
    print(cm)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    joblib.dump({"model": best_model, "classes": classes}, os.path.join(MODELS_DIR, "best_ml.joblib"))
    with open(os.path.join(METRICS_DIR, "ml_report.txt"), "w", encoding="utf-8") as f:
        f.write("Class weights (balanced): " + str(class_weight_dict) + "\n\n")
        for name in models:
            f.write("Model: " + name + "\n")
            f.write("F1 macro train: " + str(per_model_reports[name]["f1_macro_train"]) + "\n")
            f.write("F1 macro test: " + str(per_model_reports[name]["f1_macro_test"]) + "\n")
            f.write(per_model_reports[name]["report"] + "\n")
            f.write(str(per_model_reports[name]["cm"]) + "\n\n")

        f.write("Best model: " + best_name + "\n")
        f.write(report + "\n")
        f.write(str(cm))

if __name__ == "__main__":
    train()