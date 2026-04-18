import os
import re
import shutil

ML_REPORT = "outputs/metrics/ml_report.txt"
DL_REPORT = "outputs/metrics/dl_report.txt"
MODELS_DIR = "models"
OUT_PATH = "outputs/metrics/best_overall.txt"


def _extract_ml_best_f1(report_text):
    match = re.search(r"Best model: (\w+)", report_text)
    best_name = match.group(1) if match else "rf"

    # Find the F1 macro test for the best model
    pattern = rf"Model: {best_name}.*?F1 macro test: ([0-9.]+)"
    match = re.search(pattern, report_text, flags=re.S)
    best_f1 = float(match.group(1)) if match else None

    return best_name, best_f1


def _extract_dl_best_f1(report_text):
    match = re.search(r"Best DL model: (\w+)", report_text)
    best_name = match.group(1) if match else "mobilenetv2"

    pattern = rf"Model: {best_name}.*?F1 macro: ([0-9.]+)"
    match = re.search(pattern, report_text, flags=re.S)
    best_f1 = float(match.group(1)) if match else None

    return best_name, best_f1


def compare_and_save():
    with open(ML_REPORT, "r", encoding="utf-8") as f:
        ml_text = f.read()
    with open(DL_REPORT, "r", encoding="utf-8") as f:
        dl_text = f.read()

    ml_name, ml_f1 = _extract_ml_best_f1(ml_text)
    dl_name, dl_f1 = _extract_dl_best_f1(dl_text)

    if ml_f1 is None or dl_f1 is None:
        raise ValueError("Could not parse F1 scores from reports.")

    if ml_f1 >= dl_f1:
        winner = "ml"
        winner_name = ml_name
        winner_f1 = ml_f1
        src_model = os.path.join(MODELS_DIR, "best_ml.joblib")
        if not os.path.exists(src_model):
            src_model = os.path.join(MODELS_DIR, "ml", "best_ml.joblib")
        dst_model = os.path.join(MODELS_DIR, "best_overall.joblib")
    else:
        winner = "dl"
        winner_name = dl_name
        winner_f1 = dl_f1
        src_model = os.path.join(MODELS_DIR, "dl", "best_dl.keras")
        dst_model = os.path.join(MODELS_DIR, "best_overall.keras")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Best overall: {winner} ({winner_name})\n")
        f.write(f"F1 macro: {winner_f1}\n")
        f.write(f"ML best F1: {ml_f1}\n")
        f.write(f"DL best F1: {dl_f1}\n")

    if os.path.exists(src_model):
        shutil.copy2(src_model, dst_model)

    return winner, winner_name, winner_f1


if __name__ == "__main__":
    compare_and_save()
