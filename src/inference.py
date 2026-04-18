import joblib
from .preprocess import load_image, resize_image, gaussian_blur
from .segmentation import hsv_mask
from .features import extract_features
from .config import IMG_SIZE_ML

def predict_image(model_path, img_path):
	data = joblib.load(model_path)
	model = data["model"]
	classes = data["classes"]

	img = load_image(img_path)
	if img is None:
		raise ValueError(f"Image introuvable ou invalide: {img_path}")
        

	img = resize_image(img, IMG_SIZE_ML)
	img_blur = gaussian_blur(img, k=5)
	mask = hsv_mask(img_blur)
	feat = extract_features(img_blur, mask=mask).reshape(1, -1)

	pred = model.predict(feat)[0]
	return classes[pred]

if __name__ == "__main__":
	model_path = "models/best_overall.joblib"
	img_path = "data/test/blight.png"
	print("Prediction:", predict_image(model_path, img_path))
