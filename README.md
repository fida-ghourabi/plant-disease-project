# Plant Disease Detection (Tomato)

End-to-end project for tomato leaf disease classification: data preparation, classical ML, deep learning, comparison, and deployment (API + Streamlit).

## Project Structure

- data/: datasets (raw + extracted)
- src/: pipeline code (preprocessing, segmentation, features, training)
- notebooks/: analysis and visualization
- models/: saved ML/DL models
- outputs/: figures, metrics, logs
- api/: FastAPI service
- app/: Streamlit app

## Hosted Demo (Render)

- Streamlit app: https://streamlit-1-z0qp.onrender.com/
- FastAPI: https://plant-api-g8ab.onrender.com/

## Architecture

```
	    +-----------------------+
	    |  PlantVillage Dataset |
	    +-----------+-----------+
			|
			v
	+---------------+------------------+
	|  data/extractor.py (tomato only) |
	+---------------+------------------+
			|
			v
	+---------------+------------------+
	|   src/ (ML/DL pipeline)          |
	|   - preprocess.py                |
	|   - segmentation.py              |
	|   - features.py                  |
	|   - train_ml.py                  |
	|   - train_dl.py                  |
	|   - compare_models.py            |
	+---------------+------------------+
			|
			v
	+---------------+------------------+
	|  models/ & outputs/ (artifacts)  |
	+---------------+------------------+
			|
			v
	+---------------+------------------+
	|  api/main.py (FastAPI /predict)  |
	+---------------+------------------+
			|
			v
	+---------------+------------------+
	|  app/streamlit_app.py (UI)       |
	+----------------------------------+
```

Components summary:

- Data extraction: `data/extractor.py` copies tomato classes into `data/dataset_tomato/`.
- ML pipeline: preprocessing + HSV segmentation + feature extraction + classical ML training.
- DL pipeline: CNN baseline + MobileNetV2 transfer learning.
- Model selection: `src.compare_models` picks the best model and writes it to `models/`.
- Inference: FastAPI loads the best model for `/predict` and Streamlit consumes the API.

## Quick Start (Local)

1) Extract tomato classes

```
python data/extractor.py
```

2) Train ML models

```
python -m src.train_ml
```

3) Train DL models

```
python -m src.train_dl
```

4) Compare models and save the best overall model

```
python -m src.compare_models
```

This step writes the best model to `models/best_overall.joblib` (used by the API).

## Run API (FastAPI)

```
uvicorn api.main:app --reload
```

## Run App (Streamlit)

```
streamlit run app/streamlit_app.py
```

## Docker (Local)

```
docker compose up --build
```

## Notes

- Run `python -m src.compare_models` after training to save the best model for inference.
- The API expects `models/best_overall.joblib` to exist; otherwise it returns an error.
