# Plant Disease Detection (Tomato)

Professional end-to-end project: data preparation, classical ML, deep learning, and deployment.

## Project Structure

- data/: datasets (raw + extracted)
- src/: pipeline code (preprocessing, segmentation, features, training)
- notebooks/: analysis and visualization
- models/: saved ML/DL models
- outputs/: figures, metrics, logs
- api/: FastAPI service
- app/: Streamlit app

## Quick Start

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

## Run API (FastAPI)

```
uvicorn api.main:app --reload
```

## Run App (Streamlit)

```
streamlit run app/streamlit_app.py
```
