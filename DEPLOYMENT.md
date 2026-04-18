# Deployment Guide

## Option A: Render (API)

1) Create a new Web Service
2) Build command: `pip install -r requirements.txt`
3) Start command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
4) Set environment variables if needed

## Option B: Streamlit Community Cloud

1) Connect GitHub repo
2) App file: `app/streamlit_app.py`
3) Requirements: `requirements.txt`

## Option C: Railway

1) New project from GitHub
2) Start command: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
