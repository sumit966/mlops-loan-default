# End-to-End MLOps Pipeline for Loan Default Prediction

Production-grade MLOps pipeline that trains, tracks, versions, and serves a machine learning model for predicting loan default risk.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-111F68?style=for-the-badge&logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## 📋 Overview

A complete end-to-end MLOps pipeline that generates data, engineers features, trains multiple models, selects the best, and serves predictions via a FastAPI REST API. Built as a production-grade reference for ML deployment workflows.

## 🎯 Problem Statement

Financial institutions lose millions of dollars annually from loan defaults. Manual risk assessment is slow, inconsistent, and not scalable. Machine learning can predict default risk in milliseconds with higher accuracy than manual review.

## 💡 Solution

An end-to-end MLOps pipeline that:
- Generates or loads a loan dataset (5,000 records)
- Engineers 11 features from 8 raw inputs
- Trains 3 candidate models with stratified split
- Evaluates each model on 5 metrics
- Saves the best model (highest AUC-ROC)
- Serves predictions via FastAPI
- Runs tests on every push via GitHub Actions

## 🏗️ Architecture

Raw Data (CSV)
    ↓
Data Ingestion (data_ingestion.py)
    ↓
Feature Engineering (preprocess.py)
    ↓
Train/Test Split + StandardScaler
    ↓
Multi-Model Training (train.py)
    ↓
[Logistic Regression | Random Forest | XGBoost (Best)]
    ↓
Evaluate (evaluate.py) → Best Model Saved (joblib)
    ↓
FastAPI Endpoint (api/main.py)
    ↓
Docker + GitHub Actions CI/CD

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest, httpx |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

## 📦 Project Structure

mlops-loan-default/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── loan_data.csv
│   └── processed/
│       └── .gitkeep
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── metadata.json
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

## 🚀 Quick Start

### 1. Clone

git clone https://github.com/sumit966/mlops-loan-default.git
cd mlops-loan-default

### 2. Virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS/Linux:
python3 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Generate dataset

python src/data_ingestion.py

### 5. Train models

python src/train.py

### 6. Start API

uvicorn api.main:app --reload

### 7. Open Swagger UI

http://localhost:8000/docs

## 🔌 API Usage

### POST /predict

Request:
{
  "age": 30,
  "income": 50000,
  "loan_amount": 15000,
  "credit_score": 720,
  "employment_years": 5,
  "num_previous_loans": 1,
  "has_mortgage": 0,
  "num_dependents": 0
}

Response:
{
  "default_probability": 0.12,
  "risk_level": "Low",
  "recommendation": "Approve"
}

### Risk Levels

| Probability | Risk Level | Recommendation |
|-------------|------------|----------------|
| < 0.30 | Low | Approve |
| 0.30 - 0.60 | Medium | Approve with conditions |
| > 0.60 | High | Reject or require collateral |

### Other Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | API info |
| /health | GET | Health check |
| /docs | GET | Swagger UI |
| /redoc | GET | Alternative docs |

### cURL

curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"age\": 30, \"income\": 50000, \"loan_amount\": 15000, \"credit_score\": 720, \"employment_years\": 5}"

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.82 | 0.79 | 0.75 | 0.77 | 0.86 |
| Random Forest | 0.86 | 0.84 | 0.81 | 0.82 | 0.91 |
| XGBoost (Best) | 0.89 | 0.87 | 0.84 | 0.85 | 0.94 |

## 🧪 Testing

pytest tests/ -v

Covers:
- Root endpoint
- Health check
- /predict returns valid probability
- Invalid input (age < 18) returns 422

## 🐳 Docker

Build:
docker build -t mlops-loan-default .

Run:
docker run -p 8000:8000 mlops-loan-default

## 🔄 CI/CD Pipeline

Every push to main triggers GitHub Actions:
1. Install Python + dependencies
2. Generate synthetic data
3. Train all 3 models
4. Run pytest
5. Verify Docker build

See .github/workflows/ci.yml

## 🎓 Key Learnings

- Multi-model evaluation (Logistic Regression, Random Forest, XGBoost)
- Feature engineering improved AUC-ROC
- AUC-ROC > accuracy for imbalanced classification
- Stratified split preserves class distribution
- Pydantic validation at API boundary
- pytest + GitHub Actions catch regressions

## 🚀 Future Improvements

- MLflow experiment tracking
- DVC for data versioning
- Model monitoring (Evidently AI)
- SHAP explanations
- GCP Cloud Run deployment
- Model drift detection

## 📄 License

MIT License — see LICENSE file.

## 👤 Author

Sumit Raj
- M.Tech Applied AI & ML @ VNIT Nagpur
- Ex-Software Engineer Intern @ Salesforce
- GitHub: https://github.com/sumit966
- LinkedIn: https://www.linkedin.com/in/er-sumit-raj-/
- Portfolio: https://sumit966-github-io.vercel.app
- Email: info.sr0909@gmail.com

⭐ If you found this project useful, please consider giving it a star!
