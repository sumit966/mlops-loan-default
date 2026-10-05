"""Train multiple models with experiment tracking."""
import os
import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

from data_ingestion import load_data
from preprocess import split_and_scale


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=5, random_state=42,
        eval_metric="logloss", verbosity=0
    ),
}


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
    }


def train_all_models():
    """Train all models and save best."""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, features = split_and_scale(df)

    os.makedirs("models", exist_ok=True)
    best_auc = 0
    best_model_name = None
    all_results = {}

    for name, model in MODELS.items():
        print(f"\n[Training] {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        all_results[name] = metrics

        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_model_name = name
            joblib.dump(model, "models/best_model.joblib")
            joblib.dump(scaler, "models/scaler.joblib")

    with open("models/metadata.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "auc_roc": best_auc,
            "features": features,
            "all_results": all_results,
        }, f, indent=2)

    print(f"\n[Best Model] {best_model_name} (AUC-ROC: {best_auc:.4f})")
    print("[OK] Saved to models/best_model.joblib")


if __name__ == "__main__":
    train_all_models()
