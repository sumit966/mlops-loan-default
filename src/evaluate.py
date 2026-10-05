"""Evaluate saved model against test data."""
import json
import joblib
from data_ingestion import load_data
from preprocess import split_and_scale
from train import evaluate_model


def evaluate_saved_model():
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")

    with open("models/metadata.json") as f:
        metadata = json.load(f)

    df = load_data()
    _, X_test, _, y_test, _, _ = split_and_scale(df)

    metrics = evaluate_model(model, X_test, y_test)

    print(f"\n[Evaluation] Test set performance:")
    print(f"   Model: {metadata['best_model']}")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    evaluate_saved_model()
