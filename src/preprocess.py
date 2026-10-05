"""Preprocess and feature engineer the loan dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "age", "income", "loan_amount", "credit_score",
    "employment_years", "num_previous_loans",
    "has_mortgage", "num_dependents"
]
TARGET_COL = "default"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["income"]
    df["credit_per_year"] = df["credit_score"] / (df["employment_years"] + 1)
    df["risk_score"] = (
        (700 - df["credit_score"]).clip(lower=0) * 0.01
        + df["loan_to_income"] * 0.5
    )
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data and scale features."""
    df = engineer_features(df)
    features = FEATURE_COLS + ["loan_to_income", "credit_per_year", "risk_score"]

    X = df[features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features


if __name__ == "__main__":
    from data_ingestion import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, features = split_and_scale(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features: {features}")
