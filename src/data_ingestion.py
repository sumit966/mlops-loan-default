"""Generate synthetic loan default dataset."""
import os
import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan default dataset."""
    np.random.seed(seed)
    data = {
        "age": np.random.randint(21, 65, n_samples),
        "income": np.random.randint(20000, 200000, n_samples),
        "loan_amount": np.random.randint(5000, 50000, n_samples),
        "credit_score": np.random.randint(300, 850, n_samples),
        "employment_years": np.random.randint(0, 40, n_samples),
        "num_previous_loans": np.random.randint(0, 10, n_samples),
        "has_mortgage": np.random.randint(0, 2, n_samples),
        "num_dependents": np.random.randint(0, 6, n_samples),
    }
    df = pd.DataFrame(data)

    # Realistic default logic
    risk_score = (
        (700 - df["credit_score"]).clip(lower=0) * 0.01
        + (df["loan_amount"] / df["income"]) * 0.5
        + (5 - df["employment_years"]).clip(lower=0) * 0.1
        + df["num_previous_loans"] * 0.05
    )
    probability = 1 / (1 + np.exp(-(risk_score - 2)))
    df["default"] = (np.random.random(n_samples) < probability).astype(int)

    return df


def load_data(input_path: str = "data/raw/loan_data.csv") -> pd.DataFrame:
    """Load or generate dataset."""
    if os.path.exists(input_path):
        return pd.read_csv(input_path)
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(input_path, index=False)
    print(f"[OK] Generated dataset: {input_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\nDefault rate: {df['default'].mean():.2%}")
