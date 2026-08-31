"""
Use the already-trained model to score NEW customers (ones not in the
original training data). This is the "use the model" step — everything
before this was training.

Usage:
    python src/predict.py data/raw/new_customers.csv

The input CSV must have the same raw columns as the original dataset,
EXCEPT the Churn column (since we don't know it yet — that's what we're
predicting).
"""
import sys
import pandas as pd
import joblib
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_new_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning logic as data_prep.py, minus anything that needs the target column."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def engineer_new_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Same feature engineering as features.py, minus anything that needs the target column."""
    cols = cfg["data"]["columns"]
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df[cols["tenure"]],
        bins=[-1, 6, 12, 24, 48, 1000],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"],
    )
    df["avg_monthly_spend"] = df[cols["revenue"]] / df[cols["tenure"]].replace(0, 1)

    addon_cols = [c for c in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ] if c in df.columns]
    if addon_cols:
        df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)

    id_col = cols["customer_id"]
    feature_df = df.drop(columns=[id_col])
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_df[id_col] = df[id_col].values
    feature_df[cols["revenue"]] = df[cols["revenue"]].values
    return feature_df


def align_to_training_columns(feature_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """New customers might not trigger every dummy column the training data did
    (e.g. no customer in this batch has InternetService='Fiber optic'). Add any
    missing columns as 0, and drop anything extra, so shapes match exactly."""
    for col in feature_names:
        if col not in feature_df.columns:
            feature_df[col] = 0
    return feature_df[feature_names]


def main(input_path: str):
    cfg = load_config()
    cols = cfg["data"]["columns"]
    bundle = joblib.load(cfg["model"]["output_path"])
    model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]

    raw = pd.read_csv(input_path)
    cleaned = clean_new_data(raw)
    feat_df = engineer_new_features(cleaned, cfg)

    X = align_to_training_columns(feat_df, feature_names)
    X_scaled = scaler.transform(X)
    churn_prob = model.predict_proba(X_scaled)[:, 1]

    revenue = feat_df[cols["revenue"]]
    out = pd.DataFrame({
        cols["customer_id"]: feat_df[cols["customer_id"]],
        "churn_probability": churn_prob.round(3),
        "will_likely_churn": (churn_prob >= 0.5),
        "monthly_revenue": revenue,
        "retention_priority_score": (churn_prob * revenue).round(2),
    }).sort_values("retention_priority_score", ascending=False)

    print(out.to_string(index=False))
    out.to_csv("reports/new_customer_predictions.csv", index=False)
    print(f"\nSaved to reports/new_customer_predictions.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_new_customers.csv>")
        sys.exit(1)
    main(sys.argv[1])
