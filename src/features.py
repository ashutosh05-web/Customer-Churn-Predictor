"""
Feature engineering: encodes categoricals and adds a few engineered
features that aren't in the raw data but tend to matter for churn.
"""
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = cfg["data"]["columns"]
    df = df.copy()

    # --- engineered features ---
    # Tenure buckets: churn often isn't linear with tenure, it's front-loaded
    df["tenure_bucket"] = pd.cut(
        df[cols["tenure"]],
        bins=[-1, 6, 12, 24, 48, 1000],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49mo+"],
    )

    # Average revenue per month of tenure — flags customers paying a lot
    # relative to how new they are (higher price sensitivity / risk)
    df["avg_monthly_spend"] = df[cols["revenue"]] / df[cols["tenure"]].replace(0, 1)

    # Count of add-on services subscribed (proxy for "stickiness")
    addon_cols = [c for c in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ] if c in df.columns]
    if addon_cols:
        df["num_addon_services"] = (df[addon_cols] == "Yes").sum(axis=1)

    # --- encode categoricals ---
    target_col = "churn_flag"
    id_col = cols["customer_id"]
    drop_cols = [id_col, cols["target"], target_col]

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_df = pd.get_dummies(feature_df, drop_first=True)

    feature_df[target_col] = df[target_col].values
    feature_df[id_col] = df[id_col].values
    feature_df[cols["revenue"]] = df[cols["revenue"]].values  # keep raw for business_metrics.py

    return feature_df


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["processed_path"])
    feat_df = engineer_features(df, cfg)
    out_path = cfg["data"]["processed_path"].replace(".csv", "_features.csv")
    feat_df.to_csv(out_path, index=False)
    print(f"Feature-engineered data saved to {out_path} ({feat_df.shape[1]} columns)")
