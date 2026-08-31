"""
Cleans the raw dataset using the column mapping in config.yaml.
This is the ONLY file that needs to know about messy raw-data quirks
(e.g. TotalCharges being stored as a string with blanks in the Telco set).
Everything downstream works off the cleaned, standardized output.
"""
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_data(cfg: dict) -> pd.DataFrame:
    cols = cfg["data"]["columns"]
    df = pd.read_csv(cfg["data"]["raw_path"])

    # --- generic cleaning that applies to almost any tabular churn dataset ---
    df.columns = [c.strip() for c in df.columns]

    # TotalCharges in the Telco set is a string with some blank values for
    # brand-new customers (tenure=0). Coerce to numeric, fill with 0.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop exact duplicate rows and rows missing the customer id or target
    df = df.drop_duplicates()
    df = df.dropna(subset=[cols["customer_id"], cols["target"]])

    # Standardize target to binary 0/1 in a new column, keep original too
    positive = cfg["data"]["target_positive_label"]
    df["churn_flag"] = (df[cols["target"]] == positive).astype(int)

    return df


def validate_schema(df: pd.DataFrame, cfg: dict):
    """Fails loudly (with a clear message) if a swapped-in dataset is missing
    a column the pipeline depends on — this is what makes it safe to plug in
    your own dataset later instead of silently producing garbage results."""
    cols = cfg["data"]["columns"]
    required = [cols["customer_id"], cols["target"], cols["revenue"], cols["tenure"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Update the `columns:` mapping in config.yaml to match your dataset."
        )


if __name__ == "__main__":
    cfg = load_config()
    df = clean_data(cfg)
    validate_schema(df, cfg)
    df.to_csv(cfg["data"]["processed_path"], index=False)
    print(f"Cleaned data saved to {cfg['data']['processed_path']} ({len(df)} rows)")
    print(f"Churn rate: {df['churn_flag'].mean():.2%}")
