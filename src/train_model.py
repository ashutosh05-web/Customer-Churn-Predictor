"""
Trains and compares multiple models, picks the best one by the metric
set in config.yaml (default: F1, since accuracy is misleading on
imbalanced churn data), and saves it to models/.
"""
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


MODEL_REGISTRY = {
    "logistic_regression": lambda rs: LogisticRegression(max_iter=1000, random_state=rs),
    "random_forest": lambda rs: RandomForestClassifier(n_estimators=200, random_state=rs),
    "xgboost": lambda rs: xgb.XGBClassifier(
        n_estimators=200, eval_metric="logloss", random_state=rs
    ),
}


def prepare_xy(feat_df: pd.DataFrame, cfg: dict):
    cols = cfg["data"]["columns"]
    drop_cols = ["churn_flag", cols["customer_id"]]
    X = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])
    y = feat_df["churn_flag"]
    return X, y


def train_and_compare(X, y, cfg: dict):
    mcfg = cfg["model"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=mcfg["test_size"], random_state=mcfg["random_state"], stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if mcfg["use_smote"]:
        sm = SMOTE(random_state=mcfg["random_state"])
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)

    results = {}
    fitted_models = {}
    for name in mcfg["models_to_compare"]:
        model = MODEL_REGISTRY[name](mcfg["random_state"])
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        fitted_models[name] = model

    return results, fitted_models, scaler, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    cfg = load_config()
    feat_path = cfg["data"]["processed_path"].replace(".csv", "_features.csv")
    feat_df = pd.read_csv(feat_path)

    X, y = prepare_xy(feat_df, cfg)
    results, fitted_models, scaler, X_train, X_test, y_train, y_test = train_and_compare(X, y, cfg)

    print("\n=== Model comparison (test set) ===")
    for name, m in results.items():
        print(f"{name:20s} | precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}")

    metric = cfg["model"]["primary_metric"]
    best_name = max(results, key=lambda n: results[n][metric])
    best_model = fitted_models[best_name]
    print(f"\nBest model by {metric}: {best_name}")

    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "model_name": best_name,
        "metrics": results[best_name],
    }, cfg["model"]["output_path"])
    print(f"Saved best model to {cfg['model']['output_path']}")
