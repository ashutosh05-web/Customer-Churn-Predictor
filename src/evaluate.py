"""
Generates evaluation plots (confusion matrix, ROC curve, feature importance
via SHAP) saved to reports/figures/ for the README.
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import shap


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    mcfg = cfg["model"]
    cols = cfg["data"]["columns"]

    feat_path = cfg["data"]["processed_path"].replace(".csv", "_features.csv")
    feat_df = pd.read_csv(feat_path)

    drop_cols = ["churn_flag", cols["customer_id"]]
    X = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])
    y = feat_df["churn_flag"]

    bundle = joblib.load(mcfg["output_path"])
    model, scaler = bundle["model"], bundle["scaler"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=mcfg["test_size"], random_state=mcfg["random_state"], stratify=y
    )
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {bundle['model_name']}")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/roc_curve.png", dpi=150)
    plt.close()

    # --- SHAP explainability (this is the differentiator most tutorial
    # projects skip — shows WHY the model predicts churn, per customer,
    # not just a global feature_importances_ bar chart) ---
    try:
        if bundle["model_name"] in ("random_forest", "xgboost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            explainer = shap.LinearExplainer(model, X_train.pipe(lambda d: scaler.transform(d)))
            sv = explainer.shap_values(X_test_scaled)

        plt.figure()
        shap.summary_plot(sv, X_test, feature_names=bundle["feature_names"], show=False)
        plt.tight_layout()
        plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP summary plot saved.")
    except Exception as e:
        print(f"SHAP plot skipped ({e})")

    print(f"Evaluation plots saved to reports/figures/")
    print(f"Test set — precision={bundle['metrics']['precision']:.3f}, "
          f"recall={bundle['metrics']['recall']:.3f}, f1={bundle['metrics']['f1']:.3f}, "
          f"roc_auc={bundle['metrics']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
