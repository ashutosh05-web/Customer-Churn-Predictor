"""
Translates raw churn probabilities into a business-ready output:
a Retention Priority Score that ranks customers not just by *likelihood*
of churning, but by *revenue impact* of losing them.

This is the piece most churn-prediction portfolio projects skip — they
stop at "here's the model's accuracy." This answers the question a
retention team actually asks: "who should we call first, and why?"
"""
import pandas as pd
import joblib
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_priority_scores(cfg: dict) -> pd.DataFrame:
    cols = cfg["data"]["columns"]
    bundle = joblib.load(cfg["model"]["output_path"])
    model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]

    feat_path = cfg["data"]["processed_path"].replace(".csv", "_features.csv")
    feat_df = pd.read_csv(feat_path)

    X = feat_df[feature_names]
    X_scaled = scaler.transform(X)
    churn_prob = model.predict_proba(X_scaled)[:, 1]

    revenue = feat_df[cols["revenue"]]
    factor = cfg["business"]["retention_offer_cost_factor"]

    out = pd.DataFrame({
        cols["customer_id"]: feat_df[cols["customer_id"]],
        "churn_probability": churn_prob.round(3),
        "monthly_revenue": revenue,
        "annual_revenue_at_risk": (revenue * 12 * churn_prob).round(2),
        "retention_priority_score": (churn_prob * revenue * factor).round(2),
    })
    out = out.sort_values("retention_priority_score", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    cfg = load_config()
    priority_df = compute_priority_scores(cfg)
    priority_df.to_csv("reports/retention_priority_list.csv", index=False)

    total_at_risk = priority_df["annual_revenue_at_risk"].sum()
    top10_at_risk = priority_df.head(int(len(priority_df) * 0.1))["annual_revenue_at_risk"].sum()

    print(f"Retention priority list saved to reports/retention_priority_list.csv")
    print(f"Total projected annual revenue at risk: ${total_at_risk:,.0f}")
    print(f"Revenue at risk in top 10% priority customers: ${top10_at_risk:,.0f} "
          f"({top10_at_risk/total_at_risk:.1%} of total risk in 10% of customers)")
    print("\nTop 5 customers to prioritize for retention outreach:")
    print(priority_df.head(5).to_string(index=False))
