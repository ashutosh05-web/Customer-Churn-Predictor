"""
Simple web UI for the churn predictor — no terminal needed after this.

Run with:
    streamlit run app.py

Two tabs:
1. Single Customer — fill a form, get an instant prediction
2. Batch Upload — upload a CSV of many customers, get a ranked table
"""
import streamlit as st
import pandas as pd
import joblib
import yaml

from src.predict import clean_new_data, engineer_new_features, align_to_training_columns

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_everything():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    bundle = joblib.load(cfg["model"]["output_path"])
    return cfg, bundle


cfg, bundle = load_everything()
model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]
cols = cfg["data"]["columns"]

st.title("📉 Customer Churn Predictor")
st.caption(f"Model: {bundle['model_name']}  ·  Test F1: {bundle['metrics']['f1']:.3f}  ·  "
           f"ROC-AUC: {bundle['metrics']['roc_auc']:.3f}")

tab1, tab2 = st.tabs(["🔍 Single Customer", "📂 Batch Upload"])

# ------------------------------------------------------------------
# TAB 1 — single customer form
# ------------------------------------------------------------------
with tab1:
    st.subheader("Enter customer details")

    col_a, col_b = st.columns(2)
    with col_a:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col_b:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)))

    if st.button("Predict Churn", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "customerID": "MANUAL-ENTRY",
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        cleaned = clean_new_data(row)
        feat_df = engineer_new_features(cleaned, cfg)
        X = align_to_training_columns(feat_df, feature_names)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        priority_score = prob * monthly_charges

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Prediction", "⚠️ Will Churn" if prob >= 0.5 else "✅ Will Stay")
        c3.metric("Retention Priority Score", f"{priority_score:.1f}")

        st.progress(min(prob, 1.0))
        if prob >= 0.7:
            st.error("High risk — prioritize this customer for retention outreach.")
        elif prob >= 0.4:
            st.warning("Moderate risk — worth monitoring.")
        else:
            st.success("Low risk — likely to stay.")

# ------------------------------------------------------------------
# TAB 2 — batch CSV upload
# ------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV of customers")
    st.caption("Same columns as the original dataset, without the Churn column.")

    uploaded = st.file_uploader("Choose CSV file", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        cleaned = clean_new_data(raw)
        feat_df = engineer_new_features(cleaned, cfg)
        X = align_to_training_columns(feat_df, feature_names)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]

        result = pd.DataFrame({
            cols["customer_id"]: feat_df[cols["customer_id"]],
            "churn_probability": probs.round(3),
            "will_likely_churn": probs >= 0.5,
            "monthly_revenue": feat_df[cols["revenue"]],
            "retention_priority_score": (probs * feat_df[cols["revenue"]]).round(2),
        }).sort_values("retention_priority_score", ascending=False)

        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download predictions as CSV",
            result.to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )
