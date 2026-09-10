import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="centered",
)

# All artifact files are expected in the SAME directory as this app.py file
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# Load model + dependencies (cached so it only loads once)
# =========================================================
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(APP_DIR, "xgb_model.json"))

    with open(os.path.join(APP_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    with open(os.path.join(APP_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)

    with open(os.path.join(APP_DIR, "preprocessing_params.json")) as f:
        preprocessing = json.load(f)

    return model, feature_names, metadata, preprocessing


try:
    model, feature_names, metadata, preprocessing = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    load_error = str(e)


# =========================================================
# Header
# =========================================================
st.title("💳 Credit Default Risk Predictor")
st.write(
    "Estimate the probability a borrower defaults within 2 years, "
    "using the trained XGBoost model from the *Give Me Some Credit* project."
)

if not artifacts_loaded:
    st.error(
        "Could not load model artifacts.\n\n"
        f"Details: {load_error}\n\n"
        "Make sure `xgb_model.json`, `feature_names.json`, `model_metadata.json`, "
        "and `preprocessing_params.json` are placed in the **same folder** as "
        "`app.py` (no subfolders)."
    )
    st.stop()

st.divider()

# =========================================================
# Sidebar — model info
# =========================================================
with st.sidebar:
    st.header("Model info")
    st.metric("ROC-AUC", f"{metadata.get('roc_auc', 0):.4f}")
    if "pr_auc" in metadata:
        st.metric("PR-AUC", f"{metadata['pr_auc']:.4f}")
    st.metric("Decision threshold", f"{metadata.get('best_threshold', 0.5):.2f}")

    with st.expander("Cost assumptions behind threshold"):
        st.write(f"Cost of a missed default (FN): **{metadata.get('cost_false_negative', 'N/A')}**")
        st.write(f"Cost of a wrongly rejected borrower (FP): **{metadata.get('cost_false_positive', 'N/A')}**")

    with st.expander("Best model hyperparameters"):
        st.json(metadata.get("best_params", {}))

    st.caption(
        "Threshold was chosen to minimize total business cost, not to "
        "maximize accuracy — see the training notebook for details."
    )

# =========================================================
# Input form
# =========================================================
st.subheader("Borrower information")
st.caption("Enter the borrower's details below, then click **Predict**.")

with st.form("borrower_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Borrower's age in years.",
        )

        credit_utilization = st.slider(
            "Credit utilization ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01,
            help="Total balance on credit cards/lines divided by total credit limits (0 = none used, 1 = maxed out).",
        )

        debt_ratio = st.number_input(
            "Debt ratio",
            min_value=0.0,
            value=0.30,
            step=0.01,
            format="%.4f",
            help="Monthly debt payments divided by monthly gross income.",
        )

        monthly_income = st.number_input(
            "Monthly income ($)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            help="Gross monthly income.",
        )

        open_credit_lines = st.number_input(
            "Open credit lines / loans",
            min_value=0,
            value=5,
            step=1,
            help="Number of open loans and credit lines (e.g. credit cards, car loans).",
        )

    with col2:
        real_estate_loans = st.number_input(
            "Real estate loans / lines",
            min_value=0,
            value=1,
            step=1,
            help="Number of mortgage / real estate loans.",
        )

        dependents = st.number_input(
            "Number of dependents",
            min_value=0,
            value=0,
            step=1,
            help="Number of dependents excluding the borrower.",
        )

        late_30_59_days = st.number_input(
            "Times 30-59 days past due",
            min_value=0,
            value=0,
            step=1,
            help="Number of times 30-59 days late (not worse) in the last 2 years.",
        )

        late_60_89_days = st.number_input(
            "Times 60-89 days past due",
            min_value=0,
            value=0,
            step=1,
            help="Number of times 60-89 days late (not worse) in the last 2 years.",
        )

        late_90_days = st.number_input(
            "Times 90+ days late",
            min_value=0,
            value=0,
            step=1,
            help="Number of times 90 or more days late.",
        )

    submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

# =========================================================
# Prediction
# =========================================================
if submitted:
    # Build a single-row dataframe with columns in the exact order the model expects
    input_dict = {
        "credit_utilization": credit_utilization,
        "age": age,
        "late_30_59_days": late_30_59_days,
        "debt_ratio": debt_ratio,
        "monthly_income": monthly_income,
        "open_credit_lines": open_credit_lines,
        "late_90_days": late_90_days,
        "real_estate_loans": real_estate_loans,
        "late_60_89_days": late_60_89_days,
        "dependents": dependents,
    }

    # Reorder/validate against the saved feature list
    missing = [f for f in feature_names if f not in input_dict]
    if missing:
        st.error(f"Missing expected features: {missing}. Check feature_names.json vs this form.")
        st.stop()

    input_df = pd.DataFrame([input_dict])[feature_names]

    proba = model.predict_proba(input_df)[0, 1]
    threshold = metadata.get("best_threshold", 0.5)
    is_risky = proba >= threshold

    st.divider()
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted default probability", f"{proba:.1%}")
    with c2:
        st.metric("Decision threshold", f"{threshold:.1%}")

    if is_risky:
        st.error(
            f"⚠️ **High risk** — predicted probability ({proba:.1%}) is at or above "
            f"the {threshold:.0%} cost-optimized threshold. Flagged for further review."
        )
    else:
        st.success(
            f"✅ **Low risk** — predicted probability ({proba:.1%}) is below "
            f"the {threshold:.0%} cost-optimized threshold."
        )

    st.progress(min(float(proba), 1.0))

    with st.expander("Raw input sent to the model"):
        st.dataframe(input_df, use_container_width=True)

    st.caption(
        "This tool is a portfolio demonstration and not a substitute for a real "
        "underwriting decision or financial advice."
    )
