"""Streamlit dashboard for the Smart Outcome Predictor project."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model.model import load_dataset, train_classification, train_regression

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Smart Outcome Predictor")
st.caption("Ensemble Machine Learning • Course Completion Classification • Final Score Regression")
st.markdown("Predict student course completion and final performance using bagging, boosting, voting, and stacking models.")

with st.sidebar:
    st.header("📂 Dataset")
    uploaded = st.file_uploader("Upload student dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
    test_size = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5) / 100
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    run = st.button("🚀 Train & Evaluate Models", type="primary", use_container_width=True)

try:
    if uploaded is not None:
        df = load_dataset(uploaded)
    else:
        df = load_dataset()
    st.success(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    st.warning(f"{e}\n\nUpload `Smart_Outcome_Predictor_Dataset_5200.xlsx` to begin.")
    st.stop()

with st.expander("🔎 Preview dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.write("**Columns:**", ", ".join(map(str, df.columns)))

if "completion_status" not in df.columns or "final_score" not in df.columns:
    st.error("The dataset must include `completion_status` and `final_score` target columns.")
    st.stop()

if run or "classification_result" not in st.session_state or "regression_result" not in st.session_state:
    with st.spinner("Training and evaluating models. This may take a few minutes..."):
        try:
            st.session_state.classification_result = train_classification(df, test_size, int(random_state))
            st.session_state.regression_result = train_regression(df, test_size, int(random_state))
            st.session_state.trained_for_shape = df.shape
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

cls = st.session_state.classification_result
reg = st.session_state.regression_result

tab1, tab2, tab3 = st.tabs(["📘 Completion Prediction", "📈 Final Score Prediction", "🧠 Model Insights"])
with tab1:
    st.subheader("Course completion classification")
    st.caption("Target: completion_status (0 = Not Completed, 1 = Completed)")
    st.dataframe(cls["results"].round(4), use_container_width=True, hide_index=True)
    if "F1-score" in cls["results"].columns and cls["results"]["F1-score"].notna().any():
        chart = cls["results"].dropna(subset=["F1-score"]).set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]]
        st.bar_chart(chart)
    st.info(f"Best model by F1-score: **{cls['best_name']}**")
    st.markdown("#### Predict for a student")
    st.caption("Fields are taken from the first dataset row as defaults. Change them to explore a scenario.")
    with st.form("classification_form"):
        row = df.iloc[0]
        feature_values = {}
        columns = st.columns(2)
        features = cls["features"]
        for i, col in enumerate(features):
            value = row[col] if col in row.index else None
            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    default = float(numeric.median()) if pd.notna(numeric.median()) else 0.0
                    feature_values[col] = st.number_input(col, value=float(value) if pd.notna(value) else default, key=f"c_{col}")
                else:
                    choices = df[col].dropna().astype(str).unique().tolist()
                    current = str(value) if pd.notna(value) else (choices[0] if choices else "Unknown")
                    feature_values[col] = st.selectbox(col, choices or [current], index=choices.index(current) if current in choices else 0, key=f"c_{col}")
        submitted = st.form_submit_button("Predict completion")
    if submitted:
        sample = pd.DataFrame([feature_values], columns=features)
        prediction = int(cls["best_model"].predict(sample)[0])
        st.metric("Predicted outcome", "Completed 🎉" if prediction == 1 else "Not Completed")
        if hasattr(cls["best_model"], "predict_proba"):
            probability = float(cls["best_model"].predict_proba(sample)[0][1])
            st.progress(max(0.0, min(1.0, probability)), text=f"Estimated completion probability: {probability:.1%}")

with tab2:
    st.subheader("Final performance score regression")
    st.caption("Target: final_score (0–100, as specified in the project brief)")
    st.dataframe(reg["results"].round(4), use_container_width=True, hide_index=True)
    if "RMSE" in reg["results"].columns and reg["results"]["RMSE"].notna().any():
        chart = reg["results"].dropna(subset=["RMSE"]).set_index("Model")[["MAE", "RMSE"]]
        st.bar_chart(chart)
    st.info(f"Best model by lowest RMSE: **{reg['best_name']}**")
    st.markdown("#### Predict a final score")
    with st.form("regression_form"):
        row = df.iloc[0]
        feature_values = {}
        columns = st.columns(2)
        for i, col in enumerate(reg["features"]):
            value = row[col] if col in row.index else None
            with columns[i % 2]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    default = float(numeric.median()) if pd.notna(numeric.median()) else 0.0
                    feature_values[col] = st.number_input(col, value=float(value) if pd.notna(value) else default, key=f"r_{col}")
                else:
                    choices = df[col].dropna().astype(str).unique().tolist()
                    current = str(value) if pd.notna(value) else (choices[0] if choices else "Unknown")
                    feature_values[col] = st.selectbox(col, choices or [current], index=choices.index(current) if current in choices else 0, key=f"r_{col}")
        submitted = st.form_submit_button("Predict final score")
    if submitted:
        sample = pd.DataFrame([feature_values], columns=reg["features"])
        score = float(reg["best_model"].predict(sample)[0])
        st.metric("Predicted final score", f"{max(0, min(100, score)):.2f} / 100")
        st.caption("Displayed score is clipped to the expected 0–100 range.")

with tab3:
    st.subheader("How to interpret the results")
    st.markdown("""
    - **Bagging** trains models on bootstrap samples and combines their predictions to reduce variance.
    - **Boosting** builds learners sequentially so later learners can improve on earlier errors.
    - **Voting** combines predictions from multiple different estimators.
    - **Stacking** trains a meta-learner to combine base-model outputs.
    - For classification, compare recall and precision as well as F1-score and ROC-AUC. The best F1 model is selected for the demo.
    - For regression, lower MAE/RMSE and higher R² generally indicate stronger held-out performance. The demo selects the lowest-RMSE model.

    These results use a held-out test split and are educational, not a guarantee of future student outcomes. Check class balance, data quality, and fairness before real-world deployment.
    """)
    left, right = st.columns(2)
    with left:
        st.markdown("**Classification best model**")
        st.write(cls["best_name"])
    with right:
        st.markdown("**Regression best model**")
        st.write(reg["best_name"])
    st.download_button("⬇️ Download classification metrics (CSV)", cls["results"].to_csv(index=False).encode("utf-8"), "classification_model_comparison.csv", "text/csv")
    st.download_button("⬇️ Download regression metrics (CSV)", reg["results"].to_csv(index=False).encode("utf-8"), "regression_model_comparison.csv", "text/csv")

st.divider()
st.caption("Smart Outcome Predictor • Supervised Learning • Built with Streamlit and scikit-learn")
