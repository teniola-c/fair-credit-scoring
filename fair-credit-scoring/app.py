# 💳 Fair Credit Scoring Dashboard (Streamlit)

import os
import numpy as np
import pandas as pd
import altair as alt
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
    demographic_parity_difference, equalized_odds_difference
)


# Page Config

st.set_page_config(page_title="Fair Credit Scoring Dashboard", layout="wide")
st.title("💳 Fair Credit Scoring Dashboard — Interactive Demo")


# Paths / Load artifacts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_model.pkl")
TEST_PATH = os.path.join(BASE_DIR, "processed_test.csv")

@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        return None

@st.cache_data
def load_test(path=TEST_PATH):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ Could not load test data: {e}")
        return None

model = load_model()
test_df = load_test()


# Sidebar inputs

st.sidebar.header("Applicant Input")
loan_amnt = st.sidebar.number_input("Loan Amount", min_value=500, max_value=50000, value=10000, step=500)
term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
annual_inc = st.sidebar.number_input("Annual Income", min_value=1000, max_value=500000, value=50000, step=1000)
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
purpose = st.sidebar.selectbox("Purpose", ["credit_card", "debt_consolidation", "home_improvement",
                                           "major_purchase", "small_business", "other"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

st.sidebar.markdown("---")
st.sidebar.header("Fairness Settings")
sensitive_col = st.sidebar.selectbox(
    "Sensitive attribute (from test set)",
    options=(test_df.columns.tolist() if test_df is not None else ["home_ownership"]),
    index=0
)
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
run_mitigation = st.sidebar.checkbox("Show Mitigation (group thresholds)", value=False)

# Single-row input frame
input_df = pd.DataFrame([{
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "annual_inc": annual_inc,
    "dti": dti,
    "purpose": purpose,
    "home_ownership": home_ownership
}])


# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Applicant Input", "📊 Prediction & Risk",
    "🔎 Explainability", "⚖️ Fairness Dashboard"
])


# TAB 1 — Applicant Input

with tab1:
    st.subheader("Applicant Input")
    st.dataframe(input_df.T, use_container_width=True)


# Helper: safe_predict_proba

def safe_predict_proba(pipeline, df_input):
    """Return probability of default (higher = riskier)."""
    try:
        return pipeline.predict_proba(df_input)[:, 1]
    except Exception:
        if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            X_trans = pipeline.named_steps['preprocessor'].transform(df_input)
            clf = pipeline.named_steps.get('classifier')
            return clf.predict_proba(X_trans)[:, 1]
        raise


# TAB 2 — Prediction & Risk

with tab2:
    st.subheader("Prediction & Risk")
    if model is None:
        st.error("Model not loaded.")
    else:
        if st.button("Run Prediction"):
            try:
                prob = float(safe_predict_proba(model, input_df)[0])
                decision = "High Risk ⚠️" if prob >= threshold else "Low Risk ✅"

                c1, c2 = st.columns(2)
                c1.metric("Default Probability", f"{prob:.2%}")
                c2.metric(f"Decision (thr={threshold:.2f})", decision)

                gauge = pd.DataFrame({"prob": [prob]})
                chart = alt.Chart(gauge).mark_bar(size=30).encode(
                    x=alt.X('prob:Q', axis=alt.Axis(format='.0%', title='Probability')),
                    y=alt.Y('prob:Q', title=None),
                    color=alt.condition(alt.datum.prob > threshold,
                                        alt.value("#d7191c"), alt.value("#1a9641"))
                ).properties(width=600, height=60)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")


# TAB 3 — Explainability (SHAP)

with tab3:
    st.subheader("Explainability")
    if model is None:
        st.error("Model not loaded.")
    else:
        try:
            # Local SHAP
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                clf = model.named_steps['classifier']
                pre = model.named_steps.get('preprocessor', None)

                if pre is not None:
                    X_for_shap = pre.transform(input_df)
                    feature_names = pre.get_feature_names_out()
                else:
                    X_for_shap = input_df
                    feature_names = input_df.columns

                explainer = shap.Explainer(clf, X_for_shap, feature_names=feature_names)
                shap_values = explainer(X_for_shap)
            else:
                explainer = shap.Explainer(model, input_df)
                shap_values = explainer(input_df)

            # Local waterfall
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig, bbox_inches="tight", dpi=300)

        except Exception as e:
            st.error(f"Local SHAP explanation failed: {e}")

        # Global SHAP
        if test_df is not None and 'default' in test_df.columns:
            st.subheader("Global Feature Importance")
            try:
                sample = test_df.sample(min(500, len(test_df)), random_state=42)
                X_sample = sample.drop(columns=['default'])

                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    X_trans = model.named_steps['preprocessor'].transform(X_sample)
                    clf = model.named_steps['classifier']
                    expl = shap.Explainer(clf, X_trans,
                                          feature_names=model.named_steps['preprocessor'].get_feature_names_out())
                    sv = expl(X_trans)
                else:
                    expl = shap.Explainer(model, X_sample)
                    sv = expl(X_sample)

                fig, ax = plt.subplots()
                shap.summary_plot(sv, show=False)
                st.pyplot(fig, bbox_inches="tight", dpi=300)

            except Exception as e:
                st.warning(f"Global SHAP failed: {e}")


# TAB 4 — Fairness Dashboard

with tab4:
    st.subheader("Fairness Dashboard — Group Analysis")
    if test_df is None:
        st.error("No test set found.")
    elif model is None:
        st.error("Model not loaded.")
    else:
        df_eval = test_df.copy()

        if "default" not in df_eval.columns:
            st.error("Test set missing 'default' column.")
        elif sensitive_col not in df_eval.columns:
            st.error(f"Sensitive column '{sensitive_col}' not found.")
        else:
            y_true = df_eval['default'].values
            X_eval = df_eval.drop(columns=['default'])

            try:
                probs = safe_predict_proba(model, X_eval)
            except Exception as e:
                st.error(f"Could not score test set: {e}")
                st.stop()

            preds = (probs >= threshold).astype(int)
            group = df_eval[sensitive_col].fillna("MISSING").astype(str)

            df_groups = pd.DataFrame({
                "sensitive": group,
                "y_true": y_true,
                "y_pred": preds,
                "prob": probs
            })

            # Summary stats
            agg = df_groups.groupby("sensitive").agg(
                n=('y_true', 'count'),
                default_rate=('y_true', 'mean'),
                predicted_positive_rate=('y_pred', 'mean'),
                avg_prob=('prob', 'mean')
            ).reset_index().sort_values('n', ascending=False)

            st.dataframe(agg, use_container_width=True)

            # Bar chart
            chart = alt.Chart(agg).mark_bar().encode(
                x=alt.X('predicted_positive_rate:Q',
                        axis=alt.Axis(format='.0%', title='Predicted Positive Rate')),
                y=alt.Y('sensitive:N', sort='-x', title=sensitive_col),
                color=alt.condition(
                    alt.datum.predicted_positive_rate > agg['predicted_positive_rate'].mean(),
                    alt.value('#1a9641'), alt.value('#d7191c')
                )
            ).properties(height=400, width=700)
            st.altair_chart(chart, use_container_width=True)

            # Fairness metrics
            metrics = {
                'selection_rate': selection_rate,
                'tpr': true_positive_rate,
                'fpr': false_positive_rate
            }
            mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=preds, sensitive_features=group)
            st.subheader("Fairness Metrics by Group")
            st.dataframe(mf.by_group.reset_index().rename(columns={'index': sensitive_col}),
                         use_container_width=True)

            dp_diff = demographic_parity_difference(y_true, preds, sensitive_features=group)
            eo_diff = equalized_odds_difference(y_true, preds, sensitive_features=group)
            c1, c2 = st.columns(2)
            c1.metric("Demographic Parity Difference", f"{dp_diff:.3f}")
            c2.metric("Equalized Odds Difference", f"{eo_diff:.3f}")

            # Mitigation
            if run_mitigation:
                st.subheader("Mitigation: Group Thresholding")
                target = agg['predicted_positive_rate'].mean()
                thresholds = {}
                for g in df_groups['sensitive'].unique():
                    g_probs = df_groups.loc[df_groups['sensitive'] == g, 'prob']
                    thr = np.quantile(g_probs, 1 - target) if len(g_probs) > 1 else 1.0
                    thresholds[g] = float(thr)

                df_groups['pred_mitig'] = df_groups.apply(
                    lambda r: int(r['prob'] >= thresholds[r['sensitive']]), axis=1
                )

                mf_mit = MetricFrame(
                    metrics=metrics,
                    y_true=df_groups['y_true'],
                    y_pred=df_groups['pred_mitig'],
                    sensitive_features=df_groups['sensitive']
                )
                st.write("Per-group thresholds:")
                st.dataframe(pd.Series(thresholds).rename("threshold").reset_index()
                             .rename(columns={'index': sensitive_col}).head(30))

                st.subheader("Metrics After Mitigation")
                st.dataframe(mf_mit.by_group.reset_index()
                             .rename(columns={'index': sensitive_col}),
                             use_container_width=True)

                dp_diff_mit = demographic_parity_difference(
                    df_groups['y_true'], df_groups['pred_mitig'],
                    sensitive_features=df_groups['sensitive']
                )
                eo_diff_mit = equalized_odds_difference(
                    df_groups['y_true'], df_groups['pred_mitig'],
                    sensitive_features=df_groups['sensitive']
                )
                c1, c2 = st.columns(2)
                c1.metric("DP Difference (mitigated)", f"{dp_diff_mit:.3f}")
                c2.metric("EO Difference (mitigated)", f"{eo_diff_mit:.3f}")


# Footer

st.markdown("---")
st.markdown("👨‍💻 Built by **Teniola Kehinde** — [GitHub](https://github.com/teniola-c/fair-credit-scoring) | [Medium](https://medium.com)")



