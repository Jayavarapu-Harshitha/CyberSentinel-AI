"""CyberSentinel AI: research-only Streamlit dashboard.
Run from project root: python -m streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from cybersentinel.fusion import blend

st.set_page_config(page_title="CyberSentinel AI", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberSentinel AI | Intrusion Detection Research Prototype")
st.caption("Research and human review only • No live traffic capture or automatic blocking • Do not upload sensitive network data")

out = Path("artifacts")
summary = out / "training_summary.json"
if not summary.exists():
    st.error("Training artifacts were not found. Start Streamlit from the project root containing artifacts/.")
    st.stop()

info = json.loads(summary.read_text())
fusion = info.get("fusion")
if not fusion:
    st.error("Fusion configuration missing from training_summary.json.")
    st.stop()

@st.cache_resource
def load_models(tree_name):
    mlp_model = joblib.load(out / "mlp.joblib")
    tree_model = joblib.load(out / f"{tree_name}.joblib")
    anomaly_path = out / "isolation_forest.joblib"
    anomaly_model = joblib.load(anomaly_path) if anomaly_path.exists() else None
    return mlp_model, tree_model, anomaly_model

try:
    mlp, tree, anomaly = load_models(fusion["tree"])
    expected = list(mlp.feature_names_in_)
except Exception as exc:
    st.error(f"Could not load saved models: {type(exc).__name__}: {exc}")
    st.stop()

st.sidebar.header("Experiment configuration")
st.sidebar.write("Tree model:", fusion["tree"])
st.sidebar.write("MLP fusion weight:", fusion["mlp_weight"])
st.sidebar.write("Alert threshold: 0.50")
st.sidebar.caption("Models and weights are loaded from saved artifacts; this page does not retrain them.")
for report in ("comparison.csv", "anomaly.json", "ablation.json", "latency.json", "day_holdout.json"):
    path = out / report
    if path.exists():
        with st.sidebar.expander(report):
            if path.suffix == ".csv":
                st.dataframe(pd.read_csv(path), width='stretch')
            else:
                st.json(json.loads(path.read_text()))

st.subheader("1. Prepare your test data")
st.markdown("""Upload a **CSV containing extracted network-flow features** with the same column names used during model training. This is **not** a raw PCAP, Wireshark capture, or server-log uploader. Each row represents one network flow. You can start with the example below.""")

with st.expander(f"See all {len(expected)} required feature columns"):
    st.code("\n".join(expected), language="text")
    st.caption("Column names are read from the saved MLP model. Additional CSV columns are ignored for inference.")

sample_path = Path("sample_upload.csv")
if sample_path.exists():
    try:
        sample_df = pd.read_csv(sample_path, nrows=10)
        if not set(expected).issubset(sample_df.columns.str.strip()):
            raise ValueError("Existing sample_upload.csv does not contain every required model feature")
        sample_df.columns = sample_df.columns.str.strip()
        sample_df = sample_df[expected]
        st.download_button("⬇️ Download example CSV (10 flows)", sample_df.to_csv(index=False).encode("utf-8"), "cybersentinel_sample.csv", "text/csv")
    except Exception as exc:
        st.warning(f"Existing sample CSV cannot be used: {exc}")
elif (out / "test.pkl").exists():
    try:
        saved = pd.read_pickle(out / "test.pkl")
        if isinstance(saved, pd.DataFrame):
            sample_df = saved
        elif isinstance(saved, (tuple, list)) and len(saved) and isinstance(saved[0], pd.DataFrame):
            sample_df = saved[0]
        else:
            raise ValueError("Unexpected test.pkl structure")
        sample_df = sample_df.head(10)[expected]
        st.download_button("⬇️ Download example CSV (10 flows)", sample_df.to_csv(index=False).encode("utf-8"), "cybersentinel_sample.csv", "text/csv")
    except Exception as exc:
        st.warning(f"Could not create sample from test.pkl: {exc}")
else:
    st.info("Example CSV unavailable. Place sample_upload.csv in the project root or provide artifacts/test.pkl.")

st.markdown("**How to test:** Download the example CSV → upload it below → review predictions → download the results. For your own data, provide every required feature with matching definitions and units.")

# Evaluation dashboard: read-only; no model fitting or inference changes.
st.divider()
st.subheader("Model Performance & Explainability")
st.caption("Saved offline evaluations, not accuracy measured from the currently uploaded CSV. Scores depend on the evaluation split and dataset.")

comparison_path = out / "comparison.csv"
if comparison_path.exists():
    try:
        comparison = pd.read_csv(comparison_path)
        metric_cols = [c for c in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc") if c in comparison.columns]
        if "model" in comparison.columns and metric_cols:
            st.markdown("**Six Supervised Models + Two Fusion Strategies**")
            display_comparison = comparison[["model"] + metric_cols].copy()
            display_comparison[metric_cols] = display_comparison[metric_cols].round(4)
            st.dataframe(display_comparison, width='stretch', hide_index=True)
            if "f1" in comparison.columns:
                st.bar_chart(comparison.set_index("model")["f1"], x_label="Model", y_label="F1-score")
            st.caption("The comparison is from the saved evaluation set; a high score here does not prove performance on unseen networks or attacks.")
        else:
            st.warning("comparison.csv is present but its expected metric columns are missing.")
    except Exception as exc:
        st.warning(f"Unable to display model comparison: {exc}")
else:
    st.info("Model comparison unavailable: artifacts/comparison.csv not found.")

holdout_path = out / "day_holdout.json"
if holdout_path.exists():
    try:
        holdout = json.loads(holdout_path.read_text())
        st.markdown("**File-held-out generalization evaluation**")
        st.caption("Separate file-held-out evaluation; not the same split as the comparison above.")
        holdout_rows = []
        for name, details in holdout.get("models", {}).items():
            test_metrics = details.get("test", {})
            if "f1" in test_metrics:
                holdout_rows.append({
                    "model": name,
                    "holdout_f1": test_metrics.get("f1"),
                    "holdout_precision": test_metrics.get("precision"),
                    "holdout_recall": test_metrics.get("recall"),
                })
        if holdout_rows:
            holdout_df = pd.DataFrame(holdout_rows)
            if comparison_path.exists() and "comparison" in locals() and "model" in comparison.columns and "f1" in comparison.columns:
                holdout_df = holdout_df.merge(comparison[["model", "f1"]].rename(columns={"f1": "random_split_f1"}), on="model", how="left")
            st.dataframe(holdout_df.round(4), width='stretch', hide_index=True)
        else:
            st.info("No per-model holdout test metrics found in the saved report.")
        st.caption(f"Held-out file pattern: {holdout.get('held_out_file_pattern', 'Not specified')}; test rows: {holdout.get('test_rows', 'Not specified')}.")
        st.warning("A file-held-out test can reveal a generalization gap. It does not establish detection of completely new attack families or live-network effectiveness.")
    except Exception as exc:
        st.warning(f"Unable to display day-holdout evaluation: {exc}")
else:
    st.info("Day-holdout evaluation unavailable: artifacts/day_holdout.json not found.")

st.info(
    "Generalization finding: LightGBM achieved an F1-score of "
    "0.9968 on the random-split test and 0.7830 on the "
    "file-held-out evaluation. This highlights the importance "
    "of testing on traffic outside the original training files."
)
st.markdown("**Global feature importance (SHAP)**")
shap_image = out / "shap_feature_importance.png"
shap_csv = out / "shap_lightgbm.csv"
if shap_image.exists():
    st.image(str(shap_image), caption="Saved SHAP feature-importance plot for LightGBM; global model explanation, not an explanation of the uploaded records.", width='stretch')
elif shap_csv.exists():
    try:
        shap_df = pd.read_csv(shap_csv)
        st.dataframe(shap_df.head(20), width='stretch', hide_index=True)
    except Exception as exc:
        st.warning(f"Unable to display SHAP data: {exc}")
else:
    st.info("SHAP outputs unavailable. Expected artifacts/shap_feature_importance.png or artifacts/shap_lightgbm.csv.")
st.caption("SHAP describes how features influence the trained model; it does not prove causation or verify a security incident.")
st.divider()

st.subheader("2. Upload and analyze traffic")
file = st.file_uploader("Choose a network-flow feature CSV (maximum 1,000 rows)", type="csv")
if file is None:
    st.info("Upload the downloaded sample CSV to try the detector.")
    st.stop()

try:
    df = pd.read_csv(file, nrows=1001)
    df.columns = df.columns.str.strip()
    if df.empty:
        st.error("The CSV is empty. Add at least one network-flow record.")
        st.stop()
    if len(df) > 1000:
        st.error("Maximum 1,000 records per upload. Split larger files into smaller batches.")
        st.stop()
    if df.columns.duplicated().any():
        st.error("Duplicate column names found. Each feature must appear exactly once.")
        st.stop()
    missing = [name for name in expected if name not in df.columns]
    if missing:
        st.error(f"Missing {len(missing)} required feature column(s):")
        st.code("\n".join(missing), language="text")
        st.stop()

    x = df[expected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.isna().all(axis=1).any():
        st.error("At least one row has no usable numeric feature values. Check the CSV formatting.")
        st.stop()
    invalid_count = int(x.isna().sum().sum())
    if invalid_count:
        st.warning(f"Found {invalid_count} missing/non-numeric/infinite feature values. Saved model preprocessing will handle these where supported; verify input quality.")

    p1 = mlp.predict_proba(x)[:, 1]
    p2 = tree.predict_proba(x)[:, 1]
    p = blend(p1, p2, fusion["mlp_weight"])
    result = pd.DataFrame({
        "mlp_probability": p1,
        "tree_probability": p2,
        "fusion_probability": p,
        "prediction": np.where(p >= 0.5, "Review: possible attack", "Benign predicted"),
        "model_disagreement": (p1 >= 0.5) != (p2 >= 0.5),
    })
    result["review_priority"] = np.select(
        [p >= 0.9, p >= 0.7, p >= 0.5],
        ["High", "Medium", "Low"], default="No alert"
    )
    if anomaly is not None:
        scores = -anomaly["model"].score_samples(anomaly["imputer"].transform(x))
        result["unusual_flow_flag"] = scores >= anomaly["threshold"]

    st.subheader("3. Review predictions")
    a, b, c = st.columns(3)
    a.metric("Rows analyzed", len(result))
    b.metric("Predicted alerts", int((p >= 0.5).sum()))
    c.metric("Model disagreements", int(result["model_disagreement"].sum()))
    st.dataframe(result, width='stretch')
    st.download_button("⬇️ Download prediction and review CSV", result.to_csv(index=False).encode("utf-8"), "cybersentinel_predictions.csv", "text/csv")
    with st.expander("How to interpret these results"):
        st.markdown("""- **Fusion probability:** weighted combination of saved MLP and tree scores; **not a calibrated probability of a real attack**.
- **Prediction:** a score of 0.50 or above is flagged for review; it is not confirmation of malicious traffic.
- **Review priority:** a heuristic score band, **not** validated incident severity.
- **Model disagreement:** MLP and tree are on opposite sides of the 0.50 threshold.
- **Unusual flow flag:** Isolation Forest detected an unusual pattern, not proof of an attack.
- **Data limitations:** training and test traffic are drawn from CIC-IDS2017-style features; results may not generalize to new networks or attack types.""")
    st.info("Research model only. Human investigation is required before any security action. Do not upload private or production network traffic.")
except Exception as exc:
    st.error(f"Unable to analyze CSV: {type(exc).__name__}: {exc}")
