import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io_helpers import load_joblib
from src.agents.explainer_agent import ExplainerAgent, ExplainerAgentConfig
from src.agents.risk_agent import RiskAgent, RiskAgentConfig
import joblib, json

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

st.set_page_config(page_title="Explanation-First Forecaster", layout="wide")
st.title("📈 Explanation-First Forecaster")

# Load model + meta
try:
    clf = load_joblib(MODELS / "calibrated_xgb.joblib")
    meta = json.loads((MODELS / "meta.json").read_text())
    features = meta["features"]
    st.success("Model loaded.")
except Exception as e:
    st.warning("Model not found. Run `python -m src.pipeline.train` first.")
    st.stop()

explainer = ExplainerAgent(ExplainerAgentConfig(topk=4))
explainer.fit(clf)
risk_agent = RiskAgent(RiskAgentConfig())

st.sidebar.header("Inference Input")
# For demo: users paste a single row of features (CSV) or upload a CSV
sample = ",".join(features)
txt = st.sidebar.text_area("Paste a single-row CSV with feature headings (use same order as trained features):", value=sample)

uploaded = st.sidebar.file_uploader("...or upload a CSV with feature columns", type=["csv"])

if uploaded:
    X = pd.read_csv(uploaded)
elif txt and txt != sample:
    try:
        # naive parse: assume header row present
        X = pd.read_csv(pd.compat.StringIO(txt))
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()
else:
    st.info("Provide inputs to score. For a full dashboard wired to live data, integrate DataAgent outputs.")
    st.stop()

missing = [c for c in features if c not in X.columns]
if missing:
    st.error(f"Missing features: {missing[:10]} ..."); st.stop()

proba = clf.predict_proba(X[features])[:,1]
st.subheader("Results")
rows = []
for i in range(len(X)):
    p_up = float(proba[i])
    top_pairs = explainer.explain(X.iloc[[i]][features])
    reason_text = explainer.to_reason_text(top_pairs, X.iloc[i])
    rows.append({"p_up": round(p_up,4), "reasons": reason_text})

st.dataframe(pd.DataFrame(rows))
st.caption("Probabilities are calibrated (sigmoid). Reasons are derived from SHAP top features.")
