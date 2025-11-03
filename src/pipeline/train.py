from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from joblib import dump
from src.agents.data_agent import DataAgent, DataAgentConfig
from src.features.tech_indicators import add_ta_features
from src.agents.model_agent import ModelAgent, ModelAgentConfig
from src.agents.explainer_agent import ExplainerAgent, ExplainerAgentConfig
from src.agents.risk_agent import RiskAgent, RiskAgentConfig
from src.agents.report_agent import ReportAgent, ReportAgentConfig
from src.utils.io_helpers import save_joblib, save_json

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

def build_dataset(ohlcvs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sym, df in ohlcvs.items():
        f = add_ta_features(df)
        f["symbol"] = sym
        rows.append(f)
    full = pd.concat(rows).sort_index()
    # Drop rows with NaNs from indicators
    full = full.dropna().copy()
    return full

def main():
    # Agents
    data_agent = DataAgent(DataAgentConfig())
    symbols = data_agent.load_tickers(DATA / "tickers.csv")
    ohlcvs = data_agent.download_ohlcv(symbols)
    if not ohlcvs:
        print("No data downloaded. Check tickers or internet."); return

    df = build_dataset(ohlcvs)
    features = [c for c in df.columns if c not in ["target_up","symbol"]]
    X = df[features]
    y = df["target_up"]

    # Time-aware split: last 15% as test
    n = len(df)
    split = int(n * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model_agent = ModelAgent(ModelAgentConfig())
    clf = model_agent.train_calibrated(X_train, y_train)

    # Basic eval
    proba_test = model_agent.predict_proba(X_test)
    metrics = model_agent.evaluate(X_test, y_test)
    metrics["brier"] = float(brier_score_loss(y_test, proba_test))
    print("Eval:", metrics)

    # Fit explainer
    explainer = ExplainerAgent(ExplainerAgentConfig(topk=4))
    explainer.fit(clf)

    # Save artifacts
    MODELS.mkdir(parents=True, exist_ok=True)
    save_joblib(clf, MODELS / "calibrated_xgb.joblib")
    save_json({"features": features}, MODELS / "meta.json")

    # Produce a small daily pick list (top-3 by p_up in last test day per symbol)
    # For demo: take last available day per symbol from X_test
    latest_idx = X_test.index.get_level_values(0) if isinstance(X_test.index, pd.MultiIndex) else X_test.index
    last_date = latest_idx.max()
    today_mask = (X_test.index == last_date)
    X_today = X_test[today_mask]
    df_today = df.loc[X_today.index]
    proba_today = model_agent.predict_proba(X_today)
    picks = []
    risk_agent = RiskAgent(RiskAgentConfig())

    for (i, row) in enumerate(X_today.to_dict("records")):
        sym = df_today["symbol"].iloc[i]
        last_close = df_today["Close"].iloc[i]
        atr = df_today["atr_14"].iloc[i]
        p_up = float(proba_today[i])
        top_pairs = explainer.explain(X_today.iloc[[i]])
        reason_text = explainer.to_reason_text(top_pairs, X_today.iloc[i])
        risk = risk_agent.stops_targets(last_close, atr)
        picks.append({
            "ticker": sym, "p_up": p_up, "conviction": risk_agent.conviction_bucket(p_up),
            "reason_text": reason_text, "stop_loss": risk["stop_loss"], "target": risk["target"],
            "last_close": round(float(last_close), 2)
        })

    picks = sorted(picks, key=lambda d: d["p_up"], reverse=True)[:3]

    # Report
    rep = ReportAgent(ReportAgentConfig(outdir=REPORTS))
    pdf_path = rep.save_simple_pdf(title="Explanation-First Forecaster — Top Picks", cards=picks, filename="daily_report.pdf")
    print("Saved report:", pdf_path)

if __name__ == "__main__":
    main()
