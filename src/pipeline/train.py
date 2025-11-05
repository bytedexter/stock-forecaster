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
    
    # Load data from Excel file instead of downloading
    ohlcvs = data_agent.load_from_excel(DATA / "Nifty_50.xlsx")
    if not ohlcvs:
        print("No data loaded from Excel. Check file path and format."); return

    df = build_dataset(ohlcvs)
    features = [c for c in df.columns if c not in ["target_up","symbol"]]
    X = df[features]
    y = df["target_up"]

    # Time-aware split: last 15% as test
    n = len(df)
    split = int(n * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model_agent = ModelAgent(ModelAgentConfig(
        sequence_length=60,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        epochs=50
    ))
    model_agent.train_calibrated(X_train, y_train)

    # Basic eval
    proba_test = model_agent.predict_proba(X_test)
    metrics = model_agent.evaluate(X_test, y_test)
    
    # Calculate Brier score with aligned lengths
    min_len = min(len(proba_test), len(y_test))
    metrics["brier"] = float(brier_score_loss(y_test.iloc[:min_len], proba_test[:min_len]))
    print("Eval:", metrics)

    # Fit explainer (now uses attention mechanism)
    explainer = ExplainerAgent(ExplainerAgentConfig(topk=4))
    explainer.fit(model_agent)

    # Save artifacts
    MODELS.mkdir(parents=True, exist_ok=True)
    # Save PyTorch model
    import torch
    torch.save({
        'model_state_dict': model_agent.model.state_dict(),
        'scaler': model_agent.scaler,
        'feature_names': model_agent.feature_names,
        'config': model_agent.cfg
    }, MODELS / "lstm_model.pth")
    save_json({"features": features}, MODELS / "meta.json")

    # Produce a small daily pick list (top-3 by p_up in last test day per symbol)
    # For LSTM, we need to work with sequences, so let's get predictions for the full test set
    # and then extract the last day per symbol
    
    # Get all predictions for test set
    proba_test_full = model_agent.predict_proba(X_test)
    
    # Align indices
    min_len = min(len(proba_test_full), len(X_test))
    X_test_aligned = X_test.iloc[:min_len]
    df_test_aligned = df.loc[X_test_aligned.index]
    
    # Group by symbol and get last entry for each
    picks = []
    risk_agent = RiskAgent(RiskAgentConfig())
    
    if "symbol" in df_test_aligned.columns:
        symbols = df_test_aligned["symbol"].unique()
        
        for sym in symbols:
            # Get last occurrence of this symbol in test set
            sym_mask = df_test_aligned["symbol"] == sym
            sym_indices = df_test_aligned[sym_mask].index
            
            if len(sym_indices) == 0:
                continue
                
            last_idx = sym_indices[-1]
            row_position = df_test_aligned.index.get_loc(last_idx)
            
            # Get data for this symbol's last entry
            last_close = df_test_aligned.loc[last_idx, "Close"]
            atr = df_test_aligned.loc[last_idx, "atr_14"]
            p_up = float(proba_test_full[row_position])
            
            # Get explanation using a window of data ending at this point
            window_start = max(0, row_position - 60)
            X_window = X_test_aligned.iloc[window_start:row_position+1]
            
            try:
                top_pairs = explainer.explain(X_window)
                reason_text = explainer.to_reason_text(top_pairs, X_test_aligned.iloc[row_position])
            except Exception as e:
                print(f"Warning: Could not generate explanation for {sym}: {e}")
                reason_text = "LSTM model prediction based on 60-day sequence."
            
            risk = risk_agent.stops_targets(last_close, atr)
            picks.append({
                "ticker": sym, "p_up": p_up, "conviction": risk_agent.conviction_bucket(p_up),
                "reason_text": reason_text, "stop_loss": risk["stop_loss"], "target": risk["target"],
                "last_close": round(float(last_close), 2)
            })
    
    picks = sorted(picks, key=lambda d: d["p_up"], reverse=True)[:3]

    # Report
    rep = ReportAgent(ReportAgentConfig(outdir=REPORTS))
    pdf_path = rep.save_simple_pdf(title="LSTM with Attention — Top Picks", cards=picks, filename="daily_report.pdf")
    print("Saved report:", pdf_path)

if __name__ == "__main__":
    main()
