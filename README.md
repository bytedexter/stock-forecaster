# Explanation-First Forecaster (Agentic ML for Stocks)

**Goal:** Next-day direction + confidence with plain-English reasons (SHAP → templates).
**Agents:** Data → Model → Explainer → Risk → Report.

## Quickstart

```bash
# 1) Create a fresh venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train (uses ./data/tickers.csv)
python -m src.pipeline.train

# 4) Run Streamlit App
streamlit run streamlit_app.py
```

## What it does
- Downloads OHLCV from yfinance for tickers in `data/tickers.csv`.
- Builds lightweight technical features (RSI, MACD, ATR, ADX, returns).
- Trains `XGBClassifier`, calibrates probabilities, explains with SHAP.
- Converts SHAP to “reason codes” via an Explainer Agent.
- Risk Agent outputs stop/target via ATR multiples + conviction buckets.
- Report Agent can generate a simple PDF daily report in `./reports/`.

> ⚠️ Educational tool. Not investment advice.

## Project Structure
```
explanation_first_forecaster/
├── data/
│   └── tickers.csv
├── models/
├── reports/
├── src/
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── model_agent.py
│   │   ├── explainer_agent.py
│   │   ├── risk_agent.py
│   │   └── report_agent.py
│   ├── features/
│   │   └── tech_indicators.py
│   ├── utils/
│   │   └── io_helpers.py
│   └── pipeline/
│       └── train.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Minimal Experiment Plan (for the paper)
- **Task:** Next-day direction on NIFTY-50 universe.
- **Metrics:** Accuracy, F1(up), Precision@3, reliability (calibration) curve.
- **Ablations:** w/ and w/o Explainer Agent (or show reasons-only UX value),
  w/ and w/o calibration, and feature group ablations (TA vs returns only).
- **Leakage control:** Walk-forward time splits; indicators computed without look-ahead.
