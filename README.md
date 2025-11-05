# Explanation-First Forecaster (Agentic ML for Stocks)

**Goal:** Next-day direction + confidence with plain-English reasons (SHAP  templates).
**Agents:** Data  Model  Explainer  Risk  Report.

## Quickstart

```bash
# 1) Create a fresh venv (recommended)
python -m venv .venv
# Activate:
#   Windows PowerShell: .venv\Scripts\Activate.ps1
#   Windows CMD: .venv\Scripts\activate.bat
#   Linux/Mac: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Prepare data
# Place your Nifty_50.xlsx file in the ./data/ folder
# The Excel file should contain one sheet per ticker with columns: Date, Open, High, Low, Close, Volume

# 4) Train the model and generate report
python -m src.pipeline.train

# 5) (Optional) Run Streamlit App for interactive dashboard
streamlit run streamlit_app.py
```

## How to Run the Project - Complete Guide

### Step 1: Environment Setup
1. Clone the repository and navigate to the project directory
2. Create and activate a virtual environment (recommended)
3. Install all dependencies: `pip install -r requirements.txt`

### Step 2: Data Preparation
1. Place `Nifty_50.xlsx` in the `./data/` folder
2. The Excel file structure:
   - **One sheet per ticker** (e.g., "RELIANCE", "TCS", "INFY")
   - **Columns required:** `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
   - The sheet name becomes the ticker symbol

### Step 3: Run the Training Pipeline
Execute: `python -m src.pipeline.train`

This will:
1. **Load Data** - Read OHLCV data from `Nifty_50.xlsx` using DataAgent
2. **Feature Engineering** - Add technical indicators (RSI, MACD, ATR, ADX, lagged returns)
3. **Model Training** - Train calibrated XGBoost classifier on 85% of data
4. **Evaluation** - Test on remaining 15% (time-aware split)
5. **Generate Explanations** - Use SHAP to create human-readable reasons
6. **Risk Analysis** - Calculate stop-loss and target prices using ATR multiples
7. **Create Report** - Generate PDF report with top 3 stock picks

### Step 4: View Results
- **PDF Report**: Located in `./reports/daily_report.pdf`
  - Contains top 3 stock picks ranked by probability of upward movement
  - Includes conviction levels, price targets, stop-losses, and plain-English explanations
  
- **Model Artifacts**: Saved in `./models/`
  - `calibrated_xgb.joblib` - Trained model
  - `meta.json` - Feature metadata

- **Console Output**: Shows evaluation metrics (accuracy, precision, recall, F1, Brier score)

### Step 5: (Optional) Interactive Dashboard
Run: `streamlit run streamlit_app.py`
- Opens web interface for manual scoring and exploration

## What it does
- Loads OHLCV data from Excel file (`Nifty_50.xlsx`) using DataAgent
- Builds lightweight technical features (RSI, MACD, ATR, ADX, returns) via FeatureAgent
- Trains `XGBClassifier` with probability calibration via ModelAgent
- Explains predictions using SHAP  plain-English reasons via ExplainerAgent
- Calculates stop-loss/target prices and conviction buckets via RiskAgent
- Generates PDF report with top picks and visualizations via ReportAgent

>  Educational tool. Not investment advice.

## Project Structure
```
stock-forecaster/
 data/
    tickers.csv
    Nifty_50.xlsx         Place your Excel file here
 models/
    calibrated_xgb.joblib     Generated after training
    meta.json                 Generated after training
 reports/
    daily_report.pdf          Generated after training
 src/
    agents/
       data_agent.py         Loads data from Excel
       model_agent.py        Trains XGBoost model
       explainer_agent.py    Generates SHAP explanations
       risk_agent.py         Calculates risk metrics
       report_agent.py       Creates PDF reports
    features/
       tech_indicators.py    Technical indicator calculations
    utils/
       io_helpers.py
    pipeline/
        train.py              Main training pipeline
 streamlit_app.py
 requirements.txt
 README.md
```

## Data Format Requirements

Your `Nifty_50.xlsx` file should follow this structure:
- **Multiple sheets**: One sheet per stock ticker
- **Sheet names**: Use ticker symbols (e.g., "RELIANCE", "TCS", "INFY", "HDFCBANK")
- **Columns in each sheet**:
  - `Date` - Trading date
  - `Open` - Opening price
  - `High` - Highest price
  - `Low` - Lowest price
  - `Close` - Closing price
  - `Volume` - Trading volume

## Pipeline Flow

```
Nifty_50.xlsx
    
DataAgent.load_from_excel()
    
Technical Indicators (RSI, MACD, ATR, ADX, Returns)
    
XGBoost Model Training (85% train / 15% test split)
    
SHAP Explanations
    
Risk Calculations (Stop Loss & Targets)
    
PDF Report (Top 3 Picks)
```

## Minimal Experiment Plan (for the paper)
- **Task:** Next-day direction on NIFTY-50 universe.
- **Metrics:** Accuracy, F1(up), Precision@3, reliability (calibration) curve.
- **Ablations:** w/ and w/o Explainer Agent (or show reasons-only UX value),
  w/ and w/o calibration, and feature group ablations (TA vs returns only).
- **Leakage control:** Walk-forward time splits; indicators computed without look-ahead.
