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
        sequence_length=7,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=64,
        epochs=1
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

    # Produce a small daily pick list using Gemini AI
    # Get predictions for test set
    proba_test_full = model_agent.predict_proba(X_test)
    
    print(f"Test set size: {len(X_test)}, Predictions size: {len(proba_test_full)}")
    
    # Align predictions with test data
    min_len = min(len(proba_test_full), len(X_test))
    
    if min_len == 0:
        print("Warning: No predictions generated. Cannot create picks.")
        return
    
    # Prepare data for Gemini - get last prediction per symbol
    symbol_predictions = []
    
    # Build a cleaner mapping using test data directly
    print("Building stock predictions data...")
    
    for i in range(min_len):
        try:
            # Get data from test set position
            test_row = X_test.iloc[i]
            test_idx = X_test.index[i]
            
            # Get corresponding row from df
            df_row = df.loc[test_idx]
            
            # Handle case where loc returns Series (duplicate index) or scalar
            if isinstance(df_row, pd.DataFrame):
                # Multiple rows with same index, take first
                symbol = str(df_row['symbol'].iloc[0])
                close_price = float(df_row['Close'].iloc[0])
                atr = float(df_row['atr_14'].iloc[0])
            else:
                # Single row
                symbol = str(df_row['symbol'])
                close_price = float(df_row['Close'])
                atr = float(df_row['atr_14'])
            
            p_up = float(proba_test_full[i])
            
            symbol_predictions.append({
                'symbol': symbol,
                'close': close_price,
                'atr': atr,
                'p_up': p_up,
                'index': i
            })
        except Exception as e:
            # Skip problematic rows
            continue
    
    print(f"Processed {len(symbol_predictions)} predictions")
    
    # Group by symbol and get last prediction for each
    pred_df = pd.DataFrame(symbol_predictions)
    
    if len(pred_df) == 0:
        print("Warning: No predictions to process. Cannot create picks.")
        return
    
    last_predictions = pred_df.groupby('symbol').tail(1)
    
    print(f"\nPreparing data for {len(last_predictions)} symbols to send to Gemini AI...")
    
    # Prepare summary for Gemini
    stock_data_summary = []
    risk_agent = RiskAgent(RiskAgentConfig())
    
    for _, row in last_predictions.iterrows():
        sym = row['symbol']
        p_up = row['p_up']
        close = row['close']
        atr = row['atr']
        
        risk = risk_agent.stops_targets(close, atr)
        conviction = risk_agent.conviction_bucket(p_up)
        
        stock_data_summary.append({
            'ticker': sym,
            'probability_up': f"{p_up:.2%}",
            'conviction': conviction,
            'last_close': round(float(close), 2),
            'stop_loss': risk['stop_loss'],
            'target': risk['target'],
            'atr': round(float(atr), 2)
        })
    
    # Sort by probability
    stock_data_summary = sorted(stock_data_summary, key=lambda x: float(x['probability_up'].strip('%'))/100, reverse=True)
    
    # Send to Gemini to generate report
    print("\n🤖 Sending data to Gemini AI for intelligent report generation...")
    
    import os
    from google import genai
    
    # Set API key as environment variable
    os.environ['GEMINI_API_KEY'] = "AIzaSyABLqP9xZVNJ7BgbQpFKlZlsh7CUn6Qq6g"
    client = genai.Client()
    
    # Create comprehensive prompt
    prompt = f"""You are an expert financial analyst. I have prediction data for {len(stock_data_summary)} Indian stocks (NSE).

Your task: Analyze this data and select the TOP 3 stock picks for tomorrow based on the probability of upward movement.

Here is ALL the stock data (sorted by probability):

{pd.DataFrame(stock_data_summary).to_string()}

Requirements for your report:
1. Select the TOP 3 stocks with highest probability of upward movement
2. For each stock, provide:
   - Ticker symbol
   - Conviction level (HIGH/MEDIUM/LOW based on probability: >60%=HIGH, 50-60%=MEDIUM, <50%=LOW)
   - Probability percentage
   - Current price (last_close)
   - Target price (from data)
   - Stop loss (from data)
   - A professional 2-3 sentence plain-English explanation of WHY this is a good pick, considering the probability, risk/reward ratio, and conviction level

Format your response EXACTLY as a JSON array like this:
[
  {{
    "ticker": "SYMBOL.NS",
    "conviction": "HIGH",
    "p_up": 0.65,
    "last_close": 1234.56,
    "target": 1300.00,
    "stop_loss": 1200.00,
    "reason_text": "Your professional explanation here..."
  }},
  ...3 stocks total...
]

Return ONLY the JSON array, no other text."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        response_text = response.text.strip()
        
        # Extract JSON from response
        import json
        import re
        
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            picks = json.loads(json_match.group())
            print(f"✅ Gemini generated {len(picks)} stock picks!")
        else:
            print("⚠ Could not parse Gemini response, using fallback...")
            # Fallback to top 3 from our data
            picks = []
            for stock in stock_data_summary[:3]:
                picks.append({
                    "ticker": stock['ticker'],
                    "conviction": stock['conviction'],
                    "p_up": float(stock['probability_up'].strip('%'))/100,
                    "last_close": stock['last_close'],
                    "target": stock['target'],
                    "stop_loss": stock['stop_loss'],
                    "reason_text": f"LSTM model predicts {stock['probability_up']} probability of upward movement with {stock['conviction']} conviction based on 7-day technical analysis."
                })
    
    except Exception as e:
        print(f"⚠ Gemini API error: {e}")
        print("Using fallback method...")
        # Fallback to top 3
        picks = []
        for stock in stock_data_summary[:3]:
            picks.append({
                "ticker": stock['ticker'],
                "conviction": stock['conviction'],
                "p_up": float(stock['probability_up'].strip('%'))/100,
                "last_close": stock['last_close'],
                "target": stock['target'],
                "stop_loss": stock['stop_loss'],
                "reason_text": f"LSTM model predicts {stock['probability_up']} probability of upward movement with {stock['conviction']} conviction based on 7-day technical analysis."
            })
    
    # Display picks
    print("\n" + "="*60)
    print("📊 TOP 3 STOCK PICKS")
    print("="*60)
    for i, pick in enumerate(picks, 1):
        print(f"\n#{i} {pick['ticker']}")
        print(f"   Conviction: {pick['conviction']} | Probability: {pick['p_up']:.1%}")
        print(f"   Price: ₹{pick['last_close']} | Target: ₹{pick['target']} | Stop: ₹{pick['stop_loss']}")
        print(f"   {pick['reason_text']}")
    print("\n" + "="*60)
    
    print(f"\nGenerated {len(picks)} picks for report")
    
    # Report
    REPORTS.mkdir(parents=True, exist_ok=True)
    rep = ReportAgent(ReportAgentConfig(
        outdir=REPORTS,
        gemini_api_key=None,  # Already used Gemini above
        enhance_with_llm=False  # Disable double-enhancement
    ))
    
    try:
        pdf_path = rep.save_simple_pdf(title="Stock Picks — Daily Report", cards=picks, filename="daily_report.pdf")
        print(f"\n✅ Report saved successfully: {pdf_path}")
        print(f"📄 Open the report to see detailed analysis of top 3 stocks!")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
