from __future__ import annotations
import pandas as pd
import numpy as np
import ta

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic indicators
    df["ret_1d"] = df["Close"].pct_change()
    df["vol_10"] = df["ret_1d"].rolling(10).std()
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["adx_14"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14)
    df["atr_14"] = atr.average_true_range()
    # Lagged features
    for k in [1,2,3,5]:
        df[f"ret_lag_{k}"] = df["ret_1d"].shift(k)
    # Targets: next-day direction
    df["target_up"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df
