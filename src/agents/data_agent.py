from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yfinance as yf

@dataclass
class DataAgentConfig:
    period: str = "8y"
    interval: str = "1d"

class DataAgent:
    def __init__(self, cfg: DataAgentConfig):
        self.cfg = cfg

    def load_tickers(self, tickers_csv: str | Path) -> list[str]:
        df = pd.read_csv(tickers_csv)
        syms = [s.strip() for s in df["symbol"].dropna().unique().tolist()]
        return syms

    def download_ohlcv(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for sym in symbols:
            try:
                df = yf.download(sym, period=self.cfg.period, interval=self.cfg.interval, auto_adjust=False, progress=False)
                if not df.empty:
                    df = df.rename(columns={c: c.capitalize() for c in df.columns})
                    df.index.name = "Date"
                    out[sym] = df
            except Exception as e:
                print(f"[DataAgent] Failed {sym}: {e}")
        return out
