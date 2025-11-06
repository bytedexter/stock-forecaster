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

    def load_from_excel(self, excel_path: str | Path) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data from an Excel file with multiple sheets (one per ticker).
        Each sheet should have columns: Date, Open, High, Low, Close, Volume
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dictionary mapping ticker symbols to their OHLCV DataFrames
        """
        out = {}
        excel_path = Path(excel_path)
        
        if not excel_path.exists():
            print(f"[DataAgent] Excel file not found: {excel_path}")
            return out
        
        try:
            # Read all sheet names
            excel_file = pd.ExcelFile(excel_path)
            sheet_names = excel_file.sheet_names
            
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)
                    
                    # Ensure Date column exists and set as index
                    # Handle different possible date column names
                    date_col = None
                    for col in df.columns:
                        if 'date' in col.lower():
                            date_col = col
                            break
                    
                    if date_col:
                        # Convert to datetime, handling timezone strings properly
                        # First remove timezone info from the string, then convert
                        df['Date'] = pd.to_datetime(df[date_col].astype(str).str.replace(r'\s+[A-Z]{2,4}$', '', regex=True), errors='coerce')
                        df.set_index('Date', inplace=True)
                        # Drop the original date column if it's different
                        if date_col != 'Date' and date_col in df.columns:
                            df.drop(columns=[date_col], inplace=True)
                    else:
                        print(f"[DataAgent] Warning: No date column found in sheet '{sheet_name}'")
                        continue
                    
                    # Standardize column names (capitalize first letter)
                    df = df.rename(columns={c: c.capitalize() for c in df.columns})
                    
                    # Verify required columns exist
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_cols):
                        out[sheet_name] = df
                        print(f"[DataAgent] Loaded {len(df)} rows for {sheet_name}")
                    else:
                        print(f"[DataAgent] Missing required columns in sheet '{sheet_name}'")
                        
                except Exception as e:
                    print(f"[DataAgent] Failed to load sheet '{sheet_name}': {e}")
                    
        except Exception as e:
            print(f"[DataAgent] Failed to read Excel file: {e}")
            
        return out
