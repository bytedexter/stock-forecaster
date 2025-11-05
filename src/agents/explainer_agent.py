from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ExplainerAgentConfig:
    topk: int = 4

class ExplainerAgent:
    def __init__(self, cfg: ExplainerAgentConfig):
        self.cfg = cfg
        self.model_agent = None

    def fit(self, trained_model_agent):
        """Store reference to the trained model agent (LSTM with attention)"""
        self.model_agent = trained_model_agent

    def explain(self, X_row: pd.DataFrame) -> list[tuple[str, float]]:
        """
        Use attention weights to explain predictions.
        Returns [(feature, importance), ...] topk by absolute contribution
        """
        if self.model_agent is None:
            raise ValueError("ExplainerAgent not fitted. Call fit() first.")
        
        # Get attention weights for the last sample
        attention_weights = self.model_agent.get_attention_weights(X_row, index=-1)
        
        # Average attention weights across time steps to get feature importance
        # Shape: (sequence_length,) -> we'll use the most recent timesteps
        recent_attention = attention_weights[-10:]  # Last 10 timesteps
        avg_attention = np.mean(recent_attention)
        
        # Get feature values from the last row
        feature_names = self.model_agent.feature_names
        feature_values = X_row[feature_names].iloc[-1].values
        
        # Calculate feature importance based on:
        # 1. Attention weights (temporal importance)
        # 2. Feature magnitude (feature contribution)
        feature_importance = []
        for i, feat_name in enumerate(feature_names):
            # Use attention weight and feature value magnitude
            importance = avg_attention * abs(feature_values[i])
            # Keep sign information
            signed_importance = importance * np.sign(feature_values[i])
            feature_importance.append((feat_name, float(signed_importance)))
        
        # Sort by absolute importance
        feature_importance.sort(key=lambda t: abs(t[1]), reverse=True)
        
        return feature_importance[: self.cfg.topk]

    def to_reason_text(self, top_pairs: list[tuple[str,float]], X_row: pd.Series) -> str:
        templates = []
        for f, v in top_pairs:
            sign = "increasing" if v > 0 else "decreasing"
            if "rsi" in f.lower():
                templates.append(f"RSI suggests momentum {sign}.")
            elif "macd" in f.lower() and "signal" not in f.lower():
                templates.append(f"MACD {sign} influencing trend.")
            elif "macd_signal" in f.lower():
                templates.append(f"MACD signal {sign}.")
            elif "atr" in f.lower():
                templates.append(f"Volatility (ATR) {sign}; risk adjusted accordingly.")
            elif "adx" in f.lower():
                templates.append(f"Trend strength (ADX) {sign}.")
            elif "ret_lag" in f.lower():
                templates.append(f"Recent returns ({f}) affect outlook.")
            elif "vol_" in f.lower():
                templates.append(f"Rolling volatility {sign}.")
            else:
                templates.append(f"{f} {sign}.")
        # Deduplicate phrases and join
        dedup = []
        for s in templates:
            if s not in dedup:
                dedup.append(s)
        return " ".join(dedup)
