from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

@dataclass
class ExplainerAgentConfig:
    topk: int = 4

class ExplainerAgent:
    def __init__(self, cfg: ExplainerAgentConfig):
        self.cfg = cfg
        self._explainer = None

    def fit(self, trained_estimator):
        # CalibratedClassifierCV has a base_estimator called "base_estimator_"
        base = getattr(trained_estimator, "base_estimator_", trained_estimator)
        # If it's a pipeline or similar, try to get underlying booster
        model = base if isinstance(base, XGBClassifier) else getattr(base, "best_estimator_", None) or base
        self._explainer = shap.TreeExplainer(model)

    def explain(self, X_row: pd.DataFrame) -> list[tuple[str, float]]:
        # Returns [(feature, shap_value), ...] topk by absolute contribution
        sv = self._explainer.shap_values(X_row)
        # For binary XGB, shap_values is (n_samples, n_features)
        vals = sv[0] if isinstance(sv, list) else sv
        vals = vals.flatten()
        feats = X_row.columns.tolist()
        pairs = list(zip(feats, vals))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return pairs[: self.cfg.topk]

    def to_reason_text(self, top_pairs: list[tuple[str,float]], X_row: pd.Series) -> str:
        templates = []
        for f, v in top_pairs:
            sign = "increasing" if v > 0 else "decreasing"
            if "rsi" in f:
                templates.append(f"RSI suggests momentum {sign}.")
            elif "macd" in f and "signal" not in f:
                templates.append(f"MACD {sign} influencing trend.")
            elif "macd_signal" in f:
                templates.append(f"MACD signal {sign}.")
            elif "atr" in f:
                templates.append(f"Volatility (ATR) {sign}; risk adjusted accordingly.")
            elif "adx" in f:
                templates.append(f"Trend strength (ADX) {sign}.")
            elif "ret_lag" in f:
                templates.append(f"Recent returns ({f}) affect outlook.")
            elif "vol_" in f:
                templates.append(f"Rolling volatility {sign}.")
            else:
                templates.append(f"{f} {sign}.")
        # Deduplicate phrases and join
        dedup = []
        for s in templates:
            if s not in dedup:
                dedup.append(s)
        return " ".join(dedup)
