from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

@dataclass
class ModelAgentConfig:
    n_splits: int = 5
    random_state: int = 42

class ModelAgent:
    def __init__(self, cfg: ModelAgentConfig):
        self.cfg = cfg
        self.model = None

    def _build_base(self):
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.cfg.random_state,
            n_jobs=-1
        )

    def train_calibrated(self, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
        base = self._build_base()
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_splits)
        calib = CalibratedClassifierCV(base, method="sigmoid", cv=tscv)
        calib.fit(X, y)
        self.model = calib
        return calib

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:,1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1_up": float(f1_score(y, preds, zero_division=0))
        }
