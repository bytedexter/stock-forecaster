from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskAgentConfig:
    atr_mult_stop: float = 1.5
    atr_mult_target: float = 2.5

class RiskAgent:
    def __init__(self, cfg: RiskAgentConfig):
        self.cfg = cfg

    def conviction_bucket(self, p_up: float) -> str:
        if p_up >= 0.7: return "high"
        if p_up >= 0.55: return "medium"
        return "low"

    def stops_targets(self, last_close: float, atr: float) -> dict:
        stop = max(0.0, last_close - self.cfg.atr_mult_stop * atr)
        target = last_close + self.cfg.atr_mult_target * atr
        return {"stop_loss": round(stop, 2), "target": round(target, 2)}
