from __future__ import annotations
from pathlib import Path
import joblib, json

def save_joblib(obj, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: str | Path):
    return joblib.load(path)

def save_json(obj, path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
