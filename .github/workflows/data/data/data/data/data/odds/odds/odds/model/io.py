import pandas as pd
from pathlib import Path

def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def write_csv(df: pd.DataFrame, path: str):
    ensure_dir(str(Path(path).parent))
    df.to_csv(path, index=False)

def american_to_decimal(odds: float) -> float:
    return 1 + (100/abs(odds)) if odds < 0 else 1 + (odds/100)

def american_to_implied(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)

def prob_to_american(p: float) -> int:
    p = min(max(p, 1e-6), 1-1e-6)
    return int(round(-100*(p/(1-p)))) if p>=0.5 else int(round(100*((1-p)/p)))

def devig_two_way(p_over: float, p_under: float):
    s = p_over + p_under
    if s <= 0: return None, None
    return p_over/s, p_under/s
