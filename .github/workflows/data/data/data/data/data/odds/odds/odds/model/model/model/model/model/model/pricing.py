import pandas as pd
from .io import american_to_implied, prob_to_american

def price_row(side, p_over, p_under, odds_over, odds_under):
    p = p_over if side=="Over" else p_under
    fair = prob_to_american(p) if pd.notna(p) else None
    imp = american_to_implied(odds_over if side=="Over" else odds_under)
    edge = (p - imp) if pd.notna(p) else None
    return p, fair, imp, edge

def label(edge_pct, cfg):
    eg = cfg["risk"]["edge_green"]; ea = cfg["risk"]["edge_amber"]
    if edge_pct is None: return "Pass"
    if edge_pct >= eg: return "Bet (Green)"
    if edge_pct >= ea: return "Lean (Amber)"
    return "Pass"
