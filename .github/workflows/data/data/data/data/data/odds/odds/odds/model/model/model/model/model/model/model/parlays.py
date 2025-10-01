import numpy as np, pandas as pd
from .io import american_to_decimal

def build_parlays(pred_df, cfg):
    pen = float(cfg["parlays"]["correlation_penalty"])
    outs = []
    for b in cfg["parlays"]["buckets"]:
        name = b["name"]; min_edge = 100*b["min_edge_pct"] if b["min_edge_pct"]<1 else b["min_edge_pct"]; max_legs = b["max_legs"]
        pool = pred_df[pred_df["edge_pct"]>=min_edge].sort_values("edge_pct", ascending=False).head(40)
        if pool.empty: 
            outs.append({"bucket": name, "legs":"", "parlay_prob":0.0, "decimal_odds":1.0, "ev_per_unit":-1.0})
            continue
        legs = pool.head(max_legs)
        probs = legs["side_prob"].clip(0.001,0.999).values * pen
        parlay_prob = float(np.prod(probs))
        dec_odds = float(np.prod(legs["decimal_odds"].values))
        ev = parlay_prob*dec_odds - 1.0
        outs.append({
            "bucket": name,
            "legs": " | ".join(f'{r.player} {r.market} {r.side} {r.line} ({r.book_odds_american})' for _,r in legs.iterrows()),
            "parlay_prob": parlay_prob,
            "decimal_odds": dec_odds,
            "ev_per_unit": ev
        })
    return pd.DataFrame(outs)
