import os, pandas as pd, numpy as np, yaml
from .io import read_csv, write_csv, ensure_dir, american_to_implied, prob_to_american, american_to_decimal
from .feeds import load_feed_mods
from .features import attach_context, adjusted_mu_sd
from .markets import yardage_probs
from .pricing import price_row, label
from .kelly import kelly_fraction
from .parlays import build_parlays

def market_key(m):
    m = str(m).lower()
    if "receiving" in m and "yard" in m: return "receiving_yards"
    if "receptions" in m: return "receptions"
    if "rushing" in m and "yard" in m: return "rushing_yards"
    if "rush" in m and "att" in m: return "rush_attempts"
    if "passing" in m and "yard" in m: return "passing_yards"
    if "anytime" in m and "td" in m: return "anytime_td"
    return "other"

def run(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    outdir = cfg["outputs"]["folder"]; ensure_dir(outdir)

    # Load inputs
    priors     = read_csv("data/priors.csv")
    injuries   = read_csv("data/injuries.csv")
    team_rates = read_csv("data/team_rates.csv")
    weather    = read_csv("data/weather.csv")
    schedule   = read_csv("data/schedule.csv")

    straight   = read_csv("odds/straight.csv")
    alt        = read_csv("odds/alt.csv")
    td         = read_csv("odds/td.csv")

    # External feed modifiers (NFLverse/RBSDM; robust to outages)
    mods = load_feed_mods(cfg)

    # ----- Yardage markets (straight + alt) -----
    def eval_yardage(df):
        if df.empty: return df
        df = attach_context(df, priors, injuries, team_rates, schedule, weather)
        rows = []
        for _, r in df.iterrows():
            mk = market_key(r["market"])
            if mk not in ("receiving_yards","rushing_yards","passing_yards","receptions","rush_attempts"): 
                continue
            mu, sd = adjusted_mu_sd(r, mods, cfg)
            if mu is None or sd is None: continue
            p_over, p_under = yardage_probs(mu, sd, float(r["line"]))
            # pick side
            for side in ("Over","Under"):
                p, fair, imp, edge = price_row(side, p_over, p_under, float(r["odds_over"]), float(r["odds_under"]))
                edge_pct = edge*100 if edge is not None else None
                kcap = cfg["risk"]["kelly_cap_alt"] if "alt" in str(df.columns) else cfg["risk"]["kelly_cap_straight"]
                kelly = kelly_fraction(p, float(r["odds_over"] if side=="Over" else r["odds_under"]))
                kelly = min(kelly, kcap)
                rows.append({
                    "game_id": r["game_id"], "team": r["team"], "player": r["player"], "market": r["market"],
                    "line": r["line"], "book_odds_american": int(r["odds_over"] if side=="Over" else r["odds_under"]),
                    "side": side,
                    "model_mean": round(mu,2), "model_sd": round(sd,2),
                    "over_pct": round(p_over,4), "under_pct": round(p_under,4),
                    "fair_odds_over": fair if side=="Over" else None,
                    "fair_odds_under": fair if side=="Under" else None,
                    "edge_pct": round(edge_pct,2) if edge_pct is not None else None,
                    "kelly_fraction": round(kelly,4),
                    "side_prob": p,
                    "decimal_odds": american_to_decimal(float(r["odds_over"] if side=="Over" else r["odds_under"])),
                    "recommendation": label(edge_pct/100 if edge_pct is not None else None, cfg).replace("Bet","Bet "+side),
                    "notes": ""
                })
        return pd.DataFrame(rows)

    straight_eval = eval_yardage(straight)
    alt_eval      = eval_yardage(alt)

    # ----- TD markets (Poisson thinning w/ simple shares) -----
    td_eval = pd.DataFrame()
    if not td.empty:
        # Minimal TD model: use team context; you can wire red-zone shares later
        td_calc = []
        for _, r in td.iterrows():
            # crude prior: team ~2.3 TDs, player share ~0.18 (can be refined)
            lam_team = 2.3
            share = 0.18
            p_yes = 1 - np.exp(-lam_team*share)
            fair = prob_to_american(p_yes)
            imp = american_to_implied(float(r["odds_yes"]))
            edge_pct = (p_yes - imp)/imp*100
            kelly = kelly_fraction(p_yes, float(r["odds_yes"]))
            td_calc.append({
                "game_id": r["game_id"], "team": r["team"], "player": r["player"], "market": r["market"],
                "odds_yes": int(r["odds_yes"]), "p_yes": round(p_yes,4),
                "fair_odds_yes": fair, "edge_yes_pct": round(edge_pct,2), "kelly_yes": round(kelly,4),
                "recommendation": "Yes" if edge_pct>=1.0 else "Pass"
            })
        td_eval = pd.DataFrame(td_calc)

    # ----- Outputs -----
    if not straight_eval.empty: write_csv(straight_eval, f"{outdir}/straight_eval.csv")
    if not alt_eval.empty:      write_csv(alt_eval,      f"{outdir}/alt_eval.csv")
    if not td_eval.empty:       write_csv(td_eval,       f"{outdir}/td_eval.csv")

    # Top edges (combine)
    tops = []
    if not straight_eval.empty: tops.append(straight_eval.assign(market_type="straight"))
    if not alt_eval.empty:      tops.append(alt_eval.assign(market_type="alt"))
    if not td_eval.empty:       tops.append(td_eval.assign(market_type="td"))
    if tops:
        top_df = pd.concat(tops, ignore_index=True)
        if "edge_pct" in top_df.columns:
            top_df["best_edge_pct"] = top_df["edge_pct"]
        elif "edge_yes_pct" in top_df.columns:
            top_df["best_edge_pct"] = top_df["edge_yes_pct"]
        top_df = top_df.sort_values("best_edge_pct", ascending=False)
        write_csv(top_df, f"{outdir}/top_edges.csv")

    # Excel summary
    try:
        import openpyxl
        from openpyxl import Workbook
        wb = Workbook()
        def _dump(name, df):
            ws = wb.create_sheet(name)
            ws.append(list(df.columns))
            for row in df.itertuples(index=False): ws.append(list(row))
        wb.remove(wb.active)
        if not straight_eval.empty: _dump("Straight", straight_eval)
        if not alt_eval.empty:      _dump("Alt", alt_eval)
        if not td_eval.empty:       _dump("TD", td_eval)
        if tops:                    _dump("Top_Edges", top_df.head(100))
        wb.save(os.path.join(outdir, cfg["outputs"]["excel"]))
    except Exception as e:
        print(f"[Excel skipped] {e}")
