from __future__ import annotations
import os, io, math, sys, json
from datetime import datetime
from typing import Optional

import yaml
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------- small odds helpers ----------
def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)): return None
    try: odds = float(odds)
    except: return None
    if odds == 0: return None
    if odds > 0:  # +120
        return 100.0 / (odds + 100.0)
    else:        # -130
        return (-1.0 * odds) / ((-1.0 * odds) + 100.0)

def prob_to_american(p: float) -> Optional[int]:
    if p is None or np.isnan(p) or p <= 0 or p >= 1: return None
    if p >= 0.5:
        dec = p / (1 - p)
        return int(round(-100 * dec))
    else:
        dec = (1 - p) / p
        return int(round(100 * dec))

def ev_from_prob(price_american: Optional[float], p: Optional[float]) -> Optional[float]:
    """
    Expected value per $1 stake at given American price and win prob p.
    Returns None if inputs missing.
    """
    if price_american is None or p is None: return None
    # decimal payout (incl. stake) from American
    if price_american > 0:
        dec = 1 + (price_american / 100.0)
    else:
        dec = 1 + (100.0 / abs(price_american))
    # Profit-only multiplier:
    prof = dec - 1.0
    return p * prof - (1 - p) * 1.0

def kelly_fraction(price_american: Optional[float], p: Optional[float]) -> Optional[float]:
    """
    Kelly f* with American price.
    """
    if price_american is None or p is None or p <= 0 or p >= 1:
        return None
    b = (price_american / 100.0) if price_american > 0 else (100.0 / abs(price_american))
    q = 1 - p
    f = (b * p - q) / b
    return max(0.0, f)

# ---------- resilient CSV fetcher (public feeds) ----------
def try_read_csv(url: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(url)
    except Exception:
        return None

# ---------- core adjustments ----------
def normalize_rank(series: pd.Series) -> pd.Series:
    """Return series scaled roughly to [-1, +1] by rank (robust to outliers)."""
    s = series.rank(pct=True).fillna(0.5)
    return (s - 0.5) * 2.0

def apply_external_adjustments(df: pd.DataFrame,
                               feeds: dict,
                               team_col_for_merge: str = "team") -> pd.DataFrame:
    """
    Merge simple team/QB context and adjust model_mean slightly.
    Expects df columns: team (or home/away team), player, market, model_mean, model_sd
    """
    adj = df.copy()

    # Team EPA
    team_epa = try_read_csv(feeds.get("team_epa", "")) if feeds else None
    if team_epa is not None and team_col_for_merge in adj.columns:
        # Expect columns with 'team' and 'off_epa' style; try to guess
        cand_team = [c for c in team_epa.columns if c.lower() in ("team","posteam","abbr","team_abbr")]
        cand_epa  = [c for c in team_epa.columns if "epa" in c.lower()]
        if cand_team and cand_epa:
            team_epa = team_epa.rename(columns={cand_team[0]: "team_feed",
                                                cand_epa[0]:  "team_epa_val"})
            team_epa = team_epa[["team_feed", "team_epa_val"]].dropna()
            team_epa["epa_norm"] = normalize_rank(team_epa["team_epa_val"])
            adj = adj.merge(team_epa, left_on=team_col_for_merge, right_on="team_feed", how="left")

    # Team pace (more plays → slightly bump yards/receptions)
    team_pace = try_read_csv(feeds.get("team_pace", "")) if feeds else None
    if team_pace is not None and team_col_for_merge in adj.columns:
        cand_team = [c for c in team_pace.columns if c.lower() in ("team","posteam","abbr","team_abbr")]
        cand_pace = [c for c in team_pace.columns if "pace" in c.lower() or "plays" in c.lower()]
        if cand_team and cand_pace:
            team_pace = team_pace.rename(columns={cand_team[0]: "team_feed2",
                                                  cand_pace[0]:  "team_pace_val"})
            team_pace = team_pace[["team_feed2","team_pace_val"]].dropna()
            team_pace["pace_norm"] = normalize_rank(team_pace["team_pace_val"])
            adj = adj.merge(team_pace, left_on=team_col_for_merge, right_on="team_feed2", how="left")

    # Simple mean adjustments: +3% per strong signal (bounded).
    bump = 1.0 \
           + 0.03 * adj.get("epa_norm", 0.0).fillna(0.0) \
           + 0.02 * adj.get("pace_norm", 0.0).fillna(0.0)
    bump = bump.clip(lower=0.90, upper=1.15)

    # Only apply to yardage-like markets (yards, receptions also ok)
    is_yards = adj["market"].str.contains("yd", case=False, na=False) | \
               adj["market"].str.contains("yard", case=False, na=False) | \
               adj["market"].str.contains("rec", case=False, na=False)
    adj.loc[is_yards, "model_mean"] = adj.loc[is_yards, "model_mean"] * bump.loc[is_yards]

    # Volatility widening: if qb pressure or low-tier flagged (we trigger via config in run())
    # Here we defer to run() for sd widening knobs using columns flags.
    return adj

# ---------- main public API ----------
def run(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Orchestrates:
      - load config
      - load priors.csv and odds.csv
      - merge, apply adjustments
      - compute prob/edge/kelly + outputs
    Returns the final dataframe (and also writes CSV/XLSX).
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_folder = cfg.get("outputs", {}).get("folder", "out")
    out_xlsx  = cfg.get("outputs", {}).get("excel", "Slate_Model_Output.xlsx")
    os.makedirs(out_folder, exist_ok=True)

    # 1) Load priors + odds
    # priors.csv: player, team, market, mean, sd (sd optional: fallback to cfg.defaults.sd.*)
    priors_path = os.path.join("data", "priors.csv")
    odds_path   = os.path.join("odds", "odds.csv")

    priors = pd.read_csv(priors_path) if os.path.exists(priors_path) else pd.DataFrame()
    odds   = pd.read_csv(odds_path)   if os.path.exists(odds_path)   else pd.DataFrame()

    if priors.empty or odds.empty:
        # make it loud but still complete the run
        warn = "WARNING: priors.csv or odds.csv missing/empty. Creating empty output."
        print(warn)
        empty = pd.DataFrame(columns=[
            "game","team","player","market","line","over_odds","under_odds",
            "model_mean","model_sd",
            "p_over","p_under","fair_odds_over","fair_odds_under",
            "edge_over","edge_under","kelly_over","kelly_under","bet_reco","notes"
        ])
        empty.to_csv(os.path.join(out_folder, "model_output.csv"), index=False)
        try:
            empty.to_excel(os.path.join(out_folder, out_xlsx), index=False)
        except Exception:
            pass
        return empty

    # Canonicalize columns
    for df in (priors, odds):
        for c in df.columns:
            df.rename(columns={c: c.strip().lower()}, inplace=True)

    # Default SDs
    defaults_sd = (cfg.get("defaults", {}) or {}).get("sd", {}) or {}
    sd_map = {
        "receiving_yards": defaults_sd.get("receiving_yards", 28.0),
        "receptions":      defaults_sd.get("receptions", 1.6),
        "rushing_yards":   defaults_sd.get("rushing_yards", 22.0),
        "rush_attempts":   defaults_sd.get("rush_attempts", 4.0),
        "passing_yards":   defaults_sd.get("passing_yards", 45.0),
    }

    # Normalize market naming a bit
    def normalize_market(m: str) -> str:
        m = (m or "").strip().lower()
        m = m.replace(" ", "_")
        m = m.replace("rec_", "receptions_") if m.startswith("rec_") else m
        return m

    priors["market"] = priors["market"].map(normalize_market)
    odds["market"]   = odds["market"].map(normalize_market)

    # Merge priors → odds
    # We merge on (player, team, market). If team missing in odds, merge on (player, market).
    keys = ["player","team","market"]
    have_team_in_odds = odds.columns.str.contains("^team$").any()
    left_keys = ["player","market"] + (["team"] if have_team_in_odds else [])

    merged = odds.merge(priors.rename(columns={"mean":"prior_mean","sd":"prior_sd"}),
                        on=["player","market"] + (["team"] if have_team_in_odds else []),
                        how="left")

    # Fill SD by defaults if missing from priors
    def default_sd_for_market(m: str) -> float:
        if "recep" in m: return sd_map["receptions"]
        if "rush" in m and "att" in m: return sd_map["rush_attempts"]
        if "rush" in m: return sd_map["rushing_yards"]
        if "pass" in m: return sd_map["passing_yards"]
        # yards generic
        if "yd" in m or "yard" in m: return sd_map["receiving_yards"]
        return sd_map["receiving_yards"]

    merged["model_mean"] = merged["prior_mean"]
    merged["model_sd"] = merged.apply(
        lambda r: r["prior_sd"] if pd.notna(r.get("prior_sd", np.nan)) else default_sd_for_market(r["market"]),
        axis=1
    )

    # 2) External feeds adjustments (EPA, pace, etc.)
    merged.rename(columns={"team":"team"}, inplace=True)  # ensure unified name
    merged = apply_external_adjustments(merged, cfg.get("feeds", {}), team_col_for_merge="team")

    # 3) Volatility widening (config knobs)
    vol = cfg.get("volatility", {}) or {}
    # Flags could be precomputed columns (e.g., qb_lowtier_flag, pressure_top10_flag)
    # If not present, nothing happens.
    if "widen_sd_if_pressure_top10" in vol and "pressure_top10_flag" in merged.columns:
        merged.loc[merged["pressure_top10_flag"] == 1, "model_sd"] *= (1.0 + float(vol["widen_sd_if_pressure_top10"]))
    if "widen_sd_if_qb_lowtier" in vol and "qb_lowtier_flag" in merged.columns:
        merged.loc[merged["qb_lowtier_flag"] == 1, "model_sd"] *= (1.0 + float(vol["widen_sd_if_qb_lowtier"]))

    # 4) Probabilities from Normal(mean, sd)
    # Odds columns expected: over_odds / under_odds (American). If not present, compute fair odds only.
    for c in ("over_odds","under_odds"):
        if c not in merged.columns:
            merged[c] = np.nan

    # Some sites store ints as strings like "+120"
    for c in ("over_odds","under_odds"):
        merged[c] = merged[c].apply(lambda x: float(str(x).replace("+","")) if pd.notna(x) and str(x).strip()!='' else np.nan)

    # guard sd lower bound
    merged["model_sd"] = merged["model_sd"].clip(lower=1e-6)

    # p_over = 1 - CDF(line)
    merged["p_over"]  = 1.0 - norm.cdf(merged["line"], loc=merged["model_mean"], scale=merged["model_sd"])
    merged["p_under"] = 1.0 - merged["p_over"]

    # 5) Fair odds (American)
    merged["fair_odds_over"]  = merged["p_over"].apply(prob_to_american)
    merged["fair_odds_under"] = merged["p_under"].apply(prob_to_american)

    # 6) Edge (expected value per $1 stake)
    merged["edge_over"]  = merged.apply(lambda r: ev_from_prob(r["over_odds"],  r["p_over"]),  axis=1)
    merged["edge_under"] = merged.apply(lambda r: ev_from_prob(r["under_odds"], r["p_under"]), axis=1)

    # 7) Kelly (capped)
    risk = cfg.get("risk", {}) or {}
    cap_straight = float(risk.get("kelly_cap_straight", 0.05))
    merged["kelly_over"]  = merged.apply(lambda r: min(cap_straight, kelly_fraction(r["over_odds"],  r["p_over"])  or 0.0), axis=1)
    merged["kelly_under"] = merged.apply(lambda r: min(cap_straight, kelly_fraction(r["under_odds"], r["p_under"]) or 0.0), axis=1)

    # 8) Recommendation + color tags
    edge_green = float(risk.get("edge_green", 0.04))
    edge_amber = float(risk.get("edge_amber", 0.01))

    def reco_row(r):
        eo, eu = r["edge_over"], r["edge_under"]
        if eo is None and eu is None: return "Pass"
        if eo is None: eo = -9
        if eu is None: eu = -9
        if eo > eu:
            bet = f"Over {r['line']}"
            edge = eo
        else:
            bet = f"Under {r['line']}"
            edge = eu
        tag = "GREEN" if edge >= edge_green else ("AMBER" if edge >= edge_amber else "RED")
        return f"{bet} ({tag})"

    merged["bet_reco"] = merged.apply(reco_row, axis=1)

    # Notes
    def mk_notes(r):
        nts = []
        if pd.isna(r.get("over_odds")) or pd.isna(r.get("under_odds")):
            nts.append("no_book_price")
        if "epa_norm" in r and pd.notna(r["epa_norm"]) and r["epa_norm"] > 0.5:
            nts.append("strong_team_epa")
        if "pace_norm" in r and pd.notna(r["pace_norm"]) and r["pace_norm"] > 0.5:
            nts.append("fast_pace")
        return ",".join(nts)

    merged["notes"] = merged.apply(mk_notes, axis=1)

    # Tidy columns for output
    out_cols = [
        "game","team","player","market","line","over_odds","under_odds",
        "model_mean","model_sd",
        "p_over","p_under","fair_odds_over","fair_odds_under",
        "edge_over","edge_under","kelly_over","kelly_under","bet_reco","notes"
    ]
    for c in out_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    final = merged[out_cols].copy()

    # Save outputs
    csv_path  = os.path.join(out_folder, "model_output.csv")
    xlsx_path = os.path.join(out_folder, out_xlsx)
    final.to_csv(csv_path, index=False)
    try:
        # requires openpyxl or xlsxwriter (already in requirements.txt)
        final.to_excel(xlsx_path, index=False)
    except Exception as e:
        print("Excel write failed (continuing):", e)

    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote: {xlsx_path}")
    return final
