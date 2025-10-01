import numpy as np, pandas as pd
from .markets import weather_penalty, widen_for_pressure, widen_for_low_qb

def attach_context(df, priors, injuries, team_rates, schedule, weather):
    out = df.merge(priors, on=["player","market","team"], how="left", suffixes=("","_prior"))
    out = out.merge(injuries[["player","snap_adj","share_adj"]], on="player", how="left")
    out["snap_adj"] = out["snap_adj"].fillna(1.0)
    out["share_adj"] = out["share_adj"].fillna(1.0)
    out = out.merge(schedule[["game_id","home","away"]], on="game_id", how="left")
    out["opp"] = np.where(out["team"]==out["home"], out["away"], out["home"])
    out = out.merge(team_rates.rename(columns={"team":"team_ctx"}), left_on="team", right_on="team_ctx", how="left")
    out = out.merge(weather, on="game_id", how="left")
    return out

def adjusted_mu_sd(row, mods, cfg):
    mu, sd = row["mean"], row["sd"]
    if pd.isna(mu) or pd.isna(sd): return None, None

    # Team factors
    t = str(row["team"]).upper()
    team_factor = 1.0
    team_factor *= mods["pace"].get(t, 1.0)
    team_factor *= mods["proe"].get(t, 1.0)
    team_factor *= mods["off_epa"].get(t, 1.0)

    mu *= team_factor * row.get("snap_adj",1.0) * row.get("share_adj",1.0)

    # Weather widener
    wx_widen = weather_penalty(row.get("temp_f",55), row.get("wind_mph",0), cfg)
    sd *= wx_widen

    # Pressure / QB tier wideners
    opp = str(row.get("opp","")).upper()
    sd = widen_for_pressure(opp in mods.get("top10_pressure", set()), sd, cfg)
    # QB tier (only for passing yards)
    low_qb = (str(row["player"]).upper() in mods.get("low_tier_qb", set())) if "passing" in str(row["market"]).lower() else False
    sd = widen_for_low_qb(low_qb, sd, cfg)

    return mu, sd
