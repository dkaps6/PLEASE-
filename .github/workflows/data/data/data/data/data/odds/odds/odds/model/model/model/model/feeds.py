import os, time, warnings, pandas as pd, requests

def _get_csv(url: str) -> pd.DataFrame | None:
    if not url: return None
    for i in range(3):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            from io import StringIO
            return pd.read_csv(StringIO(r.text))
        except Exception as e:
            warnings.warn(f"Feed failed: {url} ({e}), retry {i+1}/3")
            time.sleep(1+i)
    return None

def load_feed_mods(cfg):
    urls = cfg.get("feeds", {})
    team_epa  = _get_csv(urls.get("team_epa"))
    team_pace = _get_csv(urls.get("team_pace"))
    team_proe = _get_csv(urls.get("team_proe"))
    rbsdm_qb  = _get_csv(urls.get("rbsdm_qb"))

    mods = {"pace":{}, "proe":{}, "off_epa":{}, "top10_pressure": set(), "low_tier_qb": set()}

    # Map team factors (normalize to ~1.0 mean)
    def _norm_col(df, team_col, val_col, scale=0.05):
        if df is None or team_col not in df or val_col not in df: return {}
        m, s = df[val_col].mean(), max(df[val_col].std(ddof=0), 1e-6)
        out = {}
        for _, r in df.iterrows():
            t = str(r[team_col]).upper()
            out[t] = 1.0 + (float(r[val_col]) - m)/s * scale
        return out

    # Heuristic column names; adjust if your mirrors differ
    if team_pace is not None:
        # expect columns like team, pace
        tc = [c for c in team_pace.columns if c.lower() in ("team","posteam","abbr")]
        vc = [c for c in team_pace.columns if "pace" in c.lower()]
        if tc and vc:
            mods["pace"] = _norm_col(team_pace, tc[0], vc[0])

    if team_proe is not None:
        tc = [c for c in team_proe.columns if c.lower() in ("team","posteam","abbr")]
        vc = [c for c in team_proe.columns if "proe" in c.lower()]
        if tc and vc:
            mods["proe"] = _norm_col(team_proe, tc[0], vc[0])

    if team_epa is not None:
        tc = [c for c in team_epa.columns if c.lower() in ("team","posteam","abbr")]
        vc = [c for c in team_epa.columns if "off" in c.lower() and "epa" in c.lower()]
        if tc and vc:
            mods["off_epa"] = _norm_col(team_epa, tc[0], vc[0])

    # QB tiers via RBSDM EPA (very rough)
    if rbsdm_qb is not None:
        namec = [c for c in rbsdm_qb.columns if "player" in c.lower() or "name" in c.lower()]
        epac  = [c for c in rbsdm_qb.columns if "epa" in c.lower()]
        if namec and epac:
            m, s = rbsdm_qb[epac[0]].mean(), max(rbsdm_qb[epac[0]].std(ddof=0), 1e-6)
            low_cut = m - 0.5*s
            low = rbsdm_qb[rbsdm_qb[epac[0]] <= low_cut][namec[0]].dropna().astype(str).str.upper().tolist()
            mods["low_tier_qb"] = set(low)

    # Pressure top10 proxy — if you have a pressure feed, mark teams here.
    # For now, leave empty set; you can wire a pressure feed later.
    mods["top10_pressure"] = set()

    return mods
