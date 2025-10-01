import os, yaml, pandas as pd
from scrapers.odds_providers import build_provider
from model.io import ensure_dir, write_csv

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    odds_cfg = cfg.get("odds", {})
    provider_name = odds_cfg.get("provider", "theoddsapi")
    region = odds_cfg.get("region", "us")
    books = odds_cfg.get("books", ["fanduel","draftkings","betmgm"])
    market_map = odds_cfg.get("markets", {})
    alt_like = odds_cfg.get("alt_like_markets", [])
    throttle = int(odds_cfg.get("throttle_ms", 300))

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("[WARN] No ODDS_API_KEY set. Skipping odds fetch.")
        return

    provider = build_provider(provider_name, api_key=api_key, region=region, throttle_ms=throttle)
    res = provider.fetch(map_market=market_map, books=books, alt_like=alt_like)

    ensure_dir("odds")
    # Normalize columns to our schema, drop duplicates
    def norm(df, cols):
        if df is None or df.empty: return pd.DataFrame(columns=cols)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols].drop_duplicates()
        return df

    straight_cols = ["game_id","player","team","market","line","odds_over","odds_under","book"]
    alt_cols      = ["game_id","player","team","market","line","odds_over","odds_under","book"]
    td_cols       = ["game_id","player","team","market","odds_yes","book"]

    straight = norm(res.get("straight"), straight_cols)
    alt      = norm(res.get("alt"),      alt_cols)
    td       = norm(res.get("td"),       td_cols)

    # Minor cleanups
    for df in (straight, alt):
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
        df["odds_over"]  = pd.to_numeric(df["odds_over"],  errors="coerce").astype("Int64")
        df["odds_under"] = pd.to_numeric(df["odds_under"], errors="coerce").astype("Int64")

    td["odds_yes"] = pd.to_numeric(td["odds_yes"], errors="coerce").astype("Int64")

    write_csv(straight, "odds/straight.csv")
    write_csv(alt,      "odds/alt.csv")
    write_csv(td,       "odds/td.csv")
    print(f"[OK] Wrote odds: straight({len(straight)}), alt({len(alt)}), td({len(td)})")

if __name__ == "__main__":
    main()
