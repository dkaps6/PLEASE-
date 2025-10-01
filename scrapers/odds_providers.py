import os, time, requests, pandas as pd
from typing import Dict, List, Tuple

class OddsProviderBase:
    def fetch(self, **kwargs) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

# ---------------- The Odds API ----------------
class TheOddsAPI(OddsProviderBase):
    BASE = "https://api.the-odds-api.com/v4/sports"

    def __init__(self, api_key: str, region: str = "us", throttle_ms: int = 300):
        self.api_key = api_key
        self.region = region
        self.sleep = throttle_ms / 1000.0

    def _get(self, path: str, params: dict) -> list:
        url = f"{self.BASE}/{path}"
        params = {**params, "apiKey": self.api_key, "regions": self.region, "oddsFormat": "american"}
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        time.sleep(self.sleep)
        return r.json()

    @staticmethod
    def _extract_game_id(event: dict) -> str:
        # Normalize to "AWAY@HOME"
        home = event.get("home_team","").upper()
        sites = event.get("bookmakers",[])
        # Odds API gives "commence_time" and "home_team"; compete_teams in "teams"
        teams = event.get("teams",[])
        away = [t for t in teams if t.upper()!=home.upper()]
        away = away[0].upper() if away else ""
        return f"{away}@{home}"

    def fetch_markets(self, market_keys: List[str], books: List[str]) -> Dict[str, list]:
        # One call per market. NFL sport key is "americanfootball_nfl".
        out = {}
        for m in market_keys:
            data = self._get("americanfootball_nfl/odds", {"markets": m, "bookmakers": ",".join(books)})
            out[m] = data
        return out

    def _rows_from_market(self, market_key: str, events: list, map_market: Dict[str,str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (straight_df, alt_df)
        straight_df columns: game_id,player,team,market,line,odds_over,odds_under,book
        alt_df      columns: same
        TD df handled separately in fetch()
        """
        rows_main, rows_alt = [], []
        model_market = map_market.get(market_key, None)
        if not model_market:
            return pd.DataFrame(), pd.DataFrame()

        for ev in events:
            game_id = self._extract_game_id(ev)
            for bm in ev.get("bookmakers", []):
                book = bm.get("key","")
                for mk in bm.get("markets", []):
                    if mk.get("key") != market_key: 
                        continue
                    outcomes = mk.get("outcomes", [])
                    # The Odds API returns player prop outcomes with 'name', 'description'/ 'player', and 'price', 'point'
                    # "Alt" determination: multiple distinct 'point' values for same player within the same market.
                    # We'll group by (player, point) and treat each as a line; if a player has >1 unique points, those are ALTs.
                    by_player = {}
                    for o in outcomes:
                        player = (o.get("description") or o.get("player") or o.get("name") or "").strip()
                        side = (o.get("name") or "").strip().lower()    # "Over" / "Under" often in name
                        price = int(o.get("price")) if o.get("price") not in (None,"") else None
                        point = o.get("point")
                        if not player or price is None or point is None:
                            continue
                        key = (player, float(point))
                        by_player.setdefault(key, {"over": None, "under": None})
                        if "over" in side.lower():
                            by_player[key]["over"] = price
                        elif "under" in side.lower():
                            by_player[key]["under"] = price

                    # now build rows
                    # Count unique points per player to detect ALTs
                    counts = {}
                    for (player, point) in by_player.keys():
                        counts[player] = counts.get(player, 0) + 1

                    for (player, point), prices in by_player.items():
                        row = {
                            "game_id": game_id,
                            "player": player,
                            "team": "",  # optional: fill if you map players->teams
                            "market": model_market,
                            "line": point,
                            "odds_over": prices["over"],
                            "odds_under": prices["under"],
                            "book": book
                        }
                        if counts.get(player, 1) > 1:
                            rows_alt.append(row)
                        else:
                            rows_main.append(row)

        return pd.DataFrame(rows_main), pd.DataFrame(rows_alt)

    def _td_rows(self, events: list, map_market: Dict[str,str]) -> pd.DataFrame:
        rows = []
        model_market = map_market.get("player_anytime_td", "anytime_td")
        for ev in events:
            game_id = self._extract_game_id(ev)
            for bm in ev.get("bookmakers", []):
                book = bm.get("key","")
                for mk in bm.get("markets", []):
                    if mk.get("key") != "player_anytime_td":
                        continue
                    for o in mk.get("outcomes", []):
                        player = (o.get("description") or o.get("player") or o.get("name") or "").strip()
                        price = o.get("price")
                        if player and price not in (None,""):
                            rows.append({
                                "game_id": game_id,
                                "player": player,
                                "team": "",
                                "market": model_market,
                                "odds_yes": int(price),
                                "book": book
                            })
        return pd.DataFrame(rows)

    def fetch(self, map_market: Dict[str,str], books: List[str], alt_like: List[str]) -> Dict[str, pd.DataFrame]:
        mkts = list(map_market.keys())
        data_by_market = self.fetch_markets(mkts, books)

        straight_df_list, alt_df_list = [], []
        td_df = pd.DataFrame()

        for mkt, events in data_by_market.items():
            if mkt == "player_anytime_td":
                td_df = pd.concat([td_df, self._td_rows(events, map_market)], ignore_index=True)
            else:
                straight_df, alt_df = self._rows_from_market(mkt, events, map_market)
                # Some markets may not expose multiple points; treat all as straight in that case.
                if not alt_df.empty:
                    alt_df_list.append(alt_df)
                if not straight_df.empty:
                    straight_df_list.append(straight_df)

        return {
            "straight": pd.concat(straight_df_list, ignore_index=True) if straight_df_list else pd.DataFrame(),
            "alt":      pd.concat(alt_df_list, ignore_index=True)      if alt_df_list      else pd.DataFrame(),
            "td":       td_df
        }

def build_provider(name: str, **kwargs) -> OddsProviderBase:
    if name.lower() == "theoddsapi":
        return TheOddsAPI(api_key=kwargs["api_key"], region=kwargs.get("region","us"), throttle_ms=kwargs.get("throttle_ms",300))
    raise ValueError(f"Unknown odds provider: {name}")
