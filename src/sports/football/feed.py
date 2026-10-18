"""Football Odds Feed Adapter for Betclic PL.

Implements BaseOddsFeed interface, translating raw bookmaker data into universal MatchEvent domain objects.
"""

from pathlib import Path
from typing import List, Optional
import datetime
import pandas as pd

from src.core.domain import MatchEvent, SportType
from src.core.interfaces import BaseOddsFeed
from src.ingestion.betclic_feed import fetch_all_configured_leagues

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


class FootballBetclicFeed(BaseOddsFeed):
    """Betclic PL feed implementation for European football leagues."""

    def __init__(self, leagues_config: Optional[List[dict]] = None):
        self.leagues_config = leagues_config or []

    def get_sport_type(self) -> SportType:
        return SportType.FOOTBALL

    def fetch_live_events(self, force_refresh: bool = False) -> List[MatchEvent]:
        parquet_path = RAW_DATA_DIR / "betclic_live_odds.parquet"
        
        if force_refresh or not parquet_path.exists():
            if not self.leagues_config:
                # Load config if leagues_config was not explicitly provided
                import yaml
                cfg_path = BASE_DIR / "config" / "config.yaml"
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                self.leagues_config = cfg.get("leagues", [])
            df = fetch_all_configured_leagues(self.leagues_config)
        else:
            df = pd.read_parquet(parquet_path)

        events: List[MatchEvent] = []
        for _, row in df.iterrows():
            m_date = pd.to_datetime(row["match_date"]).date()
            odds_dict = {
                "1": float(row["odds_home"]),
                "X": float(row["odds_draw"]),
                "2": float(row["odds_away"]),
            }
            
            # Synthetic 1X and DNB odds
            oh, ox = odds_dict["1"], odds_dict["X"]
            if oh > 0 and ox > 0:
                odds_dict["1X"] = round(1.0 / ((1.0 / oh) + (1.0 / ox)), 2)
                odds_dict["DNB"] = round(oh * (1.0 - (1.0 / ox)), 2) if ox > 1.0 else oh

            event = MatchEvent(
                event_id=str(row["match_id"]),
                sport=SportType.FOOTBALL,
                league_name=str(row.get("league_name", "Football")),
                home_competitor=str(row["home_team"]),
                away_competitor=str(row["away_team"]),
                event_date=m_date,
                odds=odds_dict,
                metadata={
                    "competition_code": row.get("competition_code", ""),
                    "source": "betclic_pl"
                }
            )
            events.append(event)

        return events
