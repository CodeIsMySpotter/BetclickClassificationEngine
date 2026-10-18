"""Sport-agnostic Domain Entities and Data Contracts.

Provides universal data representations for multi-sport betting intelligence:
- Football (Piłka nożna)
- Basketball (Koszykówka)
- Tennis (Tenis)
- Other future disciplines
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import datetime


class SportType(str, Enum):
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    TENNIS = "tennis"
    HOCKEY = "hockey"
    VOLLEYBALL = "volleyball"


@dataclass
class MatchEvent:
    """Universal sporting event / fixture representation."""
    event_id: str
    sport: SportType
    league_name: str
    home_competitor: str
    away_competitor: str
    event_date: datetime.date
    event_time: Optional[str] = None
    
    # Generic odds dictionary: e.g. {"1": 1.45, "X": 3.80, "2": 7.50, "1X": 1.15, "DNB": 1.18}
    odds: Dict[str, float] = field(default_factory=dict)
    
    # Optional sport-specific metadata (e.g. surface for tennis, rest days, weather)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return f"{self.home_competitor} vs {self.away_competitor}"


@dataclass
class BetOpportunity:
    """A quantified betting recommendation with probability and edge metrics."""
    event_id: str
    sport: SportType
    league_name: str
    match_display: str
    event_date: datetime.date
    market_name: str           # e.g. "1X (Win or Draw)", "DNB", "Czysta wygrana (1)", "Over 210.5"
    odds: float
    model_probability: float   # 0.0 - 1.0
    fair_odds: float
    expected_value: float      # e.g. +0.075 = +7.5%
    confidence_score: float    # Combined score balancing high probability and positive edge
    
    # Staking recommendations
    kelly_stake_pct: float = 0.0
    recommended_stake_pln: float = 0.0
    
    # Metadata for transparency (why model chose it)
    rationale: str = ""
    is_value_trap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sport": self.sport.value,
            "league_name": self.league_name,
            "match": self.match_display,
            "date": str(self.event_date),
            "market": self.market_name,
            "odds": round(self.odds, 2),
            "prob": round(self.model_probability, 4),
            "fair_odds": round(self.fair_odds, 2),
            "ev": round(self.expected_value, 4),
            "confidence_score": round(self.confidence_score, 2),
            "stake_pct": round(self.kelly_stake_pct, 2),
            "stake_pln": round(self.recommended_stake_pln, 2),
            "rationale": self.rationale
        }


@dataclass
class DailySlate:
    """The optimized daily portfolio of best bets for a specific calendar day."""
    target_date: datetime.date
    sport_filter: Optional[str]
    selection_mode: str         # "safest_pewniaki", "max_return", "balanced_sharpe"
    total_budget_pln: float
    
    # Selected bets
    selections: List[BetOpportunity] = field(default_factory=list)
    
    # Portfolio level aggregates
    total_stake_pln: float = 0.0
    avg_odds: float = 0.0
    avg_probability: float = 0.0
    avg_ev: float = 0.0
    expected_daily_profit_pln: float = 0.0
    expected_daily_roi_pct: float = 0.0
    
    # Stress-test scenarios for the day
    prob_all_win: float = 0.0
    prob_at_least_80pct_win: float = 0.0
    profit_if_all_win_pln: float = 0.0
    profit_if_one_loses_pln: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_date": str(self.target_date),
            "mode": self.selection_mode,
            "total_budget_pln": round(self.total_budget_pln, 2),
            "total_stake_pln": round(self.total_stake_pln, 2),
            "count_bets": len(self.selections),
            "avg_odds": round(self.avg_odds, 2),
            "avg_prob": round(self.avg_probability, 4),
            "avg_ev": round(self.avg_ev, 4),
            "expected_profit_pln": round(self.expected_daily_profit_pln, 2),
            "expected_roi_pct": round(self.expected_daily_roi_pct, 2),
            "prob_all_win": round(self.prob_all_win, 4),
            "prob_at_least_80pct_win": round(self.prob_at_least_80pct_win, 4),
            "profit_if_all_win_pln": round(self.profit_if_all_win_pln, 2),
            "profit_if_one_loses_pln": round(self.profit_if_one_loses_pln, 2),
            "selections": [s.to_dict() for s in self.selections]
        }
