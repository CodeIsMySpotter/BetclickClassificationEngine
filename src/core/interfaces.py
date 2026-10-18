"""Abstract Interfaces for Ingestion Feeds, Sport Models, and Slate Optimizers.

Allows seamless addition of new sports (Basketball, Tennis, etc.) without altering
core portfolio optimization or reporting logic.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import datetime

from src.core.domain import BetOpportunity, DailySlate, MatchEvent, SportType


class BaseOddsFeed(ABC):
    """Abstract interface for bookmaker odds scrapers/APIs across sports."""

    @abstractmethod
    def get_sport_type(self) -> SportType:
        """Returns the sport type handled by this feed."""
        pass

    @abstractmethod
    def fetch_live_events(self) -> List[MatchEvent]:
        """Fetches all upcoming events and bookmaker odds for the configured sport."""
        pass


class BaseSportInferenceEngine(ABC):
    """Abstract interface for mathematical/predictive sport models."""

    @abstractmethod
    def get_sport_type(self) -> SportType:
        """Returns the sport type this model handles."""
        pass

    @abstractmethod
    def evaluate_event(self, event: MatchEvent) -> List[BetOpportunity]:
        """Calculates true probabilities, EV, and candidate betting opportunities for an event."""
        pass

    def evaluate_all(self, events: List[MatchEvent]) -> List[BetOpportunity]:
        """Evaluates a batch of events and returns all actionable opportunities."""
        results = []
        for ev in events:
            if ev.sport == self.get_sport_type():
                results.extend(self.evaluate_event(ev))
        return results


class BaseSlateOptimizer(ABC):
    """Abstract interface for short-term/daily portfolio optimizers."""

    @abstractmethod
    def optimize_daily_slate(
        self,
        opportunities: List[BetOpportunity],
        target_date: datetime.date,
        daily_budget: float = 50.0,
        slate_size: int = 5,
        selection_mode: str = "safest_pewniaki"
    ) -> DailySlate:
        """Selects and allocates budget to the optimal subset of bets for a target day."""
        pass
