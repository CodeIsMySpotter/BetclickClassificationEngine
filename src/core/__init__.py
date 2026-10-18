"""Core domain models and interfaces for Betclic Multi-Sport Intelligence."""

from src.core.domain import BetOpportunity, DailySlate, MatchEvent, SportType
from src.core.interfaces import BaseOddsFeed, BaseSlateOptimizer, BaseSportInferenceEngine

__all__ = [
    "SportType",
    "MatchEvent",
    "BetOpportunity",
    "DailySlate",
    "BaseOddsFeed",
    "BaseSportInferenceEngine",
    "BaseSlateOptimizer",
]
