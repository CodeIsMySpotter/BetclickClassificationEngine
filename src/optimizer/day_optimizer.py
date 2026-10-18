"""Day Return Optimizer (Dzienny Optymalizator Zwrotu).

Selects and sizes the optimal slate of several to a dozen (3-15) highest-confidence,
positive-EV bets for a single day to maximize short-term daily return while preserving capital.
"""

from typing import Dict, List, Optional
import datetime
import numpy as np

from src.core.domain import BetOpportunity, DailySlate, SportType
from src.core.interfaces import BaseSlateOptimizer


class DayReturnOptimizer(BaseSlateOptimizer):
    """Optimizes daily portfolios across single or multiple sports."""

    def __init__(
        self,
        min_probability: float = 0.78, # Podniesiony próg pewności (wycina ujemny ROI z przedziału 50-75%)
        min_ev: float = 0.04,
        max_ev: float = 0.18,          # Asymmetric Value Trap Ceiling
        min_odds: float = 1.15,
        max_odds: float = 1.60,        # Zawężone widełki kursowe (odcina -15% strat powyżej 1.60)
        max_odds_1x: float = 1.25,     # Pułapka Fake Favorite: 1X tylko do kursu 1.25!
        min_bet_stake_pln: float = 2.00,
    ):
        self.min_probability = min_probability
        self.min_ev = min_ev
        self.max_ev = max_ev
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.max_odds_1x = max_odds_1x
        self.min_bet_stake_pln = min_bet_stake_pln

    def filter_actionable(
        self,
        opportunities: List[BetOpportunity],
        target_date: Optional[datetime.date] = None,
        sport_filter: Optional[SportType] = None
    ) -> List[BetOpportunity]:
        """Applies mathematical filters and removes value traps."""
        actionable = []
        for opp in opportunities:
            if target_date is not None and opp.event_date != target_date:
                continue
            if sport_filter is not None and opp.sport != sport_filter:
                continue
            
            # Trap ceiling check: avoid bookmaker asymmetry anomalies
            if opp.is_value_trap or opp.expected_value > self.max_ev:
                continue

            # Pułapka Fake Favorite: 1X tylko do kursu 1.25
            if "1X" in opp.market_name and opp.odds > self.max_odds_1x:
                continue
            
            # Early season uncertainty check (August / September)
            is_early = opp.event_date.month in [8, 9] if hasattr(opp.event_date, "month") else False
            req_prob = max(self.min_probability, 0.80) if is_early else self.min_probability

            # Core quantitative bounds
            if (
                opp.model_probability >= req_prob and
                self.min_ev <= opp.expected_value <= self.max_ev and
                self.min_odds <= opp.odds <= self.max_odds
            ):
                actionable.append(opp)

        return actionable

    def optimize_daily_slate(
        self,
        opportunities: List[BetOpportunity],
        target_date: datetime.date,
        daily_budget: float = 50.0,
        slate_size: int = 5,
        selection_mode: str = "safest_pewniaki",
        sport_filter: Optional[str] = None
    ) -> DailySlate:
        """Constructs the optimal daily betting portfolio for the target day.

        Modes:
        - 'safest_pewniaki': Maximizes hit rate by ranking primarily by win probability.
        - 'max_return': Maximizes expected ROI by ranking by EV.
        - 'balanced_sharpe': Balances probability, EV, and variance using confidence score.
        """
        # 1. Filtrowanie daty i kryteriów
        s_filter = SportType(sport_filter) if sport_filter else None
        candidates = self.filter_actionable(opportunities, target_date=target_date, sport_filter=s_filter)

        # 2. Deduplikacja: wybieramy maksymalnie 1 rynek na dane spotkanie
        event_grouped: Dict[str, List[BetOpportunity]] = {}
        for c in candidates:
            event_grouped.setdefault(c.event_id, []).append(c)

        deduped: List[BetOpportunity] = []
        for e_id, opps in event_grouped.items():
            dnb_opt = next((o for o in opps if "DNB" in o.market_name), None)
            x1_opt = next((o for o in opps if "1X" in o.market_name), None)
            if selection_mode == "safest_pewniaki":
                # Preferencja DNB nad 1X dla ochrony kapitału (zwrot stawki przy remisie)
                if dnb_opt and x1_opt and dnb_opt.model_probability >= 0.75:
                    best = dnb_opt
                else:
                    best = sorted(opps, key=lambda x: (x.model_probability, x.expected_value), reverse=True)[0]
            elif selection_mode == "max_return":
                best = sorted(opps, key=lambda x: (x.expected_value, x.model_probability), reverse=True)[0]
            else:  # balanced_sharpe
                best = sorted(opps, key=lambda x: (x.confidence_score, x.expected_value), reverse=True)[0]
            deduped.append(best)

        # 3. Sortowanie globalne wg wybranego trybu
        if selection_mode == "safest_pewniaki":
            ranked = sorted(deduped, key=lambda x: (x.model_probability, x.confidence_score), reverse=True)
        elif selection_mode == "max_return":
            ranked = sorted(deduped, key=lambda x: (x.expected_value, x.model_probability), reverse=True)
        else:
            ranked = sorted(deduped, key=lambda x: (x.confidence_score, x.model_probability), reverse=True)

        # 4. Wybór żądanej liczby pozycji (kilka do kilkunastu betów)
        selected = ranked[:slate_size]

        if not selected:
            return DailySlate(
                target_date=target_date,
                sport_filter=sport_filter,
                selection_mode=selection_mode,
                total_budget_pln=daily_budget,
                selections=[]
            )

        # 5. Dynamiczny podział budżetu dziennego (Staking Allocation)
        # Obliczamy wagi na podstawie Confidence Score lub Probability
        scores = np.array([max(1.0, s.confidence_score) for s in selected])
        weights = scores / np.sum(scores)

        # Maksymalny limit stawki na 1 kupon: 25% budżetu dziennego, min. 2.00 zł
        total_allocated = 0.0
        for i, s in enumerate(selected):
            raw_stake = round(daily_budget * weights[i], 2)
            # Ograniczenia
            stake = max(self.min_bet_stake_pln, min(daily_budget * 0.25, raw_stake))
            s.recommended_stake_pln = stake
            s.kelly_stake_pct = round((stake / daily_budget) * 100.0, 1)
            total_allocated += stake

        # Normalizacja jeśli suma stawek nieznacznie przekracza budżet
        if total_allocated > daily_budget:
            scale = daily_budget / total_allocated
            total_allocated = 0.0
            for s in selected:
                s.recommended_stake_pln = max(self.min_bet_stake_pln, round(s.recommended_stake_pln * scale, 2))
                s.kelly_stake_pct = round((s.recommended_stake_pln / daily_budget) * 100.0, 1)
                total_allocated += s.recommended_stake_pln

        # 6. Metryki portfela dziennego
        avg_odd = float(np.mean([s.odds for s in selected]))
        avg_prob = float(np.mean([s.model_probability for s in selected]))
        avg_ev = float(np.mean([s.expected_value for s in selected]))
        
        expected_profit = sum(s.recommended_stake_pln * s.expected_value for s in selected)
        expected_roi = (expected_profit / total_allocated * 100.0) if total_allocated > 0 else 0.0

        # 7. Symulacje i scenariusze dnia (Stress-Testing)
        probs = np.array([s.model_probability for s in selected])
        prob_all_win = float(np.prod(probs))
        
        # Prawdopodobieństwo, że co najmniej (N-1) wygra (Poisson Binomial approximation)
        n = len(selected)
        if n >= 2:
            p_fail_none = float(np.prod(probs))
            p_fail_one = sum(
                (1.0 - probs[j]) * np.prod([probs[k] for k in range(n) if k != j])
                for j in range(n)
            )
            prob_at_least_80 = float(p_fail_none + p_fail_one)
        else:
            prob_at_least_80 = prob_all_win

        profit_if_all_win = sum(s.recommended_stake_pln * (s.odds - 1.0) for s in selected)
        
        # Scenariusz: odpada 1 zdarzenie o najniższym kursie
        worst_loss_odd = sorted(selected, key=lambda x: x.odds)[0]
        profit_if_one_loses = sum(
            s.recommended_stake_pln * (s.odds - 1.0) for s in selected if s != worst_loss_odd
        ) - worst_loss_odd.recommended_stake_pln

        return DailySlate(
            target_date=target_date,
            sport_filter=sport_filter,
            selection_mode=selection_mode,
            total_budget_pln=daily_budget,
            selections=selected,
            total_stake_pln=round(total_allocated, 2),
            avg_odds=round(avg_odd, 2),
            avg_probability=round(avg_prob, 4),
            avg_ev=round(avg_ev, 4),
            expected_daily_profit_pln=round(expected_profit, 2),
            expected_daily_roi_pct=round(expected_roi, 2),
            prob_all_win=round(prob_all_win, 4),
            prob_at_least_80pct_win=round(prob_at_least_80, 4),
            profit_if_all_win_pln=round(profit_if_all_win, 2),
            profit_if_one_loses_pln=round(profit_if_one_loses, 2)
        )

    def generate_daily_parlays(
        self,
        slate: DailySlate,
        max_parlays: int = 3
    ) -> List[Dict]:
        """Combines top daily bets into optimal independent doubles for the day."""
        selections = slate.selections
        if len(selections) < 2:
            return []

        # Sort by confidence
        sorted_bets = sorted(selections, key=lambda b: (b.confidence_score, b.model_probability), reverse=True)
        parlays = []

        for i in range(len(sorted_bets)):
            for j in range(i + 1, len(sorted_bets)):
                b1 = sorted_bets[i]
                b2 = sorted_bets[j]

                # Independent events
                if b1.event_id == b2.event_id:
                    continue

                comb_odds = round(b1.odds * b2.odds, 2)
                if not (1.45 <= comb_odds <= 2.20):
                    continue

                comb_prob = round(b1.model_probability * b2.model_probability, 4)
                if comb_prob < 0.60:
                    continue

                comb_ev = round((comb_prob * comb_odds) - 1.0, 4)
                if comb_ev < 0.05:
                    continue

                # Stake proportional to daily budget
                parlay_stake = max(self.min_bet_stake_pln, round(slate.total_budget_pln * 0.10, 2))
                exp_profit = round(parlay_stake * comb_ev, 2)

                parlays.append({
                    "leg1": {"match": b1.match_display, "market": b1.market_name, "odds": b1.odds, "prob": b1.model_probability},
                    "leg2": {"match": b2.match_display, "market": b2.market_name, "odds": b2.odds, "prob": b2.model_probability},
                    "combined_odds": comb_odds,
                    "combined_prob": comb_prob,
                    "ev": comb_ev,
                    "stake_pln": parlay_stake,
                    "expected_profit_pln": exp_profit,
                    "potential_return_pln": round(parlay_stake * comb_odds, 2)
                })

                if len(parlays) >= max_parlays:
                    return parlays

        return parlays
