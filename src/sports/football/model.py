"""Football Inference Engine (Dixon-Coles + LightGBM + Causal State).

Implements BaseSportInferenceEngine to evaluate football MatchEvents and produce
quantified BetOpportunity records.
"""

import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.domain import BetOpportunity, MatchEvent, SportType
from src.core.interfaces import BaseSportInferenceEngine

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class FootballHistoricalProfile:
    def __init__(self, team_name: str, elo: float = 1480.0):
        self.team_name = team_name
        self.elo = elo
        self.xg_for: List[float] = []
        self.xg_against: List[float] = []
        self.last_match_date: Optional[datetime.date] = None

    def get_rolling_xg(self, window: int = 5) -> Tuple[float, float]:
        rf = np.mean(self.xg_for[-window:]) if self.xg_for else 1.35
        ra = np.mean(self.xg_against[-window:]) if self.xg_against else 1.35
        return float(rf), float(ra)


class FootballInferenceEngine(BaseSportInferenceEngine):
    """Predictive engine for football using Dixon-Coles and form priors."""

    K_FACTOR: float = 24.0
    HOME_ADVANTAGE_ELO: float = 65.0
    SEASON_REVERSION_WEIGHT: float = 0.25

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.teams: Dict[str, FootballHistoricalProfile] = {}
        self.w_poisson = self.config.get("model_weights", {}).get("poisson", 0.93)
        self.w_lgbm = self.config.get("model_weights", {}).get("lightgbm", 0.07)
        
        strat = self.config.get("strategy", {})
        self.min_ev = strat.get("min_ev", 0.04)
        self.max_ev = strat.get("max_ev", 0.18)      # Asymmetric Value Trap ceiling
        self.min_prob = strat.get("min_model_prob", 0.70)
        self.min_odds = strat.get("min_odds", 1.15)
        self.max_odds = strat.get("max_odds", 1.85)

        self._initialize_history()

    def get_sport_type(self) -> SportType:
        return SportType.FOOTBALL

    def _get_or_create(self, name: str) -> FootballHistoricalProfile:
        if name not in self.teams:
            self.teams[name] = FootballHistoricalProfile(team_name=name)
        return self.teams[name]

    def _initialize_history(self):
        hist_file = BASE_DIR / "src" / "dashboard" / "matches_history.csv"
        if not hist_file.exists():
            return
        df_hist = pd.read_csv(hist_file)
        df_hist["date_dt"] = pd.to_datetime(df_hist["date"]).dt.date
        df_sorted = df_hist.sort_values("date_dt")

        cur_season = None
        for _, row in df_sorted.iterrows():
            season = str(row["season"])
            if cur_season is not None and season != cur_season:
                for p in self.teams.values():
                    p.elo = (1.0 - self.SEASON_REVERSION_WEIGHT) * p.elo + (self.SEASON_REVERSION_WEIGHT * 1500.0)
            cur_season = season

            m_date = row["date_dt"]
            ht = self._get_or_create(str(row["home_team"]))
            at = self._get_or_create(str(row["away_team"]))

            ht.xg_for.append(float(row["home_xg"]))
            ht.xg_against.append(float(row["away_xg"]))
            ht.last_match_date = m_date

            at.xg_for.append(float(row["away_xg"]))
            at.xg_against.append(float(row["home_xg"]))
            at.last_match_date = m_date

            gh, ga = int(row["home_goals"]), int(row["away_goals"])
            act = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
            dr = (ht.elo + self.HOME_ADVANTAGE_ELO) - at.elo
            we = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
            ht.elo += self.K_FACTOR * (act - we)
            at.elo += self.K_FACTOR * ((1.0 - act) - (1.0 - we))

    def _dixon_coles(self, elo_diff: float, xg_diff: float, rho: float = -0.06) -> Tuple[float, float, float]:
        lh = max(0.3, 1.45 + (0.0018 * elo_diff) + (0.35 * xg_diff))
        la = max(0.3, 1.05 - (0.0014 * elo_diff) - (0.28 * xg_diff))
        ph = [np.exp(-lh) * (lh**g) / math.factorial(g) for g in range(8)]
        pa = [np.exp(-la) * (la**g) / math.factorial(g) for g in range(8)]
        grid = np.outer(ph, pa)
        if lh * la > 0:
            grid[0, 0] = max(0.0, grid[0, 0] * (1.0 - lh * la * rho))
            grid[0, 1] = max(0.0, grid[0, 1] * (1.0 + lh * rho))
            grid[1, 0] = max(0.0, grid[1, 0] * (1.0 + la * rho))
            grid[1, 1] = max(0.0, grid[1, 1] * (1.0 - rho))
        ph_tot = float(np.sum(np.tril(grid, -1)))
        px_tot = float(np.sum(np.diag(grid)))
        pa_tot = float(np.sum(np.triu(grid, 1)))
        tot = ph_tot + px_tot + pa_tot
        return ph_tot / tot, px_tot / tot, pa_tot / tot

    def _lgbm_prior(self, elo_diff: float, xg_diff: float, rest_diff: int = 0) -> Tuple[float, float, float]:
        zh = 0.15 + (0.0075 * elo_diff) + (0.85 * xg_diff) + (0.04 * rest_diff)
        zx = -0.45 - (0.0015 * abs(elo_diff))
        za = -0.15 - (0.0068 * elo_diff) - (0.75 * xg_diff) - (0.03 * rest_diff)
        eh, ex, ea = np.exp(zh), np.exp(zx), np.exp(za)
        tot = eh + ex + ea
        return eh / tot, ex / tot, ea / tot

    def evaluate_event(self, event: MatchEvent) -> List[BetOpportunity]:
        ht = self._get_or_create(event.home_competitor)
        at = self._get_or_create(event.away_competitor)

        elo_diff = (ht.elo + self.HOME_ADVANTAGE_ELO) - at.elo
        hf, ha = ht.get_rolling_xg(5)
        af, aa = at.get_rolling_xg(5)
        xg_diff = (hf - ha) - (af - aa)

        h_rest = (event.event_date - ht.last_match_date).days if ht.last_match_date else 7
        a_rest = (event.event_date - at.last_match_date).days if at.last_match_date else 7
        rest_diff = int(h_rest - a_rest)

        p_dc_h, p_dc_x, p_dc_a = self._dixon_coles(elo_diff, xg_diff)
        p_lg_h, p_lg_x, p_lg_a = self._lgbm_prior(elo_diff, xg_diff, rest_diff)

        prob_h = (self.w_poisson * p_dc_h) + (self.w_lgbm * p_lg_h)
        prob_x = (self.w_poisson * p_dc_x) + (self.w_lgbm * p_lg_x)
        prob_a = (self.w_poisson * p_dc_a) + (self.w_lgbm * p_lg_a)

        prob_1x = prob_h + prob_x
        prob_dnb = prob_h / (prob_h + prob_a) if (prob_h + prob_a) > 0 else 0.5

        odd_h = event.odds.get("1", 0.0)
        odd_1x = event.odds.get("1X", 0.0)
        odd_dnb = event.odds.get("DNB", 0.0)

        candidates: List[BetOpportunity] = []

        # Helper to compute confidence score
        # Higher probability + positive edge + low odds variance = higher confidence score
        def calc_confidence(prob: float, ev: float, odd: float) -> float:
            variance = prob * (1.0 - prob)
            base_score = (prob ** 2) * (1.0 + ev) / max(0.05, variance)
            return round(base_score * 10.0, 2)

        # 1. Rynek 1X (Win or Draw)
        if odd_1x > 1.0:
            ev_1x = (prob_1x * odd_1x) - 1.0
            fair_1x = 1.0 / prob_1x if prob_1x > 0 else 99.0
            is_trap = ev_1x > self.max_ev
            score = calc_confidence(prob_1x, ev_1x, odd_1x)
            candidates.append(BetOpportunity(
                event_id=event.event_id,
                sport=SportType.FOOTBALL,
                league_name=event.league_name,
                match_display=event.display_name,
                event_date=event.event_date,
                market_name="1X (Win or Draw)",
                odds=odd_1x,
                model_probability=prob_1x,
                fair_odds=fair_1x,
                expected_value=ev_1x,
                confidence_score=score,
                is_value_trap=is_trap,
                rationale=f"Elo Diff: {elo_diff:+.1f}, xG Diff: {xg_diff:+.2f}, szansa wygranej/remisu: {prob_1x*100:.1f}%"
            ))

        # 2. Rynek DNB (Remis = Zwrot)
        if odd_dnb > 1.0:
            ev_dnb = (prob_dnb * odd_dnb) - 1.0
            fair_dnb = 1.0 / prob_dnb if prob_dnb > 0 else 99.0
            is_trap = ev_dnb > self.max_ev
            score = calc_confidence(prob_dnb, ev_dnb, odd_dnb)
            candidates.append(BetOpportunity(
                event_id=event.event_id,
                sport=SportType.FOOTBALL,
                league_name=event.league_name,
                match_display=event.display_name,
                event_date=event.event_date,
                market_name="DNB (Remis = Zwrot)",
                odds=odd_dnb,
                model_probability=prob_dnb,
                fair_odds=fair_dnb,
                expected_value=ev_dnb,
                confidence_score=score,
                is_value_trap=is_trap,
                rationale=f"Pewność bez remisu: {prob_dnb*100:.1f}%, EV: {ev_dnb*100:+.1f}%"
            ))

        # 3. Czysta 1
        if odd_h > 1.0:
            ev_h = (prob_h * odd_h) - 1.0
            fair_h = 1.0 / prob_h if prob_h > 0 else 99.0
            is_trap = ev_h > self.max_ev
            score = calc_confidence(prob_h, ev_h, odd_h)
            candidates.append(BetOpportunity(
                event_id=event.event_id,
                sport=SportType.FOOTBALL,
                league_name=event.league_name,
                match_display=event.display_name,
                event_date=event.event_date,
                market_name="Czysta wygrana (1)",
                odds=odd_h,
                model_probability=prob_h,
                fair_odds=fair_h,
                expected_value=ev_h,
                confidence_score=score,
                is_value_trap=is_trap,
                rationale=f"Dominacja gospodarzy (Pewność {prob_h*100:.1f}%, kurs {odd_h:.2f})"
            ))

        return candidates
