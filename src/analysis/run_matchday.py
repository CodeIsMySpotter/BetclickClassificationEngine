"""Multi-League Gameweek Runner & Real-Time Decision Engine.

Supports simultaneous monitoring and evaluation across multiple top leagues:
- La Liga (ESP)
- Premier League (ENG)
- Bundesliga (GER)

Executes Causal State Tracking (Elo + xG), calculates de-margined implied probabilities,
applies the Poisson-LGBM hybrid distribution, enforces stable-odds risk limits,
and outputs execution feeds for the dashboard.
"""

import argparse
from dataclasses import dataclass
import datetime
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.ingestion.betclic_feed import fetch_all_configured_leagues, save_betclic_odds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MultiLeagueMatchdayRunner")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.warning("Brak pliku config.yaml. Używanie parametrów domyślnych.")
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class TeamHistoricalProfile:
    team_name: str
    elo: float = 1500.0
    xg_for_history: List[float] = None
    xg_against_history: List[float] = None
    sot_for_history: List[int] = None
    sot_against_history: List[int] = None
    last_match_date: Optional[datetime.date] = None

    def __post_init__(self):
        if self.xg_for_history is None:
            self.xg_for_history = []
        if self.xg_against_history is None:
            self.xg_against_history = []
        if self.sot_for_history is None:
            self.sot_for_history = []
        if self.sot_against_history is None:
            self.sot_against_history = []

    def get_rolling_metrics(self, window: int = 5) -> Tuple[float, float, float]:
        if not self.xg_for_history:
            return 1.35, 1.35, 0.50

        sub_xg_f = self.xg_for_history[-window:]
        sub_xg_a = self.xg_against_history[-window:]
        sub_sot_f = self.sot_for_history[-window:]
        sub_sot_a = self.sot_against_history[-window:]

        roll_xg_f = float(np.mean(sub_xg_f))
        roll_xg_a = float(np.mean(sub_xg_a))
        tot_sot = sum(sub_sot_f) + sum(sub_sot_a)
        roll_sotr = (sum(sub_sot_f) / tot_sot) if tot_sot > 0 else 0.50

        return roll_xg_f, roll_xg_a, roll_sotr


class CausalStateManager:
    K_FACTOR: float = 24.0
    HOME_ADVANTAGE_ELO: float = 65.0
    SEASON_REVERSION_WEIGHT: float = 0.25

    def __init__(self) -> None:
        self.teams: Dict[str, TeamHistoricalProfile] = {}
        self.current_season: Optional[str] = None

    def _get_or_create(self, team_name: str) -> TeamHistoricalProfile:
        if team_name not in self.teams:
            self.teams[team_name] = TeamHistoricalProfile(team_name=team_name, elo=1480.0)
        return self.teams[team_name]

    def apply_season_mean_reversion(self) -> None:
        for profile in self.teams.values():
            profile.elo = (1.0 - self.SEASON_REVERSION_WEIGHT) * profile.elo + (self.SEASON_REVERSION_WEIGHT * 1500.0)

    def process_historical_matches(self, matches_df: pd.DataFrame, cutoff_date: datetime.date) -> None:
        df_sorted = matches_df.copy()
        df_sorted["date_dt"] = pd.to_datetime(df_sorted["date"]).dt.date
        df_sorted = df_sorted[df_sorted["date_dt"] < cutoff_date].sort_values("date_dt")

        logger.info("Rekonstrukcja stanu zespołów z 3 lig na podstawie %d spotkań (Cutoff: %s)...", len(df_sorted), cutoff_date)

        for _, row in df_sorted.iterrows():
            season = str(row["season"])
            if self.current_season is not None and season != self.current_season:
                self.apply_season_mean_reversion()
            self.current_season = season

            m_date = row["date_dt"]
            h_team = self._get_or_create(str(row["home_team"]))
            a_team = self._get_or_create(str(row["away_team"]))

            h_team.xg_for_history.append(float(row["home_xg"]))
            h_team.xg_against_history.append(float(row["away_xg"]))
            h_team.sot_for_history.append(int(row["home_sot"]))
            h_team.sot_against_history.append(int(row["away_sot"]))
            h_team.last_match_date = m_date

            a_team.xg_for_history.append(float(row["away_xg"]))
            a_team.xg_against_history.append(float(row["home_xg"]))
            a_team.sot_for_history.append(int(row["away_sot"]))
            a_team.sot_against_history.append(int(row["home_sot"]))
            a_team.last_match_date = m_date

            goals_h = int(row["home_goals"])
            goals_a = int(row["away_goals"])
            actual_score_h = 1.0 if goals_h > goals_a else (0.5 if goals_h == goals_a else 0.0)

            dr = (h_team.elo + self.HOME_ADVANTAGE_ELO) - a_team.elo
            we_h = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)

            h_team.elo += self.K_FACTOR * (actual_score_h - we_h)
            a_team.elo += self.K_FACTOR * ((1.0 - actual_score_h) - (1.0 - we_h))


class EnhancedInferenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.w_poisson = config.get("model_weights", {}).get("poisson", 0.93)
        self.w_lgbm = config.get("model_weights", {}).get("lightgbm", 0.07)
        self.min_ev = config.get("strategy", {}).get("min_ev", 0.05)
        self.kelly_frac = config.get("strategy", {}).get("kelly_fraction", 0.25)
        self.cap_pct = config.get("strategy", {}).get("max_cap_pct", 0.035)
        self.bankroll = config.get("general", {}).get("default_bankroll", 10000.0)
        
        self.min_odds = config.get("strategy", {}).get("min_odds", 1.20)
        self.max_odds = config.get("strategy", {}).get("max_odds", 2.20)
        self.min_model_prob = config.get("strategy", {}).get("min_model_prob", 0.50)
        
        self.tax_rate = config.get("betclic", {}).get("tax_rate", 0.0)
        self.apply_tax = config.get("betclic", {}).get("apply_tax", True)

    def _dixon_coles_grid(self, elo_diff: float, xg_diff: float, rho: float = -0.06) -> Tuple[np.ndarray, float, float, float]:
        """Rozkład bramek Dixon-Coles z korektą współzależności niskich wyników (0:0, 1:0, 0:1, 1:1).

        Parametr rho ujemny odzwierciedla zjawisko under-dispersion i podbija szansę remisów bezbramkowych.
        """
        lh = max(0.3, 1.45 + (0.0018 * elo_diff) + (0.35 * xg_diff))
        la = max(0.3, 1.05 - (0.0014 * elo_diff) - (0.28 * xg_diff))

        ph = [np.exp(-lh) * (lh**g) / math.factorial(g) for g in range(8)]
        pa = [np.exp(-la) * (la**g) / math.factorial(g) for g in range(8)]
        grid = np.outer(ph, pa)

        # Funkcja korekcyjna Dixon-Coles tau(x, y, lambda_h, lambda_a, rho)
        if lh * la > 0:
            tau_00 = 1.0 - (lh * la * rho)
            tau_01 = 1.0 + (lh * rho)
            tau_10 = 1.0 + (la * rho)
            tau_11 = 1.0 - rho

            grid[0, 0] = max(0.0, grid[0, 0] * tau_00)
            grid[0, 1] = max(0.0, grid[0, 1] * tau_01)
            grid[1, 0] = max(0.0, grid[1, 0] * tau_10)
            grid[1, 1] = max(0.0, grid[1, 1] * tau_11)

        prob_h = float(np.sum(np.tril(grid, -1)))
        prob_x = float(np.sum(np.diag(grid)))
        prob_a = float(np.sum(np.triu(grid, 1)))
        tot = prob_h + prob_x + prob_a
        return grid / tot, prob_h / tot, prob_x / tot, prob_a / tot

    def _lgbm_prior(self, elo_diff: float, xg_diff: float, rest_diff: int) -> Tuple[float, float, float]:
        zh = 0.15 + (0.0075 * elo_diff) + (0.85 * xg_diff) + (0.04 * rest_diff)
        zx = -0.45 - (0.0015 * abs(elo_diff))
        za = -0.15 - (0.0068 * elo_diff) - (0.75 * xg_diff) - (0.03 * rest_diff)
        eh, ex, ea = np.exp(zh), np.exp(zx), np.exp(za)
        tot = eh + ex + ea
        return eh / tot, ex / tot, ea / tot

    def evaluate_betclic_fixture(self, fixture: dict, state_mgr: CausalStateManager) -> dict:
        h_profile = state_mgr._get_or_create(fixture["home_team"])
        a_profile = state_mgr._get_or_create(fixture["away_team"])

        elo_diff = (h_profile.elo + state_mgr.HOME_ADVANTAGE_ELO) - a_profile.elo
        h_xg_f, h_xg_a, _ = h_profile.get_rolling_metrics(5)
        a_xg_f, a_xg_a, _ = a_profile.get_rolling_metrics(5)
        xg_diff = (h_xg_f - h_xg_a) - (a_xg_f - a_xg_a)

        f_date = pd.to_datetime(fixture["match_date"]).date() if fixture.get("match_date") else datetime.date.today()
        h_rest = (f_date - h_profile.last_match_date).days if h_profile.last_match_date else 7
        a_rest = (f_date - a_profile.last_match_date).days if a_profile.last_match_date else 7
        rest_diff = int(h_rest - a_rest)

        grid, p_poi_h, p_poi_x, p_poi_a = self._dixon_coles_grid(elo_diff, xg_diff)
        p_lgb_h, p_lgb_x, p_lgb_a = self._lgbm_prior(elo_diff, xg_diff, rest_diff)

        prob_h = round((self.w_poisson * p_poi_h) + (self.w_lgbm * p_lgb_h), 4)
        prob_x = round((self.w_poisson * p_poi_x) + (self.w_lgbm * p_lgb_x), 4)
        prob_a = round((self.w_poisson * p_poi_a) + (self.w_lgbm * p_lgb_a), 4)

        prob_over_25 = 0.0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if i + j > 2:
                    prob_over_25 += grid[i, j]
        prob_under_25 = 1.0 - prob_over_25
        prob_dnb_h = prob_h / (prob_h + prob_a) if (prob_h + prob_a) > 0 else 0.5

        odd_h = fixture["odds_home"]
        odd_x = fixture["odds_draw"]
        odd_a = fixture["odds_away"]
        inv_sum = (1.0 / odd_h) + (1.0 / odd_x) + (1.0 / odd_a)
        fair_h = round((1.0 / odd_h) / inv_sum, 4)

        # 1. Prawdopodobieństwo i syntetyczny kurs na 1X (Win or Draw)
        prob_1x = round(prob_h + prob_x, 4)
        # Syntetyczny kurs 1X = 1 / ((1/odd_h) + (1/odd_x))
        odd_1x = round(1.0 / ((1.0 / odd_h) + (1.0 / odd_x)), 2)

        # 2. Prawdopodobieństwo i kurs na Draw No Bet (DNB - Remis = Zwrot)
        prob_dnb = round(prob_h / (prob_h + prob_a), 4) if (prob_h + prob_a) > 0 else 0.5
        # Syntetyczny kurs DNB = odd_h * (1 - 1/odd_x)
        odd_dnb = round(odd_h * (1.0 - (1.0 / odd_x)), 2) if odd_x > 1.0 else odd_h

        tax_multiplier = (1.0 - self.tax_rate) if self.apply_tax else 1.0

        # Wartości oczekiwane (EV) dla każdego wariantu
        ev_home = round((prob_h * (odd_h * tax_multiplier)) - 1.0, 4)
        ev_1x = round((prob_1x * (odd_1x * tax_multiplier)) - 1.0, 4)
        ev_dnb = round((prob_dnb * (odd_dnb * tax_multiplier)) - 1.0, 4)

        # Ocena profilu inwestycyjnego (Capital Preservation):
        # Preferujemy rynek o najwyższej pewności (prob >= min_model_prob), dodatnim EV i w bezpiecznym zakresie kursów
        candidates = []
        
        # Kandydat A: 1X (Najwyższa ochrona kapitału, Win or Draw)
        if prob_1x >= self.min_model_prob and (self.min_odds <= odd_1x <= self.max_odds) and ev_1x >= self.min_ev:
            candidates.append({
                "market_type": "1X (Win or Draw)",
                "recommended_odd": odd_1x,
                "win_prob": prob_1x,
                "ev": ev_1x,
                "b": (odd_1x * tax_multiplier) - 1.0
            })
            
        # Kandydat B: DNB (Remis = Zwrot)
        if prob_dnb >= self.min_model_prob and (self.min_odds <= odd_dnb <= self.max_odds) and ev_dnb >= self.min_ev:
            candidates.append({
                "market_type": "DNB (Remis = Zwrot)",
                "recommended_odd": odd_dnb,
                "win_prob": prob_dnb,
                "ev": ev_dnb,
                "b": (odd_dnb * tax_multiplier) - 1.0
            })

        # Kandydat C: Czysta wygrana gospodarza (1)
        if prob_h >= self.min_model_prob and (self.min_odds <= odd_h <= self.max_odds) and ev_home >= self.min_ev:
            candidates.append({
                "market_type": "Czysta wygrana (1)",
                "recommended_odd": odd_h,
                "win_prob": prob_h,
                "ev": ev_home,
                "b": (odd_h * tax_multiplier) - 1.0
            })

        if candidates:
            # Sortujemy wg szansy wygranej (bezpieczeństwo kapitału), a przy zbliżonej szansie wg EV
            best = sorted(candidates, key=lambda c: (c["win_prob"], c["ev"]), reverse=True)[0]
            decision = "EXECUTE_BET"
            market_chosen = best["market_type"]
            rec_odd = best["recommended_odd"]
            rec_prob = best["win_prob"]
            rec_ev = best["ev"]
            
            effective_b = best["b"]
            full_k = rec_ev / effective_b if effective_b > 0 else 0.0
            adj_k = min(self.cap_pct, full_k * self.kelly_frac)
            stake_units = round(adj_k * self.bankroll, 2)
            stake_pct = round(adj_k * 100.0, 2)
        else:
            decision = "PASS"
            market_chosen = "Brak"
            rec_odd = odd_h
            rec_prob = prob_h
            rec_ev = ev_home
            stake_units = 0.0
            stake_pct = 0.0

        return {
            "league_name": fixture.get("league_name", "La Liga"),
            "match_id": fixture["match_id"],
            "match_date": str(fixture["match_date"]),
            "match": f"{fixture['home_team']} vs {fixture['away_team']}",
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "elo_diff": round(elo_diff, 1),
            "roll_xg_diff_5m": round(xg_diff, 2),
            "rest_days_diff": rest_diff,
            "prob_home": prob_h,
            "prob_draw": prob_x,
            "prob_away": prob_a,
            "prob_1x": prob_1x,
            "prob_dnb": prob_dnb,
            "odds_home": odd_h,
            "odds_draw": odd_x,
            "odds_away": odd_a,
            "odds_1x": odd_1x,
            "odds_dnb": odd_dnb,
            "ev_home": ev_home,
            "ev_1x": ev_1x,
            "ev_dnb": ev_dnb,
            "decision": decision,
            "market_chosen": market_chosen,
            "rec_odd": rec_odd,
            "rec_prob": rec_prob,
            "rec_ev": rec_ev,
            "stake_pct": stake_pct,
            "stake_units": stake_units,
        }


def run_matchday_pipeline():
    logger.info("=== START: MULTI-LEAGUE MATCHDAY INTELLIGENCE RUN ===")
    config = load_config()
    leagues_cfg = config.get("leagues", [])

    # 1. Pobieranie kursów ze wszystkich skonfigurowanych lig
    df_all_odds = fetch_all_configured_leagues(leagues_cfg)

    # 2. Wczytanie historii i odtworzenie profilu Elo/xG
    hist_file = BASE_DIR / "src" / "dashboard" / "matches_history.csv"
    if not hist_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku historii: {hist_file}")
    df_hist = pd.read_csv(hist_file)

    cutoff_date = pd.to_datetime(df_all_odds["match_date"].min()).date()
    state_mgr = CausalStateManager()
    state_mgr.process_historical_matches(df_hist, cutoff_date=cutoff_date)

    # 3. Ewaluacja predykcyjna i Value Betting
    engine = EnhancedInferenceEngine(config)
    fixtures_records = df_all_odds.to_dict(orient="records")

    logger.info("Przetwarzanie %d spotkań ze wszystkich lig...", len(fixtures_records))
    results = [engine.evaluate_betclic_fixture(f, state_mgr) for f in fixtures_records]
    df_results = pd.DataFrame(results)

    # 4. Zapis artefaktów
    csv_out = ARTIFACTS_DIR / "gameweek_live_predictions.csv"
    json_out = ARTIFACTS_DIR / "gameweek_live_predictions.json"

    df_results.to_csv(csv_out, index=False, encoding="utf-8")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Pomyślnie zaktualizowano artefakty live:")
    logger.info("-> %s", csv_out)
    logger.info("-> %s", json_out)

    # 5. Generator Optymalnych Dubli Inwestycyjnych (2-Leg Parlays)
    parlays = generate_optimal_parlays(df_results, bankroll=engine.bankroll, tax_rate=engine.tax_rate if engine.apply_tax else 0.0)
    parlays_json = ARTIFACTS_DIR / "optimal_parlays.json"
    with open(parlays_json, "w", encoding="utf-8") as f:
        json.dump(parlays, f, indent=2, ensure_ascii=False)

    # 6. Podsumowanie decyzji
    bets_to_execute = df_results[df_results["decision"] == "EXECUTE_BET"]
    print("\n" + "=" * 80)
    print("PORTFEL INWESTYCYJNY: CAPITAL PRESERVATION (1X / DNB / 1)")
    print(f"Zakres kursów: {engine.min_odds:.2f} - {engine.max_odds:.2f} | Min Pewność: {engine.min_model_prob*100:.0f}%")
    print("=" * 80)
    if bets_to_execute.empty:
        print("Brak rekomendowanych zakładów o zadanych kryteriach w tej kolejce.")
    else:
        print(f"Znaleziono {len(bets_to_execute)} ultra-stabilnych pozycji inwestycyjnych:")
        for _, b in bets_to_execute.iterrows():
            print(f"[*] [{b['league_name']}] {b['match']} ({b['match_date']})")
            print(f"    Rekomendowany rynek: >>> {b['market_chosen']} <<<")
            print(f"    Kurs: {b['rec_odd']} | Pewność modelu: {b['rec_prob']*100:.1f}% | EV: {b['rec_ev']*100:+.2f}%")
            print(f"    Stawka Quarter-Kelly: {b['stake_pct']}% bankrolla ({b['stake_units']} PLN)\n")

    if parlays:
        print("=" * 80)
        print(f"OPTYMALNE DUBLE INWESTYCYJNE (2-Leg Parlay): Znaleziono {len(parlays)} zbalansowanych par")
        print("=" * 80)
        for p in parlays[:3]:
            print(f"[*] DUBEL (Kurs łączny: {p['combined_odds']} | Szansa: {p['combined_prob']*100:.1f}% | EV: {p['ev']*100:+.1f}%)")
            print(f"    Noga 1: {p['leg1']['match']} -> {p['leg1']['market']} (Kurs: {p['leg1']['odd']})")
            print(f"    Noga 2: {p['leg2']['match']} -> {p['leg2']['market']} (Kurs: {p['leg2']['odd']})")
            print(f"    Rekomendowana stawka: {p['stake_pct']}% banku ({p['stake_units']} PLN)\n")

    print("=" * 80 + "\n")
    return df_results


def generate_optimal_parlays(df_results: pd.DataFrame, bankroll: float = 10000.0, tax_rate: float = 0.0) -> List[dict]:
    """Generuje matematycznie zoptymalizowane kupony 2-zdarzeniowe (duble) o najwyższej pewności."""
    actionable = df_results[df_results["decision"] == "EXECUTE_BET"].copy()
    if len(actionable) < 2:
        return []

    # Sortujemy od najwyższego iloczynu pewności i EV
    candidates = actionable.sort_values(by=["rec_prob", "rec_ev"], ascending=False).to_dict(orient="records")
    parlays = []

    # Łączymy w pary niezależne mecze (z różnych terminów lub lig, aby zminimalizować korelację)
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            c1 = candidates[i]
            c2 = candidates[j]

            # Odrzucamy mecze tych samych drużyn
            if c1["home_team"] in [c2["home_team"], c2["away_team"]] or c1["away_team"] in [c2["home_team"], c2["away_team"]]:
                continue

            comb_odds = round(c1["rec_odd"] * c2["rec_odd"], 2)
            # Optymalne widełki dubla: kurs 1.50 - 2.10
            if not (1.45 <= comb_odds <= 2.20):
                continue

            comb_prob = round(c1["rec_prob"] * c2["rec_prob"], 4)
            # Szansa dubla musi wynosić minimum 65%
            if comb_prob < 0.65:
                continue

            tax_mult = (1.0 - tax_rate)
            ev = round((comb_prob * (comb_odds * tax_mult)) - 1.0, 4)
            if ev < 0.05:
                continue

            # Fractional Kelly na dubel (15% Kelly dla bezpieczeństwa)
            b = (comb_odds * tax_mult) - 1.0
            full_k = ev / b if b > 0 else 0.0
            adj_k = min(0.025, full_k * 0.15)
            stake_units = round(adj_k * bankroll, 2)
            stake_pct = round(adj_k * 100.0, 2)

            parlays.append({
                "combined_odds": comb_odds,
                "combined_prob": comb_prob,
                "ev": ev,
                "stake_pct": stake_pct,
                "stake_units": stake_units,
                "leg1": {
                    "match": c1["match"],
                    "league": c1["league_name"],
                    "market": c1["market_chosen"],
                    "odd": c1["rec_odd"],
                    "prob": c1["rec_prob"]
                },
                "leg2": {
                    "match": c2["match"],
                    "league": c2["league_name"],
                    "market": c2["market_chosen"],
                    "odd": c2["rec_odd"],
                    "prob": c2["rec_prob"]
                }
            })

    # Sortujemy duble według najniższego ryzyka i najwyższego EV
    return sorted(parlays, key=lambda p: (p["combined_prob"], p["ev"]), reverse=True)[:5]


if __name__ == "__main__":
    run_matchday_pipeline()
