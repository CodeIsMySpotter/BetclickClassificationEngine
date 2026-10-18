"""CLI runner to generate and inspect optimized daily slates.

Usage:
  python -m src.analysis.run_daily_slate --date 2026-10-17 --budget 50 --size 5 --mode safest_pewniaki
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.domain import SportType
from src.optimizer.day_optimizer import DayReturnOptimizer
from src.sports.football.feed import FootballBetclicFeed
from src.sports.football.model import FootballInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DailySlateRunner")

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_daily_optimizer(
    target_date_str: str,
    daily_budget: float = 50.0,
    slate_size: int = 5,
    selection_mode: str = "safest_pewniaki",
    sport_name: str = "football"
):
    target_date = datetime.date.fromisoformat(target_date_str)
    
    # 1. Wczytanie konfiguracji
    cfg_file = BASE_DIR / "config" / "config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Pobranie zdarzeń i predykcji (dla piłki nożnej, modułowo gotowe pod kosz/tenis)
    feed = FootballBetclicFeed(leagues_config=config.get("leagues", []))
    events = feed.fetch_live_events(force_refresh=False)
    
    engine = FootballInferenceEngine(config=config)
    opportunities = engine.evaluate_all(events)
    logger.info("Wygenerowano %d możliwości inwestycyjnych dla %d spotkań.", len(opportunities), len(events))

    # 3. Uruchomienie Day Return Optimizer
    optimizer = DayReturnOptimizer(
        min_probability=config.get("strategy", {}).get("min_model_prob", 0.70),
        min_ev=config.get("strategy", {}).get("min_ev", 0.04),
        max_ev=config.get("strategy", {}).get("max_ev", 0.18),
        min_odds=config.get("strategy", {}).get("min_odds", 1.15),
        max_odds=config.get("strategy", {}).get("max_odds", 1.85),
    )

    slate = optimizer.optimize_daily_slate(
        opportunities=opportunities,
        target_date=target_date,
        daily_budget=daily_budget,
        slate_size=slate_size,
        selection_mode=selection_mode,
        sport_filter=sport_name
    )

    parlays = optimizer.generate_daily_parlays(slate, max_parlays=3)

    # 4. Zapis do artefaktów
    out_json = ARTIFACTS_DIR / "daily_slate_latest.json"
    data_to_save = slate.to_dict()
    data_to_save["daily_parlays"] = parlays
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    # 5. Prezentacja wyników
    print("\n" + "=" * 95)
    print(f"OPTYMALNY ROZKŁAD DZIENNY (DAY RETURN OPTIMIZER) - DATA: {slate.target_date}")
    print(f"Tryb: {selection_mode.upper()} | Budżet na dzień: {daily_budget:.2f} PLN | Rozmiar: {len(slate.selections)} betów")
    print("=" * 95)
    
    if not slate.selections:
        print(f"Brak zakładów spełniających rygorystyczne kryteria bezpieczeństwa dla dnia {target_date}.")
        print("Dostępne daty w bieżącej ofercie:", sorted(list(set(str(e.event_date) for e in events))))
        print("=" * 95 + "\n")
        return slate

    for i, b in enumerate(slate.selections, 1):
        print(f"[{i}] [{b.league_name}] {b.match_display}")
        print(f"    Rekomendowany rynek: >>> {b.market_name} <<<")
        print(f"    Kurs: {b.odds:.2f} (Fair: {b.fair_odds:.2f}) | Pewność modelu: {b.model_probability*100:.1f}% | EV: {b.expected_value*100:+.1f}%")
        print(f"    Stawka dzienna: {b.recommended_stake_pln:.2f} PLN ({b.kelly_stake_pct}% budżetu dnia) | Score: {b.confidence_score:.1f}")
        print(f"    Uzasadnienie: {b.rationale}\n")

    print("-" * 95)
    print(f"PODSUMOWANIE PORTFELA DZIENNEGO:")
    print(f"- Łączna stawka: {slate.total_stake_pln:.2f} PLN / {daily_budget:.2f} PLN")
    print(f"- Średni kurs: {slate.avg_odds:.2f} | Średnia pewność: {slate.avg_probability*100:.1f}% | Średnie EV: {slate.avg_ev*100:+.1f}%")
    print(f"- Oczekiwany zysk dzienny: {slate.expected_daily_profit_pln:+.2f} PLN (Oczekiwany ROI: {slate.expected_daily_roi_pct:+.1f}%)")
    print(f"- Zysk przy wejściu wszystkich betów: +{slate.profit_if_all_win_pln:.2f} PLN")
    print(f"- Wynik jeśli 1 najniższy bet odpadnie: {slate.profit_if_one_loses_pln:+.2f} PLN")
    print(f"- Prawdopodobieństwo, że wszystkie bety wejdą: {slate.prob_all_win*100:.1f}%")
    print(f"- Prawdopodobieństwo, że min. (N-1) wejdzie: {slate.prob_at_least_80pct_win*100:.1f}%")

    if parlays:
        print("\n" + "-" * 95)
        print(f"DZIENNE DUBLE (OPTYMALNE PAROWANIE Z WYBRANYCH MECZÓW DNIA):")
        for idx, p in enumerate(parlays, 1):
            print(f"[*] Dubel #{idx} (Kurs łączny: {p['combined_odds']:.2f} | Szansa: {p['combined_prob']*100:.1f}% | EV: {p['ev']*100:+.1f}%)")
            print(f"    Noga 1: {p['leg1']['match']} -> {p['leg1']['market']} ({p['leg1']['odds']:.2f})")
            print(f"    Noga 2: {p['leg2']['match']} -> {p['leg2']['market']} ({p['leg2']['odds']:.2f})")
            print(f"    Stawka: {p['stake_pln']:.2f} PLN | Do wygrania: {p['potential_return_pln']:.2f} PLN (Zysk: +{p['potential_return_pln']-p['stake_pln']:.2f} PLN)")

    print("=" * 95 + "\n")
    return slate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day Return Optimizer Runner")
    parser.add_argument("--date", type=str, default="2026-10-18", help="Target date YYYY-MM-DD")
    parser.add_argument("--budget", type=float, default=50.0, help="Daily budget in PLN")
    parser.add_argument("--size", type=int, default=5, help="Number of bets in daily slate (e.g. 3-12)")
    parser.add_argument("--mode", type=str, default="safest_pewniaki", choices=["safest_pewniaki", "max_return", "balanced_sharpe"], help="Optimization mode")
    parser.add_argument("--sport", type=str, default="football", help="Sport filter")
    args = parser.parse_args()

    run_daily_optimizer(
        target_date_str=args.date,
        daily_budget=args.budget,
        slate_size=args.size,
        selection_mode=args.mode,
        sport_name=args.sport
    )
