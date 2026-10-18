"""Live Paper Trading & Forward Testing Audit System.

Enables out-of-sample forward testing on live matches:
1. Records weekly/daily slates for 3 distinct philosophies:
   - 🛡️ PEWNIAKI (safest_pewniaki)
   - 💰 ZWROT (max_return)
   - ⚖️ SHARPE (balanced_sharpe)
2. Persists slates with exact timestamps and odds into an audit ledger.
3. Automatically resolves finished matches against true outcomes (H / D / A, score).
4. Computes realized PnL, Win Rate, and ROI comparing all 3 approaches side-by-side.
5. Supports replaying completed 2026 matchdays to test year-to-date performance.
"""

from dataclasses import asdict
import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.domain import BetOpportunity, DailySlate, SportType
from src.optimizer.day_optimizer import DayReturnOptimizer
from src.sports.football.feed import FootballBetclicFeed
from src.sports.football.model import FootballInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("LivePaperTrader")

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_FILE = ARTIFACTS_DIR / "paper_trading_ledger.json"


def load_ledger() -> dict:
    if LEDGER_FILE.exists():
        try:
            with open(LEDGER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Błąd odczytu ledgera: %s. Tworzenie nowej bazy.", e)
    return {"campaigns": []}


def save_ledger(data: dict) -> None:
    with open(LEDGER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_next_week_campaign(
    target_date_str: str,
    budget_per_strategy: float = 50.0,
    slate_size: int = 5,
    campaign_name: Optional[str] = None
) -> dict:
    """Captures and locks in predictions for all 3 strategies (Pewniaki, Zwrot, Sharpe) for a given matchday."""
    import yaml
    cfg_file = BASE_DIR / "config" / "config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_date = datetime.date.fromisoformat(target_date_str)
    
    # 1. Pobieramy bieżące kursy z Betclica
    feed = FootballBetclicFeed(leagues_config=config.get("leagues", []))
    events = feed.fetch_live_events(force_refresh=False)
    
    engine = FootballInferenceEngine(config=config)
    opportunities = engine.evaluate_all(events)

    optimizer = DayReturnOptimizer(
        min_probability=config.get("strategy", {}).get("min_model_prob", 0.70),
        min_ev=config.get("strategy", {}).get("min_ev", 0.04),
        max_ev=config.get("strategy", {}).get("max_ev", 0.18),
        min_odds=config.get("strategy", {}).get("min_odds", 1.15),
        max_odds=config.get("strategy", {}).get("max_odds", 1.85),
    )

    strategies = {
        "safest_pewniaki": "PEWNIAKI (Safest)",
        "max_return": "ZWROT (Max EV)",
        "balanced_sharpe": "SHARPE (Balanced)"
    }

    c_id = f"campaign_{target_date_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    campaign_data = {
        "campaign_id": c_id,
        "name": campaign_name or f"Wyzwanie Dnia {target_date_str}",
        "created_at": datetime.datetime.now().isoformat(),
        "target_date": target_date_str,
        "budget_per_strategy": budget_per_strategy,
        "slate_size": slate_size,
        "status": "ACTIVE",
        "strategies": {}
    }

    for mode_key, mode_name in strategies.items():
        slate = optimizer.optimize_daily_slate(
            opportunities=opportunities,
            target_date=target_date,
            daily_budget=budget_per_strategy,
            slate_size=slate_size,
            selection_mode=mode_key
        )
        parlays = optimizer.generate_daily_parlays(slate, max_parlays=2)

        bets_records = []
        for b in slate.selections:
            bets_records.append({
                "bet_id": f"{b.event_id}_{b.market_name.replace(' ', '_')}",
                "event_id": b.event_id,
                "league": b.league_name,
                "match": b.match_display,
                "date": str(b.event_date),
                "market": b.market_name,
                "odds": b.odds,
                "model_prob": b.model_probability,
                "ev": b.expected_value,
                "stake_pln": b.recommended_stake_pln,
                "status": "PENDING",
                "result": None,
                "pnl": None
            })

        parlays_records = []
        for p in parlays:
            parlays_records.append({
                "combined_odds": p["combined_odds"],
                "combined_prob": p["combined_prob"],
                "ev": p["ev"],
                "stake_pln": p["stake_pln"],
                "leg1": p["leg1"],
                "leg2": p["leg2"],
                "status": "PENDING",
                "pnl": None
            })

        campaign_data["strategies"][mode_key] = {
            "name": mode_name,
            "total_stake_pln": slate.total_stake_pln,
            "expected_profit_pln": slate.expected_daily_profit_pln,
            "expected_roi_pct": slate.expected_daily_roi_pct,
            "bets": bets_records,
            "parlays": parlays_records,
            "summary": {
                "total_bets": len(bets_records),
                "won": 0, "lost": 0, "refunded": 0, "pending": len(bets_records),
                "realized_pnl": 0.0, "realized_roi": 0.0, "current_balance": budget_per_strategy
            }
        }

    ledger = load_ledger()
    ledger["campaigns"].append(campaign_data)
    save_ledger(ledger)
    logger.info("Zapisano nową kampanię Live Paper Trading: %s", c_id)
    return campaign_data


def fetch_match_result_from_history(home_team: str, away_team: str, match_date: str) -> Optional[dict]:
    """Looks up match score in local historical files if already completed."""
    hist_file = BASE_DIR / "src" / "dashboard" / "matches_history.csv"
    if hist_file.exists():
        df_h = pd.read_csv(hist_file)
        match = df_h[
            (df_h["home_team"].str.lower() == home_team.lower()) &
            (df_h["away_team"].str.lower() == away_team.lower()) &
            (df_h["date"].astype(str) == match_date)
        ]
        if not match.empty:
            row = match.iloc[0]
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            ftr = "H" if hg > ag else ("D" if hg == ag else "A")
            return {"fthg": hg, "ftag": ag, "ftr": ftr}

    parquet_file = BASE_DIR / "data" / "processed" / "matches_merged.parquet"
    if parquet_file.exists():
        df_p = pd.read_parquet(parquet_file)
        df_p["m_date_str"] = pd.to_datetime(df_p["match_date"]).dt.strftime("%Y-%m-%d")
        match = df_p[
            (df_p["home_team"].str.lower() == home_team.lower()) &
            (df_p["away_team"].str.lower() == away_team.lower()) &
            (df_p["m_date_str"] == match_date)
        ]
        if not match.empty:
            row = match.iloc[0]
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            return {"fthg": hg, "ftag": ag, "ftr": str(row["FTR"])}

    return None


def resolve_bet_outcome(market: str, odds: float, stake_pln: float, ftr: str) -> Tuple[str, float]:
    """Evaluates whether a bet won, lost, or pushed, returning (status, realized_pnl)."""
    if "1X" in market:
        if ftr in ["H", "D"]:
            return "WON", round(stake_pln * (odds - 1.0), 2)
        else:
            return "LOST", -round(stake_pln, 2)
    elif "DNB" in market:
        if ftr == "H":
            return "WON", round(stake_pln * (odds - 1.0), 2)
        elif ftr == "D":
            return "REFUNDED", 0.0
        else:
            return "LOST", -round(stake_pln, 2)
    else:  # Czysta wygrana 1
        if ftr == "H":
            return "WON", round(stake_pln * (odds - 1.0), 2)
        else:
            return "LOST", -round(stake_pln, 2)


def evaluate_active_campaigns(manual_results: Optional[Dict[str, dict]] = None) -> dict:
    """Checks and resolves all pending bets across active campaigns using historical or provided results."""
    ledger = load_ledger()
    updated = False

    for campaign in ledger.get("campaigns", []):
        all_resolved = True
        budget = campaign.get("budget_per_strategy", 50.0)

        for mode_key, strat in campaign.get("strategies", {}).items():
            won_cnt, lost_cnt, ref_cnt, pend_cnt = 0, 0, 0, 0
            realized_pnl = 0.0

            for b in strat.get("bets", []):
                if b["status"] == "PENDING":
                    # Check manual results first (by match display e.g. "Roma vs Genoa")
                    res = None
                    m_key = b["match"]
                    if manual_results and m_key in manual_results:
                        res = manual_results[m_key]
                    else:
                        parts = b["match"].split(" vs ")
                        if len(parts) == 2:
                            res = fetch_match_result_from_history(parts[0].strip(), parts[1].strip(), b["date"])

                    if res:
                        status, pnl = resolve_bet_outcome(b["market"], b["odds"], b["stake_pln"], res["ftr"])
                        b["status"] = status
                        b["result"] = res
                        b["pnl"] = pnl
                        updated = True

                # Zliczanie
                if b["status"] == "WON":
                    won_cnt += 1
                    realized_pnl += b["pnl"]
                elif b["status"] == "LOST":
                    lost_cnt += 1
                    realized_pnl += b["pnl"]
                elif b["status"] == "REFUNDED":
                    ref_cnt += 1
                else:
                    pend_cnt += 1
                    all_resolved = False

            total_stake = strat.get("total_stake_pln", budget)
            roi = (realized_pnl / total_stake * 100.0) if total_stake > 0 else 0.0

            strat["summary"] = {
                "total_bets": len(strat.get("bets", [])),
                "won": won_cnt,
                "lost": lost_cnt,
                "refunded": ref_cnt,
                "pending": pend_cnt,
                "realized_pnl": round(realized_pnl, 2),
                "realized_roi": round(roi, 2),
                "current_balance": round(budget + realized_pnl, 2)
            }

        if all_resolved:
            campaign["status"] = "COMPLETED"

    if updated:
        save_ledger(ledger)

    return ledger


def print_paper_trading_scorecard(ledger: Optional[dict] = None):
    """Prints a beautiful, side-by-side terminal scorecard comparing the 3 approaches."""
    if ledger is None:
        ledger = load_ledger()

    campaigns = ledger.get("campaigns", [])
    if not campaigns:
        print("\n[!] Brak zarejestrowanych kampanii Paper Trading. Uruchom 'record' aby zapisać prognozę.")
        return

    latest = campaigns[-1]
    print("\n" + "=" * 100)
    print(f"RAPORT TESTU NA ŻYWYM ORGANIZMIE: {latest['name'].upper()} (Data: {latest['target_date']})")
    print(f"Budżet na strategię: {latest['budget_per_strategy']:.2f} PLN | Rozmiar portfela: {latest['slate_size']} betów")
    print("=" * 100)

    for mode_key, strat in latest["strategies"].items():
        sm = strat["summary"]
        print(f"\n[*] STRATEGIA: {strat['name']}")
        print(f"    Stan konta: {sm['current_balance']:.2f} PLN (Start: {latest['budget_per_strategy']:.2f} PLN)")
        print(f"    Czysty zysk / strata: {sm['realized_pnl']:+.2f} PLN | ROI: {sm['realized_roi']:+.1f}%")
        print(f"    Rozliczenie kuponów: {sm['won']} wygranych, {sm['refunded']} zwrotów, {sm['lost']} przegranych, {sm['pending']} oczekujących")
        
        # Wypisujemy kupony
        for b in strat["bets"]:
            status_symbol = "[OK]" if b["status"] == "WON" else ("[X]" if b["status"] == "LOST" else ("[=]" if b["status"] == "REFUNDED" else "[?]"))
            res_str = f"({b['result']['fthg']}:{b['result']['ftag']})" if b.get("result") else ""
            pnl_str = f"{b['pnl']:+.2f} PLN" if b.get("pnl") is not None else "oczekuje"
            print(f"    {status_symbol} {b['match']} {res_str} -> {b['market']} @ {b['odds']:.2f} [Stawka: {b['stake_pln']:.2f} PLN | Wynik: {pnl_str}]")

    print("\n" + "=" * 100 + "\n")


def run_ytd_historical_replay(num_matchdays: int = 15, budget_per_day: float = 50.0, slate_size: int = 5):
    """Replays the completed matchdays of 2026/current season to evaluate Pewniaki vs Zwrot vs Sharpe."""
    parquet_file = BASE_DIR / "data" / "processed" / "matches_merged.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Brak pliku danych: {parquet_file}")

    df = pd.read_parquet(parquet_file).dropna(subset=["B365H", "B365D", "B365A", "FTHG", "FTAG", "FTR"]).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    # Bierzemy ostatnie kolejki (2025/2026)
    matchday_dates = sorted(df["match_date"].dt.date.unique())[-num_matchdays:]
    
    print("\n" + "=" * 105)
    print(f"SYMULACJA YTD: TEST 3 PODEJŚĆ (PEWNIAKI vs ZWROT vs SHARPE) NA {len(matchday_dates)} OSTATNICH DNIACH MECZOWYCH")
    print(f"Okres: {matchday_dates[0]} do {matchday_dates[-1]} | Budżet na 1 dzień: {budget_per_day:.2f} PLN | Rozmiar: {slate_size} betów/dzień")
    print("=" * 105)

    import yaml
    with open(BASE_DIR / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    optimizer = DayReturnOptimizer(
        min_probability=float(config.get("strategy", {}).get("min_model_prob", 0.78)),
        min_ev=0.04,
        max_ev=0.18,
        min_odds=1.15,
        max_odds=1.60,
        max_odds_1x=float(config.get("strategy", {}).get("max_odds_1x", 1.25))
    )

    results_by_mode = {
        "safest_pewniaki": {"name": "PEWNIAKI (Safest)", "pnls": [], "wins": 0, "losses": 0, "refunds": 0, "stakes": 0.0, "bets": [], "cum_pnl": 0.0},
        "max_return": {"name": "ZWROT (Max EV)", "pnls": [], "wins": 0, "losses": 0, "refunds": 0, "stakes": 0.0, "bets": [], "cum_pnl": 0.0},
        "balanced_sharpe": {"name": "SHARPE (Balanced)", "pnls": [], "wins": 0, "losses": 0, "refunds": 0, "stakes": 0.0, "bets": [], "cum_pnl": 0.0}
    }

    timeline_points = []

    # Uruchamiamy walk-forward
    engine = FootballInferenceEngine(config=config)

    for m_date in matchday_dates:
        df_day = df[df["match_date"].dt.date == m_date]
        if len(df_day) < 3:
            continue

        # Tworzymy obiekty MatchEvent dla tego dnia
        events = []
        scores_by_event = {}
        for idx, row in df_day.iterrows():
            oh, ox, oa = float(row["B365H"]), float(row["B365D"]), float(row["B365A"])
            o_1x = round(1.0 / ((1.0/oh) + (1.0/ox)), 2)
            o_dnb = round(oh * (1.0 - (1.0/ox)), 2) if ox > 1.0 else oh
            odds = {"1": oh, "X": ox, "2": oa, "1X": o_1x, "DNB": o_dnb}
            
            ev_id = f"hist_{idx}"
            from src.core.domain import MatchEvent, SportType
            ev = MatchEvent(
                event_id=ev_id,
                sport=SportType.FOOTBALL,
                league_name="Liga",
                home_competitor=row["home_team"],
                away_competitor=row["away_team"],
                event_date=m_date,
                odds=odds,
                metadata={"ftr": row["FTR"]}
            )
            events.append(ev)
            scores_by_event[ev_id] = {
                "fthg": int(row["FTHG"]),
                "ftag": int(row["FTAG"]),
                "ftr": str(row["FTR"])
            }

        opps = engine.evaluate_all(events)
        if not opps:
            continue

        day_record = {"date": str(m_date), "matches_count": len(events)}

        # Ewaluacja dla 3 trybów
        for mode in ["safest_pewniaki", "max_return", "balanced_sharpe"]:
            slate = optimizer.optimize_daily_slate(
                opportunities=opps, target_date=m_date, daily_budget=budget_per_day, slate_size=slate_size, selection_mode=mode
            )
            day_pnl = 0.0
            for b in slate.selections:
                ev_match = next((e for e in events if e.event_id == b.event_id), None)
                if ev_match:
                    score_info = scores_by_event.get(b.event_id, {"fthg": 0, "ftag": 0, "ftr": "H"})
                    status, pnl = resolve_bet_outcome(b.market_name, b.odds, b.recommended_stake_pln, score_info["ftr"])
                    day_pnl += pnl
                    results_by_mode[mode]["stakes"] += b.recommended_stake_pln
                    if status == "WON": results_by_mode[mode]["wins"] += 1
                    elif status == "LOST": results_by_mode[mode]["losses"] += 1
                    else: results_by_mode[mode]["refunds"] += 1

                    results_by_mode[mode]["bets"].append({
                        "date": str(m_date),
                        "match": b.match_display,
                        "league": b.league_name,
                        "market": b.market_name,
                        "odds": b.odds,
                        "fair_odds": b.fair_odds,
                        "model_prob": round(b.model_probability * 100, 1),
                        "ev": round(b.expected_value * 100, 1),
                        "stake_pln": round(b.recommended_stake_pln, 2),
                        "status": status,
                        "score": f"{score_info['fthg']}:{score_info['ftag']}",
                        "pnl": round(pnl, 2),
                        "strategy": mode,
                        "strategy_name": results_by_mode[mode]["name"],
                        "rationale": b.rationale
                    })

            results_by_mode[mode]["pnls"].append(day_pnl)
            results_by_mode[mode]["cum_pnl"] = round(results_by_mode[mode]["cum_pnl"] + day_pnl, 2)
            day_record[f"{mode}_day_pnl"] = round(day_pnl, 2)
            day_record[f"{mode}_cum_pnl"] = results_by_mode[mode]["cum_pnl"]
            day_record[f"{mode}_balance"] = round(budget_per_day + results_by_mode[mode]["cum_pnl"], 2)

        timeline_points.append(day_record)

    # Zapis do artifacts/replay_2026_timeline.json
    replay_artifact = {
        "generated_at": datetime.datetime.now().isoformat(),
        "num_matchdays": len(timeline_points),
        "budget_per_day": budget_per_day,
        "timeline": timeline_points,
        "strategies": {
            mode: {
                "name": data["name"],
                "total_bets": data["wins"] + data["losses"] + data["refunds"],
                "won": data["wins"],
                "lost": data["losses"],
                "refunded": data["refunds"],
                "win_rate_pct": round((data["wins"] / (data["wins"] + data["losses"] + data["refunds"]) * 100.0), 1) if (data["wins"] + data["losses"] + data["refunds"]) > 0 else 0.0,
                "total_stake": round(data["stakes"], 2),
                "net_pnl": round(sum(data["pnls"]), 2),
                "roi_pct": round((sum(data["pnls"]) / data["stakes"] * 100.0), 2) if data["stakes"] > 0 else 0.0,
                "bets": data["bets"]
            }
            for mode, data in results_by_mode.items()
        }
    }

    replay_file = ARTIFACTS_DIR / "replay_2026_timeline.json"
    with open(replay_file, "w", encoding="utf-8") as f:
        json.dump(replay_artifact, f, indent=2, ensure_ascii=False)

    # Podsumowanie porównawcze
    print(f"{'STRATEGIA':<28} | {'ZAKŁADY':<8} | {'WIN RATE':<10} | {'OBRÓT (PLN)':<12} | {'CZYSTY PnL':<14} | {'ROI (%)':<8}")
    print("-" * 105)
    for mode, data in replay_artifact["strategies"].items():
        print(f"{data['name']:<28} | {data['total_bets']:<8} | {data['win_rate_pct']:<9.1f}% | {data['total_stake']:<12.2f} | {data['net_pnl']:+12.2f} PLN | {data['roi_pct']:+6.2f}%")

    print(f"\n[OK] Zapisano szczegolowe dane osi czasu do: {replay_file}")
    print("=" * 105 + "\n")
    return replay_artifact


def fetch_recent_engine_picks(
    num_bets: int = 20,
    strategy_mode: str = "safest_pewniaki",
    daily_budget: float = 50.0,
    slate_size: int = 5,
    min_prob: float = 0.78,
    min_ev: float = 0.04,
    max_ev: float = 0.18,
    min_odds: float = 1.15,
    max_odds: float = 1.60,
    max_odds_1x: float = 1.25,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """Gathers the exact last N engine bets by scanning completed matches backwards in time."""
    prob_tag = int(round(min_prob * 100))
    cache_file = ARTIFACTS_DIR / f"recent_engine_picks_{strategy_mode}_{num_bets}_p{prob_tag}.json"
    if cache_file.exists() and not force_refresh:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    parquet_file = BASE_DIR / "data" / "processed" / "matches_merged.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Brak pliku danych: {parquet_file}")

    df = pd.read_parquet(parquet_file).dropna(subset=["B365H", "B365D", "B365A", "FTHG", "FTAG", "FTR"]).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    # Sort dates descending (newest first)
    matchday_dates = sorted(df["match_date"].dt.date.unique(), reverse=True)

    import yaml
    with open(BASE_DIR / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_copy = dict(config)
    cfg_copy["strategy"] = dict(config.get("strategy", {}))
    cfg_copy["strategy"]["min_model_prob"] = min_prob
    cfg_copy["strategy"]["min_ev"] = min_ev
    cfg_copy["strategy"]["max_ev"] = max_ev
    cfg_copy["strategy"]["min_odds"] = min_odds
    cfg_copy["strategy"]["max_odds"] = max_odds
    cfg_copy["strategy"]["max_odds_1x"] = max_odds_1x

    optimizer = DayReturnOptimizer(
        min_probability=min_prob,
        min_ev=min_ev,
        max_ev=max_ev,
        min_odds=min_odds,
        max_odds=max_odds,
        max_odds_1x=max_odds_1x
    )
    engine = FootballInferenceEngine(config=cfg_copy)

    collected_bets = []
    scanned_days = 0

    for m_date in matchday_dates:
        if len(collected_bets) >= num_bets:
            break
        df_day = df[df["match_date"].dt.date == m_date]
        if df_day.empty:
            continue
        scanned_days += 1

        events = []
        scores_by_event = {}
        for idx, row in df_day.iterrows():
            oh, ox, oa = float(row["B365H"]), float(row["B365D"]), float(row["B365A"])
            o_1x = round(1.0 / ((1.0/oh) + (1.0/ox)), 2)
            o_dnb = round(oh * (1.0 - (1.0/ox)), 2) if ox > 1.0 else oh
            odds = {"1": oh, "X": ox, "2": oa, "1X": o_1x, "DNB": o_dnb}
            
            ev_id = f"hist_{idx}"
            from src.core.domain import MatchEvent, SportType
            ev = MatchEvent(
                event_id=ev_id,
                sport=SportType.FOOTBALL,
                league_name="La Liga",
                home_competitor=row["home_team"],
                away_competitor=row["away_team"],
                event_date=m_date,
                odds=odds,
                metadata={"ftr": row["FTR"]}
            )
            events.append(ev)
            scores_by_event[ev_id] = {
                "fthg": int(row["FTHG"]),
                "ftag": int(row["FTAG"]),
                "ftr": str(row["FTR"])
            }

        opps = engine.evaluate_all(events)
        if not opps:
            continue

        effective_slate_size = 20 if num_bets > 50 else slate_size
        slate = optimizer.optimize_daily_slate(
            opportunities=opps,
            target_date=m_date,
            daily_budget=daily_budget,
            slate_size=effective_slate_size,
            selection_mode=strategy_mode
        )

        for b in slate.selections:
            if len(collected_bets) >= num_bets:
                break
            score_info = scores_by_event.get(b.event_id, {"fthg": 0, "ftag": 0, "ftr": "H"})
            status, pnl = resolve_bet_outcome(b.market_name, b.odds, b.recommended_stake_pln, score_info["ftr"])
            collected_bets.append({
                "date": str(m_date),
                "match": b.match_display,
                "league": b.league_name,
                "market": b.market_name,
                "odds": b.odds,
                "fair_odds": b.fair_odds,
                "model_prob": round(b.model_probability * 100, 1),
                "ev": round(b.expected_value * 100, 1),
                "stake_pln": round(b.recommended_stake_pln, 2),
                "status": status,
                "score": f"{score_info['fthg']}:{score_info['ftag']}",
                "pnl": round(pnl, 2),
                "strategy": strategy_mode,
                "strategy_name": "PEWNIAKI (Safest)" if strategy_mode == "safest_pewniaki" else ("ZWROT (Max EV)" if strategy_mode == "max_return" else "SHARPE (Balanced)"),
                "rationale": b.rationale
            })

    # Sort collected bets chronologically (oldest to newest)
    collected_bets_chrono = sorted(collected_bets, key=lambda x: x["date"])
    
    # Calculate running cumulative balance
    running_balance = daily_budget
    timeline = []
    for b in collected_bets_chrono:
        running_balance = round(running_balance + b["pnl"], 2)
        b["running_balance"] = running_balance
        timeline.append({
            "date": b["date"],
            "match": b["match"],
            "pnl": b["pnl"],
            "running_balance": running_balance
        })

    won = sum(1 for b in collected_bets if b["status"] == "WON")
    lost = sum(1 for b in collected_bets if b["status"] == "LOST")
    refunded = sum(1 for b in collected_bets if b["status"] == "REFUNDED")
    tot_stake = round(sum(b["stake_pln"] for b in collected_bets), 2)
    net_pnl = round(sum(b["pnl"] for b in collected_bets), 2)
    win_rate = round(won / (won + lost) * 100.0, 1) if (won + lost) > 0 else 0.0
    roi = round(net_pnl / tot_stake * 100.0, 2) if tot_stake > 0 else 0.0

    result_payload = {
        "generated_at": datetime.datetime.now().isoformat(),
        "requested_bets": num_bets,
        "actual_bets": len(collected_bets),
        "scanned_matchdays": scanned_days,
        "strategy": strategy_mode,
        "summary": {
            "total_bets": len(collected_bets),
            "won": won,
            "lost": lost,
            "refunded": refunded,
            "win_rate_pct": win_rate,
            "total_stake_pln": tot_stake,
            "net_pnl_pln": net_pnl,
            "roi_pct": roi,
            "start_balance": daily_budget,
            "end_balance": running_balance
        },
        "bets": collected_bets,
        "bets_chrono": collected_bets_chrono,
        "timeline": timeline
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    return result_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Paper Trader & Audit System")
    parser.add_argument("--action", type=str, default="scorecard", choices=["record", "evaluate", "scorecard", "replay"])
    parser.add_argument("--date", type=str, default="2026-10-18", help="Target date YYYY-MM-DD for recording")
    parser.add_argument("--budget", type=float, default=50.0, help="Daily budget per strategy")
    parser.add_argument("--size", type=int, default=5, help="Number of bets per day")
    args = parser.parse_args()

    if args.action == "record":
        record_next_week_campaign(target_date_str=args.date, budget_per_strategy=args.budget, slate_size=args.size)
        print_paper_trading_scorecard()
    elif args.action == "evaluate":
        evaluate_active_campaigns()
        print_paper_trading_scorecard()
    elif args.action == "replay":
        run_ytd_historical_replay(num_matchdays=20, budget_per_day=args.budget, slate_size=args.size)
    else:
        print_paper_trading_scorecard()
