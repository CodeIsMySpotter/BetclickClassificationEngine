"""Quantitative Benchmark & Backtesting Module for Capital Preservation and 2-Leg Parlay Strategies.

Simulates walk-forward execution across 1,520 real matches (2022-2026):
1. Baseline: Naive High-Risk Strategy (Odds > 3.0, czysta 1)
2. Previous Benchmark: Home-Only Unfiltered Strategy (dowolne kursy, czysta 1)
3. Capital Preservation Singles: (1X / DNB / 1, odds 1.15 - 1.85, Win Prob >= 70%)
4. NEW: Dixon-Coles Calibrated 2-Leg Parlay Portfolio (Optymalne duble z kursem 1.45 - 2.10)

Calculates institutional risk-adjusted metrics:
- Win Rate (%) & Capital Survival Rate (Win + Refund) (%)
- Total PnL (units) & ROI (%)
- Maximum Drawdown (MaxDD)
- Sharpe Ratio
"""

import datetime
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("StrategyBenchmark")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def simulate_dixon_coles_probabilities(elo_diff: float, xg_diff: float, rho: float = -0.06) -> Tuple[float, float, float]:
    """Estymacja Dixon-Coles z korekcją tau dla niskich wyników."""
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

    prob_h = float(np.sum(np.tril(grid, -1)))
    prob_x = float(np.sum(np.diag(grid)))
    prob_a = float(np.sum(np.triu(grid, 1)))
    tot = prob_h + prob_x + prob_a
    return prob_h / tot, prob_x / tot, prob_a / tot


def run_comprehensive_benchmark():
    matches_file = PROCESSED_DATA_DIR / "matches_merged.parquet"
    if not matches_file.exists():
        raise FileNotFoundError(f"Brak pliku: {matches_file}")

    df = pd.read_parquet(matches_file)
    logger.info("Wczytano %d historycznych spotkań do benchmarku.", len(df))

    df = df.dropna(subset=["B365H", "B365D", "B365A", "FTHG", "FTAG", "FTR"]).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    teams_elo: Dict[str, float] = {}
    teams_xg_f: Dict[str, List[float]] = {}
    teams_xg_a: Dict[str, List[float]] = {}

    def get_elo(team: str) -> float:
        return teams_elo.get(team, 1500.0)

    def get_rolling_xg(team: str, window: int = 5) -> Tuple[float, float]:
        f = teams_xg_f.get(team, [])
        a = teams_xg_a.get(team, [])
        rf = np.mean(f[-window:]) if f else 1.35
        ra = np.mean(a[-window:]) if a else 1.35
        return float(rf), float(ra)

    portfolio_high_risk = []
    portfolio_baseline = []
    portfolio_preservation_singles = []
    
    # Bufor kandydatów na duble w ramach danej kolejki/okna czasowego
    gameweek_buffer: List[dict] = []
    portfolio_parlays = []

    flat_stake = 100.0

    for idx, row in df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_goals = int(row["FTHG"])
        a_goals = int(row["FTAG"])
        ftr = row["FTR"]

        odd_h = float(row["B365H"])
        odd_x = float(row["B365D"])
        odd_a = float(row["B365A"])

        h_elo = get_elo(h_team)
        a_elo = get_elo(a_team)
        elo_diff = (h_elo + 65.0) - a_elo

        h_xg_f, h_xg_a = get_rolling_xg(h_team)
        a_xg_f, a_xg_a = get_rolling_xg(a_team)
        xg_diff = (h_xg_f - h_xg_a) - (a_xg_f - a_xg_a)

        # Predykcja Dixon-Coles
        prob_h, prob_x, prob_a = simulate_dixon_coles_probabilities(elo_diff, xg_diff)
        prob_1x = prob_h + prob_x
        prob_dnb = prob_h / (prob_h + prob_a) if (prob_h + prob_a) > 0 else 0.5

        odd_1x = round(1.0 / ((1.0 / odd_h) + (1.0 / odd_x)), 2)
        odd_dnb = round(odd_h * (1.0 - (1.0 / odd_x)), 2) if odd_x > 1.0 else odd_h

        ev_h = (prob_h * odd_h) - 1.0
        ev_1x = (prob_1x * odd_1x) - 1.0
        ev_dnb = (prob_dnb * odd_dnb) - 1.0

        # 1. High-Risk (kursy > 3.0)
        if odd_h >= 3.0 and ev_h >= 0.05:
            win = 1 if ftr == "H" else 0
            pnl = (flat_stake * (odd_h - 1.0)) if win else -flat_stake
            portfolio_high_risk.append({"date": row["match_date"], "win": win, "pnl": pnl, "stake": flat_stake, "odd": odd_h})

        # 2. Baseline Home-Only
        if ev_h >= 0.05:
            win = 1 if ftr == "H" else 0
            pnl = (flat_stake * (odd_h - 1.0)) if win else -flat_stake
            portfolio_baseline.append({"date": row["match_date"], "win": win, "pnl": pnl, "stake": flat_stake, "odd": odd_h})

        # 3. Capital Preservation Singles
        candidates = []
        if prob_1x >= 0.70 and (1.15 <= odd_1x <= 1.85) and ev_1x >= 0.03:
            candidates.append({"type": "1X", "odd": odd_1x, "prob": prob_1x, "ev": ev_1x})
        if prob_dnb >= 0.70 and (1.15 <= odd_dnb <= 1.85) and ev_dnb >= 0.03:
            candidates.append({"type": "DNB", "odd": odd_dnb, "prob": prob_dnb, "ev": ev_dnb})
        if prob_h >= 0.70 and (1.15 <= odd_h <= 1.85) and ev_h >= 0.03:
            candidates.append({"type": "1", "odd": odd_h, "prob": prob_h, "ev": ev_h})

        if candidates:
            best = sorted(candidates, key=lambda c: (c["prob"], c["ev"]), reverse=True)[0]
            m_type = best["type"]
            b_odd = best["odd"]

            # Ocena wyniku meczu
            if m_type == "1X":
                win = 1 if ftr in ["H", "D"] else 0
                pnl = (flat_stake * (b_odd - 1.0)) if win else -flat_stake
            elif m_type == "DNB":
                if ftr == "H":
                    win = 1
                    pnl = flat_stake * (b_odd - 1.0)
                elif ftr == "D":
                    win = 0.5  # Zwrot stawki
                    pnl = 0.0
                else:
                    win = 0
                    pnl = -flat_stake
            else:
                win = 1 if ftr == "H" else 0
                pnl = (flat_stake * (b_odd - 1.0)) if win else -flat_stake

            match_res = {
                "date": row["match_date"], "type": m_type, "win": win, "pnl": pnl, "stake": flat_stake, "odd": b_odd,
                "home_team": h_team, "away_team": a_team, "prob": best["prob"], "ev": best["ev"]
            }
            portfolio_preservation_singles.append(match_res)

            # Rejestracja w buforze dubli
            gameweek_buffer.append(match_res)

        # Sprawdzamy czy w buforze mamy parę do ułożenia dubla (2 niezależne zdarzenia)
        if len(gameweek_buffer) >= 2:
            c1 = gameweek_buffer.pop(0)
            c2 = gameweek_buffer.pop(0)

            parlay_odd = round(c1["odd"] * c2["odd"], 2)
            parlay_prob = round(c1["prob"] * c2["prob"], 4)
            parlay_ev = (parlay_prob * parlay_odd) - 1.0

            # Kryterium inwestycyjne dubla: kurs 1.45 - 2.10, EV > 5%
            if 1.45 <= parlay_odd <= 2.20 and parlay_ev >= 0.05:
                # Obliczenie wyniku dubla:
                # Wygrana = obie nogi wygrane (win == 1)
                # Zwrot/Częściowy zwrot jeśli jedna noga to DNB remis
                leg1_win = c1["win"]
                leg2_win = c2["win"]

                if leg1_win == 1 and leg2_win == 1:
                    pnl_parlay = flat_stake * (parlay_odd - 1.0)
                    win_parlay = 1
                elif (leg1_win == 1 and leg2_win == 0.5):
                    # Noga 2 zwrot, wygrana z nogi 1
                    pnl_parlay = flat_stake * (c1["odd"] - 1.0)
                    win_parlay = 1
                elif (leg1_win == 0.5 and leg2_win == 1):
                    pnl_parlay = flat_stake * (c2["odd"] - 1.0)
                    win_parlay = 1
                elif leg1_win == 0.5 and leg2_win == 0.5:
                    pnl_parlay = 0.0
                    win_parlay = 0.5
                else:
                    pnl_parlay = -flat_stake
                    win_parlay = 0

                portfolio_parlays.append({
                    "date": c2["date"], "win": win_parlay, "pnl": pnl_parlay, "stake": flat_stake, "odd": parlay_odd
                })

        # Aktualizacja stanu po meczu
        actual_score_h = 1.0 if h_goals > a_goals else (0.5 if h_goals == a_goals else 0.0)
        dr = (h_elo + 65.0) - a_elo
        we_h = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        teams_elo[h_team] = h_elo + 24.0 * (actual_score_h - we_h)
        teams_elo[a_team] = a_elo + 24.0 * ((1.0 - actual_score_h) - (1.0 - we_h))

        teams_xg_f.setdefault(h_team, []).append(float(row["home_xg"]))
        teams_xg_a.setdefault(h_team, []).append(float(row["away_xg"]))
        teams_xg_f.setdefault(a_team, []).append(float(row["away_xg"]))
        teams_xg_a.setdefault(a_team, []).append(float(row["home_xg"]))

    def calc_metrics(bets_list: list, name: str) -> dict:
        if not bets_list:
            return {"name": name, "count": 0, "win_rate": 0, "survival_rate": 0, "total_pnl": 0, "roi": 0, "max_dd": 0, "sharpe": 0, "avg_odd": 0}
        df_b = pd.DataFrame(bets_list)
        total_pnl = df_b["pnl"].sum()
        total_stake = df_b["stake"].sum()
        roi = (total_pnl / total_stake) * 100.0
        
        wins_count = len(df_b[df_b["win"] == 1])
        draw_refunds = len(df_b[df_b["win"] == 0.5])
        clean_win_rate = (wins_count / len(df_b)) * 100.0
        capital_survival_rate = ((wins_count + draw_refunds) / len(df_b)) * 100.0

        df_b["cum_pnl"] = df_b["pnl"].cumsum()
        df_b["peak"] = df_b["cum_pnl"].cummax()
        df_b["drawdown"] = df_b["cum_pnl"] - df_b["peak"]
        max_dd = df_b["drawdown"].min()

        daily_returns = df_b["pnl"] / df_b["stake"]
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(len(df_b))) if daily_returns.std() > 0 else 0.0

        return {
            "name": name,
            "count": len(df_b),
            "win_rate": clean_win_rate,
            "survival_rate": capital_survival_rate,
            "total_pnl": total_pnl,
            "roi": roi,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "avg_odd": df_b["odd"].mean()
        }

    m1 = calc_metrics(portfolio_high_risk, "1. Wysokie kursy (Odds > 3.0, czysta 1)")
    m2 = calc_metrics(portfolio_baseline, "2. Poprzedni Baseline (Dowolne kursy, czysta 1)")
    m3 = calc_metrics(portfolio_preservation_singles, "3. Capital Preservation (Single 1X / DNB / 1)")
    m4 = calc_metrics(portfolio_parlays, "4. OPTYMALNE DUBLE (2-Leg Parlay Dixon-Coles)")

    bench_df = pd.DataFrame([m1, m2, m3, m4])
    
    print("\n" + "=" * 95)
    print("INSTYTUCJONALNY BENCHMARK STRATEGII (Out-of-Time Walk-Forward 2022-2026)")
    print("=" * 95)
    for _, r in bench_df.iterrows():
        print(f"[*] STRATEGIA: {r['name']}")
        print(f"   Liczba zawartych zakladow: {r['count']} | Sredni kurs: {r['avg_odd']:.2f}")
        print(f"   Skutecznosc (Win Rate): {r['win_rate']:.1f}% | Brak straty kapitalu (Win + Zwrot): {r['survival_rate']:.1f}%")
        print(f"   Laczny Wynik Finansowy: {r['total_pnl']:+.2f} j. | Zwrot z Inwestycji (ROI): {r['roi']:+.2f}%")
        print(f"   Maksymalne obsuniecie (Max Drawdown): {r['max_dd']:.2f} j. | Sharpe Ratio: {r['sharpe']:.2f}\n")
    print("=" * 95 + "\n")
    return bench_df


if __name__ == "__main__":
    run_comprehensive_benchmark()
