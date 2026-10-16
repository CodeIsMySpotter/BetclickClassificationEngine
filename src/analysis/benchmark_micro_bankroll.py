"""Micro-Bankroll Quantitative Benchmark (50 PLN starting capital).

Simulates bankroll growth and drawdown dynamics with a micro budget (50 PLN)
using realistic betting stakes (2.00 PLN flat stake per bet) across:
1. High-Risk Underdogs (Odds > 3.0)
2. Home-Only Baseline
3. Capital Preservation Singles (1X / DNB)
4. Optimal 2-Leg Parlays (Dixon-Coles Duble)
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def simulate_dixon_coles_probabilities(elo_diff: float, xg_diff: float, rho: float = -0.06) -> Tuple[float, float, float]:
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


def run_micro_bankroll_benchmark(initial_bankroll: float = 50.0, stake_pln: float = 2.0):
    matches_file = PROCESSED_DATA_DIR / "matches_merged.parquet"
    if not matches_file.exists():
        raise FileNotFoundError(f"Brak pliku: {matches_file}")

    df = pd.read_parquet(matches_file).dropna(subset=["B365H", "B365D", "B365A", "FTHG", "FTAG", "FTR"]).copy()
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

    # Rejestry portfeli
    p_high = []
    p_baseline = []
    p_singles = []
    p_parlays = []
    parlay_buffer = []

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

        prob_h, prob_x, prob_a = simulate_dixon_coles_probabilities(elo_diff, xg_diff)
        prob_1x = prob_h + prob_x
        prob_dnb = prob_h / (prob_h + prob_a) if (prob_h + prob_a) > 0 else 0.5

        odd_1x = round(1.0 / ((1.0 / odd_h) + (1.0 / odd_x)), 2)
        odd_dnb = round(odd_h * (1.0 - (1.0 / odd_x)), 2) if odd_x > 1.0 else odd_h

        ev_h = (prob_h * odd_h) - 1.0
        ev_1x = (prob_1x * odd_1x) - 1.0
        ev_dnb = (prob_dnb * odd_dnb) - 1.0

        # 1. High Risk (>3.0)
        if odd_h >= 3.0 and ev_h >= 0.05:
            win = 1 if ftr == "H" else 0
            pnl = (stake_pln * (odd_h - 1.0)) if win else -stake_pln
            p_high.append({"date": row["match_date"], "win": win, "pnl": pnl, "stake": stake_pln, "odd": odd_h})

        # 2. Baseline
        if ev_h >= 0.05:
            win = 1 if ftr == "H" else 0
            pnl = (stake_pln * (odd_h - 1.0)) if win else -stake_pln
            p_baseline.append({"date": row["match_date"], "win": win, "pnl": pnl, "stake": stake_pln, "odd": odd_h})

        # 3. Capital Preservation Singles z filtrem pułapek kursowych (0.04 <= EV <= 0.18)
        candidates = []
        if prob_1x >= 0.70 and (1.15 <= odd_1x <= 1.85) and (0.04 <= ev_1x <= 0.18):
            candidates.append({"type": "1X", "odd": odd_1x, "prob": prob_1x, "ev": ev_1x})
        if prob_dnb >= 0.70 and (1.15 <= odd_dnb <= 1.85) and (0.04 <= ev_dnb <= 0.18):
            candidates.append({"type": "DNB", "odd": odd_dnb, "prob": prob_dnb, "ev": ev_dnb})
        if prob_h >= 0.70 and (1.15 <= odd_h <= 1.85) and (0.04 <= ev_h <= 0.18):
            candidates.append({"type": "1", "odd": odd_h, "prob": prob_h, "ev": ev_h})

        if candidates:
            best = sorted(candidates, key=lambda c: (c["prob"], c["ev"]), reverse=True)[0]
            m_type = best["type"]
            b_odd = best["odd"]

            if m_type == "1X":
                win = 1 if ftr in ["H", "D"] else 0
                pnl = (stake_pln * (b_odd - 1.0)) if win else -stake_pln
            elif m_type == "DNB":
                if ftr == "H":
                    win = 1
                    pnl = stake_pln * (b_odd - 1.0)
                elif ftr == "D":
                    win = 0.5
                    pnl = 0.0
                else:
                    win = 0
                    pnl = -stake_pln
            else:
                win = 1 if ftr == "H" else 0
                pnl = (stake_pln * (b_odd - 1.0)) if win else -stake_pln

            single_match = {
                "date": row["match_date"], "win": win, "pnl": pnl, "stake": stake_pln, "odd": b_odd,
                "prob": best["prob"], "ev": best["ev"]
            }
            p_singles.append(single_match)
            parlay_buffer.append(single_match)

        # 4. Duble 2-Leg Parlay
        if len(parlay_buffer) >= 2:
            c1 = parlay_buffer.pop(0)
            c2 = parlay_buffer.pop(0)
            comb_odd = round(c1["odd"] * c2["odd"], 2)
            comb_prob = round(c1["prob"] * c2["prob"], 4)
            comb_ev = (comb_prob * comb_odd) - 1.0
            comb_date = max(c1["date"], c2["date"])

            if 1.45 <= comb_odd <= 2.20 and comb_ev >= 0.05:
                w1 = c1["win"]
                w2 = c2["win"]
                if w1 == 1 and w2 == 1:
                    pnl_parlay = stake_pln * (comb_odd - 1.0)
                    win_parlay = 1
                elif (w1 == 1 and w2 == 0.5):
                    pnl_parlay = stake_pln * (c1["odd"] - 1.0)
                    win_parlay = 1
                elif (w1 == 0.5 and w2 == 1):
                    pnl_parlay = stake_pln * (c2["odd"] - 1.0)
                    win_parlay = 1
                elif w1 == 0.5 and w2 == 0.5:
                    pnl_parlay = 0.0
                    win_parlay = 0.5
                else:
                    pnl_parlay = -stake_pln
                    win_parlay = 0

                p_parlays.append({
                    "date": comb_date, "win": win_parlay, "pnl": pnl_parlay, "stake": stake_pln, "odd": comb_odd
                })

        # Aktualizacja stanu
        actual_score_h = 1.0 if h_goals > a_goals else (0.5 if h_goals == a_goals else 0.0)
        dr = (h_elo + 65.0) - a_elo
        we_h = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        teams_elo[h_team] = h_elo + 24.0 * (actual_score_h - we_h)
        teams_elo[a_team] = a_elo + 24.0 * ((1.0 - actual_score_h) - (1.0 - we_h))
        teams_xg_f.setdefault(h_team, []).append(float(row["home_xg"]))
        teams_xg_a.setdefault(h_team, []).append(float(row["away_xg"]))
        teams_xg_f.setdefault(a_team, []).append(float(row["away_xg"]))
        teams_xg_a.setdefault(a_team, []).append(float(row["home_xg"]))

    # Statystyki horyzontu czasowego całego zbioru
    global_start = pd.to_datetime(df["match_date"].min()).date()
    global_end = pd.to_datetime(df["match_date"].max()).date()
    global_days = (global_end - global_start).days
    global_months = round(global_days / 30.4375, 1)
    global_years = round(global_days / 365.25, 2)

    def evaluate(records: list, label: str):
        if not records:
            return
        df_r = pd.DataFrame(records)
        df_r["cum_pnl"] = df_r["pnl"].cumsum()
        df_r["balance"] = initial_bankroll + df_r["cum_pnl"]
        
        # Daty i horyzont czasowy
        s_date = pd.to_datetime(df_r["date"].min()).date()
        e_date = pd.to_datetime(df_r["date"].max()).date()
        strat_days = (e_date - s_date).days
        strat_months = max(1.0, round(strat_days / 30.4375, 1))
        strat_years = max(0.1, round(strat_days / 365.25, 2))

        # Tempo gry
        avg_per_month = round(len(df_r) / strat_months, 1)
        avg_per_week = round(len(df_r) / max(1.0, (strat_days / 7.0)), 1)

        # Max Drawdown w PLN
        df_r["peak"] = df_r["cum_pnl"].cummax()
        df_r["drawdown"] = df_r["cum_pnl"] - df_r["peak"]
        max_dd_pln = df_r["drawdown"].min()
        min_balance_reached = df_r["balance"].min()

        total_profit = df_r["pnl"].sum()
        total_turnover = df_r["stake"].sum()
        roi_turnover = (total_profit / total_turnover) * 100.0
        roi_capital = (total_profit / initial_bankroll) * 100.0
        win_rate = (len(df_r[df_r["win"] == 1]) / len(df_r)) * 100.0
        survival_rate = (len(df_r[df_r["win"].isin([1, 0.5])]) / len(df_r)) * 100.0

        busted = min_balance_reached <= 0

        print(f"[*] {label.upper()}")
        print(f"    Czas trwania sesji: {s_date} do {e_date} ({strat_days} dni | {strat_months} mies. | {strat_years} roku)")
        print(f"    Zawarte zakłady: {len(df_r)} kuponów po {stake_pln:.2f} PLN (Średnio {avg_per_month} kuponów/mies. ~ {avg_per_week} kuponu/tydz.)")
        print(f"    Obrót łącznie: {total_turnover:.2f} PLN | Skuteczność (Win Rate): {win_rate:.1f}%")
        print(f"    Ochrona przed stratą (Win + Zwrot): {survival_rate:.1f}%")
        print(f"    Stan konta końcowy: {initial_bankroll + total_profit:.2f} PLN (Start: {initial_bankroll:.2f} PLN)")
        print(f"    Czysty zysk: {total_profit:+.2f} PLN | Stopa zwrotu z kapitału: {roi_capital:+.1f}%")
        print(f"    Najniższy stan konta w trakcie gry: {min_balance_reached:.2f} PLN (Maksymalne obsunięcie: {max_dd_pln:+.2f} PLN)")
        
        if busted:
            bust_rows = df_r[df_r["balance"] <= 0]
            if not bust_rows.empty:
                first_bust = bust_rows.iloc[0]
                b_date = pd.to_datetime(first_bust["date"]).date()
                b_days = (b_date - s_date).days
                b_idx = int(bust_rows.index[0]) + 1
                print(f"    Status kapitału: [X] BANKRUT! Portfel wyzerowany po {b_days} dniach ({b_date}) przy kuponie #{b_idx}\n")
            else:
                print("    Status kapitału: [X] BANKRUT! Portfel wyzerowany w trakcie sesji\n")
        else:
            monthly_pnl = total_profit / strat_months
            cagr = ((1.0 + (total_profit / initial_bankroll)) ** (1.0 / strat_years) - 1.0) * 100.0
            print(f"    Średni zysk miesięczny: {monthly_pnl:+.2f} PLN/miesiąc | Roczna stopa CAGR: {cagr:+.1f}%")
            print(f"    Status kapitału: [V] PRZETRWAŁ! Pełna ochrona kapitału przez {strat_years} roku bez bankructwa\n")

    def evaluate_sprints(records: list, label: str, n_bets: int = 20, stake: float = 2.50):
        if len(records) < n_bets:
            return
        sprints = []
        for i in range(0, len(records) - n_bets + 1, n_bets):
            window = records[i:i + n_bets]
            pnl = sum((w["pnl"] / stake_pln) * stake for w in window)
            wins = sum(1 for w in window if w["win"] == 1)
            refunds = sum(1 for w in window if w["win"] == 0.5)
            wr = (wins / n_bets) * 100.0
            surv = ((wins + refunds) / n_bets) * 100.0
            
            s_date = pd.to_datetime(window[0]["date"]).date()
            e_date = pd.to_datetime(window[-1]["date"]).date()
            days = (e_date - s_date).days

            cum = np.cumsum([(w["pnl"] / stake_pln) * stake for w in window])
            peak = np.maximum.accumulate(cum)
            dd = np.min(cum - peak)
            min_bal = initial_bankroll + np.min(cum)
            busted = min_bal <= 0

            sprints.append({
                "pnl": pnl, "wr": wr, "surv": surv, "days": days,
                "min_bal": min_bal, "dd": dd, "busted": busted
            })

        df_s = pd.DataFrame(sprints)
        n_total = len(df_s)
        n_prof = len(df_s[df_s["pnl"] > 0])
        n_bust = len(df_s[df_s["busted"] == True])
        
        avg_p = df_s["pnl"].mean()
        max_p = df_s["pnl"].max()
        min_p = df_s["pnl"].min()
        avg_wr = df_s["wr"].mean()
        avg_days = df_s["days"].mean()
        avg_dd = df_s["dd"].mean()

        print(f"[*] {label.upper()} (Stawka: {stake:.2f} PLN / kupon)")
        print(f"    Liczba rozegranych sesji 20-betowych w historii: {n_total} prób")
        print(f"    Średni czas realizacji 20 betów: ~{avg_days:.0f} dni (~1 miesiąc ligowy)")
        print(f"    SZANSA ZAKOŃCZENIA MIESIĄCA NA PLUSIE: {n_prof / n_total * 100:.1f}% ({n_prof}/{n_total} sesji zyskiem)")
        print(f"    Średni stan konta po 20 betach: {initial_bankroll + avg_p:.2f} PLN (Średni zysk: {avg_p:+.2f} PLN | ROI: {avg_p/initial_bankroll*100:+.1f}%)")
        print(f"    Najlepszy miesiąc: {initial_bankroll + max_p:.2f} PLN ({max_p:+.2f} PLN)")
        print(f"    Najgorszy miesiąc: {initial_bankroll + min_p:.2f} PLN ({min_p:+.2f} PLN)")
        print(f"    Średnia skuteczność trafień: {avg_wr:.1f}%")
        print(f"    Średnie maksymalne obsunięcie w trakcie miesiąca: {avg_dd:.2f} PLN")
        print(f"    Ryzyko wyzerowania kapitału 50 PLN: {n_bust / n_total * 100:.1f}%\n")

    print("\n" + "=" * 96)
    print(f"BENCHMARK MIKRO-PORTFELA: START = {initial_bankroll:.2f} PLN (Stawka płaska = {stake_pln:.2f} PLN / kupon)")
    print(f"HORYZONT CZASOWY CAŁEJ SESJI: {global_start} do {global_end} ({global_days} dni | ~{global_months} miesięcy | {global_years} roku)")
    print("Symulacja walk-forward na 1520 realnych meczach z 3 czołowych lig europejskich")
    print("=" * 96)
    evaluate(p_high, "1. Wysokie kursy (> 3.0)")
    evaluate(p_baseline, "2. Baseline (Czysta 1)")
    evaluate(p_singles, "3. Capital Preservation Singles (1X / DNB)")
    evaluate(p_parlays, "4. Optymalne Duble (2-Leg Parlay Dixon-Coles)")

    print("\n" + "=" * 96)
    print(f"MIESIĘCZNY SPRINT: '20 BETÓW I KONIEC' (Start = {initial_bankroll:.2f} PLN)")
    print("Symulacja 20-zakładowych wyzwań inwestycyjnych na przestrzeni 4 lat")
    print("=" * 96)
    evaluate_sprints(p_parlays, "Optymalne Duble (Stawka 2.50 PLN = 5% banku)", n_bets=20, stake=2.50)
    evaluate_sprints(p_parlays, "Optymalne Duble (Stawka 4.00 PLN = 8% banku)", n_bets=20, stake=4.00)
    evaluate_sprints(p_singles, "Capital Preservation Singles (Stawka 2.50 PLN = 5% banku)", n_bets=20, stake=2.50)
    print("=" * 96 + "\n")


if __name__ == "__main__":
    run_micro_bankroll_benchmark(initial_bankroll=50.0, stake_pln=2.0)
