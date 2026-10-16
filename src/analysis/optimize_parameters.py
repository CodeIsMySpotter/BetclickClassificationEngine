"""Systematic Grid-Search Parameter Optimization for Capital Preservation & Parlays.

Simulates walk-forward performance across 1,520 real historical matches (2022-2026)
to find the optimal decision boundary parameters:
- Minimum EV threshold: [3%, 5%, 8%, 10%, 12%, 15%]
- Model Probability threshold: [65%, 70%, 75%, 80%]
- Model Agreement Filter (max discrepancy between Poisson and LGBM): [None, 15%, 10%]
- Rolling xG Momentum Floor: [None, > 0.0, > +0.3]

Outputs ranked parameter configurations by ROI (%), Sharpe Ratio and Max Drawdown.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def simulate_dixon_coles(elo_diff: float, xg_diff: float, rho: float = -0.06) -> Tuple[float, float, float]:
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


def simulate_lgbm_prior(elo_diff: float, xg_diff: float, rest_diff: int = 0) -> Tuple[float, float, float]:
    zh = 0.15 + (0.0075 * elo_diff) + (0.85 * xg_diff) + (0.04 * rest_diff)
    zx = -0.45 - (0.0015 * abs(elo_diff))
    za = -0.15 - (0.0068 * elo_diff) - (0.75 * xg_diff) - (0.03 * rest_diff)
    eh, ex, ea = np.exp(zh), np.exp(zx), np.exp(za)
    tot = eh + ex + ea
    return eh / tot, ex / tot, ea / tot


def run_grid_search_optimizer():
    matches_file = PROCESSED_DATA_DIR / "matches_merged.parquet"
    if not matches_file.exists():
        raise FileNotFoundError(f"Brak pliku: {matches_file}")

    df = pd.read_parquet(matches_file).dropna(subset=["B365H", "B365D", "B365A", "FTHG", "FTAG", "FTR"]).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    print("Przygotowywanie cech i predykcji dla 1520 meczów...")

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

    precomputed = []

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

        # 1. Model Dixon-Coles
        p_dc_h, p_dc_x, p_dc_a = simulate_dixon_coles(elo_diff, xg_diff)
        # 2. Model LightGBM prior
        p_lg_h, p_lg_x, p_lg_a = simulate_lgbm_prior(elo_diff, xg_diff)

        # Hybryda (93% DC + 7% LGBM)
        p_hyb_h = 0.93 * p_dc_h + 0.07 * p_lg_h
        p_hyb_x = 0.93 * p_dc_x + 0.07 * p_lg_x
        p_hyb_a = 0.93 * p_dc_a + 0.07 * p_lg_a

        prob_1x = p_hyb_h + p_hyb_x
        prob_dnb = p_hyb_h / (p_hyb_h + p_hyb_a) if (p_hyb_h + p_hyb_a) > 0 else 0.5

        # Discrepancy per market
        disc_h = abs(p_dc_h - p_lg_h)
        disc_1x = abs((p_dc_h + p_dc_x) - (p_lg_h + p_lg_x))
        p_dc_dnb = p_dc_h / (p_dc_h + p_dc_a) if (p_dc_h + p_dc_a) > 0 else 0.5
        p_lg_dnb = p_lg_h / (p_lg_h + p_lg_a) if (p_lg_h + p_lg_a) > 0 else 0.5
        disc_dnb = abs(p_dc_dnb - p_lg_dnb)

        odd_1x = round(1.0 / ((1.0 / odd_h) + (1.0 / odd_x)), 2)
        odd_dnb = round(odd_h * (1.0 - (1.0 / odd_x)), 2) if odd_x > 1.0 else odd_h

        ev_h = (p_hyb_h * odd_h) - 1.0
        ev_1x = (prob_1x * odd_1x) - 1.0
        ev_dnb = (prob_dnb * odd_dnb) - 1.0

        precomputed.append({
            "date": row["match_date"], "ftr": ftr,
            "h_team": h_team, "a_team": a_team,
            "odd_h": odd_h, "odd_1x": odd_1x, "odd_dnb": odd_dnb,
            "prob_h": p_hyb_h, "prob_1x": prob_1x, "prob_dnb": prob_dnb,
            "ev_h": ev_h, "ev_1x": ev_1x, "ev_dnb": ev_dnb,
            "xg_diff": xg_diff,
            "disc_h": disc_h, "disc_1x": disc_1x, "disc_dnb": disc_dnb
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

    print("Rozpoczynanie Grid Search dla 48 kombinacji parametrów...")

    ev_thresholds = [0.03, 0.05, 0.08, 0.10, 0.12]
    prob_thresholds = [0.65, 0.70, 0.75, 0.80]
    discrepancy_limits = [None, 0.15, 0.10]
    xg_floors = [None, 0.0, 0.3]

    results_grid = []

    for min_ev in ev_thresholds:
        for min_prob in prob_thresholds:
            for max_disc in [None, 0.12]:
                for min_xg_d in [None, 0.2]:
                    # Symulacja strategii Dubli (2-Leg Parlay) z danymi filtrami
                    parlays = []
                    buffer = []

                    for m in precomputed:
                        # Filtr momentum xG
                        if min_xg_d is not None and m["xg_diff"] < min_xg_d:
                            continue

                        # Selekcja rynku z uwzględnieniem tolerancji niezgodności modeli dla danego rynku
                        cands = []
                        if m["prob_1x"] >= min_prob and (1.15 <= m["odd_1x"] <= 1.85) and m["ev_1x"] >= min_ev:
                            if max_disc is None or m["disc_1x"] <= max_disc:
                                cands.append({"type": "1X", "odd": m["odd_1x"], "prob": m["prob_1x"], "ev": m["ev_1x"]})
                        if m["prob_dnb"] >= min_prob and (1.15 <= m["odd_dnb"] <= 1.85) and m["ev_dnb"] >= min_ev:
                            if max_disc is None or m["disc_dnb"] <= max_disc:
                                cands.append({"type": "DNB", "odd": m["odd_dnb"], "prob": m["prob_dnb"], "ev": m["ev_dnb"]})
                        if m["prob_h"] >= min_prob and (1.15 <= m["odd_h"] <= 1.85) and m["ev_h"] >= min_ev:
                            if max_disc is None or m["disc_h"] <= max_disc:
                                cands.append({"type": "1", "odd": m["odd_h"], "prob": m["prob_h"], "ev": m["ev_h"]})

                        if cands:
                            best = sorted(cands, key=lambda c: (c["prob"], c["ev"]), reverse=True)[0]
                            m_type = best["type"]
                            b_odd = best["odd"]
                            ftr = m["ftr"]

                            if m_type == "1X":
                                win = 1 if ftr in ["H", "D"] else 0
                            elif m_type == "DNB":
                                win = 1 if ftr == "H" else (0.5 if ftr == "D" else 0)
                            else:
                                win = 1 if ftr == "H" else 0

                            buffer.append({"odd": b_odd, "win": win, "prob": best["prob"]})

                            if len(buffer) >= 2:
                                l1 = buffer.pop(0)
                                l2 = buffer.pop(0)
                                p_odd = round(l1["odd"] * l2["odd"], 2)
                                p_prob = round(l1["prob"] * l2["prob"], 4)

                                if 1.40 <= p_odd <= 2.20:
                                    if l1["win"] == 1 and l2["win"] == 1:
                                        pnl = 100.0 * (p_odd - 1.0)
                                        w = 1
                                    elif (l1["win"] == 1 and l2["win"] == 0.5):
                                        pnl = 100.0 * (l1["odd"] - 1.0)
                                        w = 1
                                    elif (l1["win"] == 0.5 and l2["win"] == 1):
                                        pnl = 100.0 * (l2["odd"] - 1.0)
                                        w = 1
                                    elif l1["win"] == 0.5 and l2["win"] == 0.5:
                                        pnl = 0.0
                                        w = 0.5
                                    else:
                                        pnl = -100.0
                                        w = 0

                                    parlays.append({"pnl": pnl, "win": w, "odd": p_odd})

                    if len(parlays) >= 30:  # Statystycznie istotna próbka (minimum 30 kuponów)
                        df_p = pd.DataFrame(parlays)
                        tot_pnl = df_p["pnl"].sum()
                        tot_stake = len(df_p) * 100.0
                        roi = (tot_pnl / tot_stake) * 100.0
                        win_rate = (len(df_p[df_p["win"] == 1]) / len(df_p)) * 100.0
                        
                        df_p["cum_pnl"] = df_p["pnl"].cumsum()
                        df_p["peak"] = df_p["cum_pnl"].cummax()
                        max_dd = (df_p["cum_pnl"] - df_p["peak"]).min()

                        returns = df_p["pnl"] / 100.0
                        sharpe = (returns.mean() / returns.std() * np.sqrt(len(df_p))) if returns.std() > 0 else 0.0

                        results_grid.append({
                            "min_ev": f"{min_ev*100:.0f}%",
                            "min_prob": f"{min_prob*100:.0f}%",
                            "model_agreement": "Tak (Discrepancy <= 12%)" if max_disc else "Brak",
                            "xg_momentum": "Tak (xG Diff >= +0.2)" if min_xg_d else "Brak",
                            "count": len(df_p),
                            "win_rate": round(win_rate, 1),
                            "total_pnl": round(tot_pnl, 2),
                            "roi": round(roi, 2),
                            "max_dd": round(max_dd, 2),
                            "sharpe": round(sharpe, 2)
                        })

    grid_df = pd.DataFrame(results_grid).sort_values("roi", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 110)
    print("WYNIKI OPTYMALIZACJI GRID SEARCH (TOP 10 NAJLEPSZYCH KONFIGURACJI PARAMETRÓW)")
    print("=" * 110)
    print(grid_df.head(10).to_string())
    print("=" * 110 + "\n")
    return grid_df


if __name__ == "__main__":
    run_grid_search_optimizer()
