"""Multi-League Ingestion and Real-Time Market Scraper for Betclic PL.

Extracts live fixtures and pre-match odds for multiple competitions:
- La Liga (ESP)
- Premier League (ENG)
- Bundesliga (GER)
Normalizes team entities across all leagues and outputs consolidated feeds.
"""

import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MultiLeagueBetclicFeed")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Słownik encji ujednolicający polskie nazwy klubów z Betclic do formy kanonicznej (Understat / Football-Data)
BETCLIC_TO_CANONICAL_TEAMS: Dict[str, str] = {
    # --- La Liga ---
    "Alaves": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Athletic Bilbao": "Athletic Club",
    "Athletic Club": "Athletic Club",
    "Atletico Madryt": "Atletico Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona",
    "FC Barcelona": "Barcelona",
    "Betis": "Real Betis",
    "Real Betis": "Real Betis",
    "Celta Vigo": "Celta Vigo",
    "Celta": "Celta Vigo",
    "Deportivo A Corunya": "Deportivo La Coruna",
    "Deportivo La Coruna": "Deportivo La Coruna",
    "Elche": "Elche",
    "Espanyol": "Espanyol",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "Levante": "Levante",
    "Mallorca": "Mallorca",
    "RCD Mallorca": "Mallorca",
    "Malaga": "Malaga",
    "Osasuna": "Osasuna",
    "Racing Santander": "Racing Santander",
    "Rayo Vallecano": "Rayo Vallecano",
    "Real Madryt": "Real Madrid",
    "Real Madrid": "Real Madrid",
    "Real Oviedo": "Real Oviedo",
    "Real Sociedad": "Real Sociedad",
    "Real Valladolid": "Real Valladolid",
    "Sevilla": "Sevilla",
    "Sevilla FC": "Sevilla",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",

    # --- Premier League ---
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Manchester United": "Manchester United",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",

    # --- Bundesliga ---
    "Augsburg": "Augsburg",
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Leverkusen": "Bayer Leverkusen",
    "Bayern Monachium": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Bochum": "Bochum",
    "Borussia Dortmund": "Borussia Dortmund",
    "Dortmund": "Borussia Dortmund",
    "Borussia Moenchengladbach": "Borussia M.Gladbach",
    "M'gladbach": "Borussia M.Gladbach",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Frankfurt": "Eintracht Frankfurt",
    "Freiburg": "Freiburg",
    "Heidenheim": "FC Heidenheim",
    "Hoffenheim": "Hoffenheim",
    "Holstein Kiel": "Holstein Kiel",
    "Mainz": "Mainz 05",
    "RB Lipsk": "RasenBallsport Leipzig",
    "RB Leipzig": "RasenBallsport Leipzig",
    "St. Pauli": "St. Pauli",
    "Stuttgart": "VfB Stuttgart",
    "Union Berlin": "Union Berlin",
    "Werder Brema": "Werder Bremen",
    "Werder Bremen": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
}


def normalize_team_name(name: str) -> str:
    clean = name.strip()
    return BETCLIC_TO_CANONICAL_TEAMS.get(clean, clean)


def fetch_betclic_league_odds(
    url: str,
    league_name: str,
    league_code: str,
    timeout: int = 10,
) -> pd.DataFrame:
    """Pobiera ofertę meczową z pojedynczego endpointu ligowego w Betclic."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    logger.info("Odpytywanie Betclic dla ligi: %s (%s)...", league_name, url)
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Błąd sieci dla %s: %s", league_name, e)
        return pd.DataFrame()

    script_matches = re.findall(r"<script[^>]*>(.*?)</script>", resp.text, re.DOTALL)
    extracted = []

    for content in script_matches:
        if "grpc:" not in content and "marketTypeCodes" not in content:
            continue
        try:
            state_data = json.loads(content)
        except Exception:
            continue

        for key, val in state_data.items():
            if not isinstance(val, dict):
                continue
            payload = val.get("response", {}).get("payload") if isinstance(val.get("response"), dict) else val.get("payload")
            if not payload or not isinstance(payload, dict):
                continue
            matches_list = payload.get("matches", [])
            if not matches_list:
                continue

            for m in matches_list:
                m_name = m.get("name", "")
                m_date_raw = m.get("matchDateUtc")
                m_id = m.get("matchId", "")
                market = m.get("market")
                if not market:
                    continue

                selections = market.get("mainSelections", [])
                if len(selections) != 3:
                    continue

                home_raw = selections[0].get("name", "").strip()
                draw_raw = selections[1].get("name", "").strip()
                away_raw = selections[2].get("name", "").strip()

                odds_h = float(selections[0].get("odds", 0.0))
                odds_x = float(selections[1].get("odds", 0.0))
                odds_a = float(selections[2].get("odds", 0.0))

                match_date = m_date_raw[:10] if m_date_raw else ""

                extracted.append(
                    {
                        "league_name": league_name,
                        "league_code": league_code,
                        "match_id": f"betclic_{m_id}",
                        "match_date": match_date,
                        "raw_fixture_name": m_name,
                        "home_team_raw": home_raw,
                        "away_team_raw": away_raw,
                        "home_team": normalize_team_name(home_raw),
                        "away_team": normalize_team_name(away_raw),
                        "odds_home": odds_h,
                        "odds_draw": odds_x,
                        "odds_away": odds_a,
                        "source": "betclic_pl",
                    }
                )
            if extracted:
                break

    df = pd.DataFrame(extracted)
    if not df.empty:
        df = df.drop_duplicates(subset=["home_team", "away_team", "match_date"])
        logger.info("-> Pomyślnie wyciągnięto %d spotkań dla %s.", len(df), league_name)
    else:
        logger.warning("-> Nie znaleziono spotkań dla %s.", league_name)
    return df


def fetch_all_configured_leagues(leagues_config: List[dict]) -> pd.DataFrame:
    """Pobiera ofertę ze wszystkich lig zdefiniowanych w config.yaml."""
    dfs = []
    for l_conf in leagues_config:
        df_l = fetch_betclic_league_odds(
            url=l_conf["betclic_url"],
            league_name=l_conf["name"],
            league_code=l_conf["code"],
        )
        if not df_l.empty:
            dfs.append(df_l)

    if not dfs:
        raise ValueError("Nie udało się pobrać żadnych kursów z Betclica dla skonfigurowanych lig.")

    all_odds = pd.concat(dfs, ignore_index=True)
    out_parquet = RAW_DATA_DIR / "betclic_live_odds.parquet"
    out_csv = RAW_DATA_DIR / "betclic_live_odds.csv"
    all_odds.to_parquet(out_parquet, index=False)
    all_odds.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Zapisano łącznie %d spotkań ze wszystkich lig.", len(all_odds))
    return all_odds


# Kompatybilność wsteczna
def fetch_betclic_laliga_odds() -> pd.DataFrame:
    return fetch_betclic_league_odds(
        url="https://www.betclic.pl/pilka-nozna-s1/hiszpania-la-liga-c7",
        league_name="La Liga (Hiszpania)",
        league_code="SP1",
    )


def save_betclic_odds(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    if output_path is None:
        output_path = RAW_DATA_DIR / "betclic_live_odds.parquet"
    df.to_parquet(output_path, index=False)
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    from src.analysis.run_matchday import load_config
    cfg = load_config()
    leagues = cfg.get("leagues", [])
    df_all = fetch_all_configured_leagues(leagues)
    print("\n--- Podsumowanie pobranych lig ---")
    print(df_all.groupby("league_name").size())
