"""Interactive Match Intelligence Dashboard (Streamlit).

Visualizes real-time Betclic fixtures, expected values (EV), Poisson score grids,
Quarter-Kelly capital allocations, and interactive bankroll risk controls.
"""

import datetime
import json
from pathlib import Path
import sys

# Upewniamy się, że główny katalog projektu jest zawsze w sys.path (niezależnie od sposobu uruchomienia streamlit)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.optimizer.day_optimizer import DayReturnOptimizer
from src.sports.football.feed import FootballBetclicFeed
from src.sports.football.model import FootballInferenceEngine

# Konfiguracja strony
st.set_page_config(
    page_title="Betclic Match Intelligence | Day Return Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
LIVE_PREDS_CSV = ARTIFACTS_DIR / "gameweek_live_predictions.csv"
LIVE_PREDS_JSON = ARTIFACTS_DIR / "gameweek_live_predictions.json"


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@st.cache_data(ttl=300)
def get_all_opportunities():
    cfg = load_config()
    feed = FootballBetclicFeed(leagues_config=cfg.get("leagues", []))
    events = feed.fetch_live_events(force_refresh=False)
    engine = FootballInferenceEngine(config=cfg)
    return engine.evaluate_all(events)


def load_predictions():
    if LIVE_PREDS_CSV.exists():
        df = pd.read_csv(LIVE_PREDS_CSV)
        return df
    return pd.DataFrame()


# Styl CSS (Glassmorphism & Dark UI)
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    .ev-badge-pos {
        background-color: #059669;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .ev-badge-pass {
        background-color: #4b5563;
        color: #d1d5db;
        padding: 4px 8px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Nagłówek
st.title("⚽ La Liga Match Intelligence & Decision System")
st.caption("Quant Strategy: Poisson GLM (93%) + LightGBM (7%) • Quarter-Kelly Allocation • Real-Time Betclic PL Feed")

# Sidebar - Kontrola parametrów
st.sidebar.header("⚙️ Parametry Decyzyjne")
config = load_config()

default_bankroll = float(config.get("general", {}).get("default_bankroll", 50.0))
bankroll = st.sidebar.number_input("Twój Bankroll (PLN)", min_value=10.0, max_value=100_000.0, value=default_bankroll, step=10.0, help="Twój całkowity budżet przeznaczony na grę (np. 50 zł)")

min_ev_input = st.sidebar.slider("Minimalny Próg EV (%)", min_value=0.0, max_value=20.0, value=float(config.get("strategy", {}).get("min_ev", 0.04)) * 100, step=1.0) / 100.0
max_ev_input = st.sidebar.slider("Maksymalny Próg EV - Filtr Pułapek (%)", min_value=5.0, max_value=35.0, value=float(config.get("strategy", {}).get("max_ev", 0.18)) * 100, step=1.0, help="Odrzuca podejrzanie wysokie EV (>18%), które często oznaczają pułapkę bukmachera (np. nagła kontuzja kluczowego gracza, rotacja składu).") / 100.0

st.sidebar.markdown("### 🎯 Profil Ryzyka & Wariancja")
odds_range = st.sidebar.slider(
    "Zakres kursów (Widełki)",
    min_value=1.10,
    max_value=3.50,
    value=(float(config.get("strategy", {}).get("min_odds", 1.15)), float(config.get("strategy", {}).get("max_odds", 1.85))),
    step=0.05,
    help="Ogranicza zakłady do bezpieczniejszych kursów (1.15 - 1.85) eliminując dużą zmienność portfela."
)

min_win_prob_input = st.sidebar.slider(
    "Minimalna szansa modelu (%)",
    min_value=40.0,
    max_value=85.0,
    value=float(config.get("strategy", {}).get("min_model_prob", 0.70)) * 100,
    step=5.0,
    help="Odrzuca typy o niskim prawdopodobieństwie sukcesu, gwarantując wysoki Win Rate i ochronę kapitału."
) / 100.0

tax_free = st.sidebar.toggle("Gra bez podatku (0% tax)", value=(config.get("betclic", {}).get("tax_rate", 0.0) == 0.0))

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Odśwież kursy z Betclic na żywo"):
    with st.spinner("Odpytywanie Betclic i przeliczanie modelu..."):
        from src.analysis.run_matchday import run_matchday_pipeline
        run_matchday_pipeline()
    st.sidebar.success("Kursy i predykcje zaktualizowane!")
    st.rerun()

# Wczytanie danych
df = load_predictions()

if df.empty:
    st.warning("Brak wygenerowanych predykcji. Kliknij przycisk po lewej stronie, aby pobrać kursy z Betclica.")
    st.stop()

# Filtr wyboru ligi
available_leagues = sorted(df["league_name"].unique().tolist()) if "league_name" in df.columns else ["La Liga"]
selected_leagues = st.sidebar.multiselect(
    "🏆 Wybierz ligi do analizy",
    options=available_leagues,
    default=available_leagues,
    help="Filtruj spotkania pod kątem konkretnych rozgrywek ligowych."
)
if selected_leagues:
    df = df[df["league_name"].isin(selected_leagues)]

# Filtr Horyzontu Czasowego (Mecze na Dziś / Weekend / Całość)
time_filter = st.sidebar.radio(
    "📅 Horyzont czasowy",
    options=["Wszystkie nadchodzące", "Tylko DZIŚ (Mecze bieżące)", "Najbliższe 3 dni (Weekend)"],
    index=0
)
today_str = datetime.date.today().strftime("%Y-%m-%d")
if time_filter == "Tylko DZIŚ (Mecze bieżące)":
    df = df[df["match_date"] == today_str]
elif time_filter == "Najbliższe 3 dni (Weekend)":
    in_3_days = (datetime.date.today() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
    df = df[(df["match_date"] >= today_str) & (df["match_date"] <= in_3_days)]

# Przeliczenie stawek dla bieżącego bankrolla i filtrów profilu ryzyka
tax_multiplier = 1.0 if tax_free else 0.88

# Używamy wyselekcjonowanego rynku inwestycyjnego (1X / DNB / 1) jeśli istnieje w danych
eval_ev = df["rec_ev"] if "rec_ev" in df.columns else ((df["prob_home"] * (df["odds_home"] * tax_multiplier)) - 1.0)
eval_odd = df["rec_odd"] if "rec_odd" in df.columns else df["odds_home"]
eval_prob = df["rec_prob"] if "rec_prob" in df.columns else df["prob_home"]

df["current_ev"] = eval_ev

# Warunek decyzyjny łączący EV + Filtr pułapek + Widełki kursowe + Minimalną szansę sukcesu
df["is_actionable"] = (
    (eval_ev >= min_ev_input) &
    (eval_ev <= max_ev_input) &
    (eval_odd >= odds_range[0]) &
    (eval_odd <= odds_range[1]) &
    (eval_prob >= min_win_prob_input)
)

# Dynamiczny Quarter Kelly pod wybrany rynek inwestycyjny (z uwzględnieniem minimalnej stawki w Betclic = 1-2 zł)
def calc_stake(row):
    if not row["is_actionable"]:
        return 0.0, 0.0
    ev = row["current_ev"]
    odd = row.get("rec_odd", row["odds_home"])
    b = (odd * tax_multiplier) - 1.0
    full_k = ev / b if b > 0 else 0.0
    # Przy małym bankrollu (np. 50 zł) pozwalamy na stawkę 2-5 zł (4-10% banku), żeby kupon był przyjmowany przez Betclic (min. 1-2 zł)
    max_risk_cap = 0.08 if bankroll <= 100.0 else 0.035
    adj_k = min(max_risk_cap, full_k * 0.25)
    stake_pln = round(adj_k * bankroll, 2)
    # Betclic wymaga minimum 1.00 zł na kupon
    if stake_pln < 1.0 and bankroll >= 10.0:
        stake_pln = 1.0
        adj_k = 1.0 / bankroll
    return round(adj_k * 100, 2), stake_pln

res = df.apply(calc_stake, axis=1)
df["stake_pct_calc"] = [r[0] for r in res]
df["stake_units_calc"] = [r[1] for r in res]

# KPI Bar
actionable_df = df[df["is_actionable"]]
total_stake = actionable_df["stake_units_calc"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Monitorowane mecze", len(df))
with col2:
    st.metric("Rekomendowane zakłady", len(actionable_df))
with col3:
    avg_ev = actionable_df["current_ev"].mean() * 100 if not actionable_df.empty else 0.0
    st.metric("Średnie EV rekomendacji", f"{avg_ev:+.1f}%")
with col4:
    st.metric("Łączna ekspozycja", f"{total_stake:.2f} PLN", f"{(total_stake/bankroll)*100:.1f}% kapitału")

st.markdown("---")

# Zakładki
tab_day, tab_bets, tab_parlays, tab_all, tab_visuals = st.tabs([
    "📅 Dzienny Optymalizator (Day Return Optimizer)",
    "🎯 Wszystkie Rekomendacje (Single)",
    "⚡ Duble Inwestycyjne (2-Leg Parlay)",
    "📋 Pełna tabela spotkań",
    "📊 Wykresy i Rozkłady"
])

with tab_day:
    st.subheader("📅 Dzienny Optymalizator Zwrotu (Day Return Optimizer)")
    st.caption("Wybierz konkretny dzień i wygeneruj portfel kilku do kilkunastu najbardziej pewnych betów, gwarantujących maksymalny zwrot i minimalne ryzyko.")

    try:
        opps = get_all_opportunities()
        available_dates = sorted(list(set(str(o.event_date) for o in opps)))
    except Exception as e:
        st.error(f"Błąd podczas ładowania możliwości inwestycyjnych: {e}")
        opps = []
        available_dates = []

    if not available_dates:
        st.info("Brak dostępnych terminów w ofercie.")
    else:
        # Pasek kontrolny dnia
        c_date, c_mode, c_size, c_budget = st.columns([2, 3, 3, 2])
        
        # Wybór dnia
        today_iso = datetime.date.today().isoformat()
        default_date_idx = available_dates.index(today_iso) if today_iso in available_dates else 0
        selected_date_str = c_date.selectbox("🗓️ Wybierz dzień", available_dates, index=default_date_idx)
        selected_date = datetime.date.fromisoformat(selected_date_str)

        # Wybór trybu
        mode_label = c_mode.radio(
            "🎯 Cel optymalizacji",
            ["🛡️ Najbardziej Pewne (Pewniaki Dnia)", "💰 Maksymalizator Zwrotu (Max EV)", "⚖️ Zbalansowany Sharpe"],
            index=0,
            horizontal=False
        )
        mode_key_map = {
            "🛡️ Najbardziej Pewne (Pewniaki Dnia)": "safest_pewniaki",
            "💰 Maksymalizator Zwrotu (Max EV)": "max_return",
            "⚖️ Zbalansowany Sharpe": "balanced_sharpe"
        }
        chosen_mode = mode_key_map[mode_label]

        # Rozmiar portfela (kilka do kilkunastu)
        slate_size = c_size.slider("Liczba betów na ten dzień", min_value=3, max_value=15, value=5, help="Wybierz ile betów (od 3 do 15) chcesz postawić w tym dniu.")

        # Budżet na ten konkretny dzień
        day_budget = c_budget.number_input("Budżet na ten dzień (PLN)", min_value=10.0, max_value=1000.0, value=float(bankroll), step=10.0)

        # Uruchomienie DayReturnOptimizer
        optimizer = DayReturnOptimizer(
            min_probability=min_win_prob_input,
            min_ev=min_ev_input,
            max_ev=max_ev_input,
            min_odds=odds_range[0],
            max_odds=odds_range[1],
        )

        slate = optimizer.optimize_daily_slate(
            opportunities=opps,
            target_date=selected_date,
            daily_budget=day_budget,
            slate_size=slate_size,
            selection_mode=chosen_mode
        )

        daily_parlays = optimizer.generate_daily_parlays(slate, max_parlays=3)

        if not slate.selections:
            st.warning(f"Brak zakładów spełniających kryteria dla dnia {selected_date_str}. Spróbuj poluzować suwaki w lewym pasku lub wybrać inny dzień.")
        else:
            # Górny panel statystyk dnia
            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric("Oczekiwany zysk dzienny", f"{slate.expected_daily_profit_pln:+.2f} PLN", f"ROI: {slate.expected_daily_roi_pct:+.1f}%")
            with sm2:
                st.metric("Średnia szansa trafienia", f"{slate.avg_probability*100:.1f}%", f"Średni kurs: {slate.avg_odds:.2f}")
            with sm3:
                st.metric("Szansa, że WSZYSTKIE wejdą", f"{slate.prob_all_win*100:.1f}%", f"Zysk: +{slate.profit_if_all_win_pln:.2f} PLN")
            with sm4:
                st.metric("Szansa min. (N-1) wygranych", f"{slate.prob_at_least_80pct_win*100:.1f}%", f"Wynik przy 1 błędzie: {slate.profit_if_one_loses_pln:+.2f} PLN")

            st.markdown(f"### 📋 Selekcja Dnia: **{len(slate.selections)} Najlepszych Betów** (Łączna stawka: **{slate.total_stake_pln:.2f} PLN** / {day_budget:.2f} PLN)")
            
            for idx, bet in enumerate(slate.selections, 1):
                with st.container():
                    b_col1, b_col2, b_col3, b_col4 = st.columns([4, 2, 2, 2])
                    with b_col1:
                        st.markdown(f"#### **#{idx} [{bet.league_name}] {bet.match_display}**")
                        st.markdown(f"🎯 Rekomendowany rynek: **<span style='color:#10b981; font-size:1.15em;'>{bet.market_name}</span>**", unsafe_allow_html=True)
                        st.caption(f"💡 {bet.rationale}")
                    with b_col2:
                        st.metric("Kurs Betclic", f"{bet.odds:.2f}")
                        st.caption(f"Kurs sprawiedliwy (Fair): {bet.fair_odds:.2f}")
                    with b_col3:
                        st.metric("Pewność Modelu", f"{bet.model_probability*100:.1f}%", f"EV: {bet.expected_value*100:+.1f}%")
                        st.caption(f"Score zaufania: {bet.confidence_score:.1f}")
                    with b_col4:
                        potential_win = round(bet.recommended_stake_pln * bet.odds, 2)
                        st.metric("Stawka na ten mecz", f"{bet.recommended_stake_pln:.2f} PLN", f"{bet.kelly_stake_pct}% budżetu")
                        st.caption(f"Wygrana: **{potential_win:.2f} PLN** (Zysk: +{potential_win - bet.recommended_stake_pln:.2f} PLN)")
                    st.markdown("---")

            if daily_parlays:
                st.markdown("### ⚡ Rekomendowane Duble Dnia (Optymalne pary z powyższych meczów)")
                d_cols = st.columns(len(daily_parlays))
                for idx, (d_col, p) in enumerate(zip(d_cols, daily_parlays), 1):
                    with d_col:
                        st.info(
                            f"**DUBEL #{idx} | Kurs Łączny: {p['combined_odds']:.2f}**\n\n"
                            f"🔥 Szansa wejścia: **{p['combined_prob']*100:.1f}%** | EV: **{p['ev']*100:+.1f}%**\n\n"
                            f"⚽ **Mecz 1:** {p['leg1']['match']} -> **{p['leg1']['market']}** ({p['leg1']['odds']:.2f})\n\n"
                            f"⚽ **Mecz 2:** {p['leg2']['match']} -> **{p['leg2']['market']}** ({p['leg2']['odds']:.2f})\n\n"
                            f"💵 **Stawka:** **{p['stake_pln']:.2f} PLN** -> Do wygrania: **{p['potential_return_pln']:.2f} PLN**"
                        )

PARLAYS_JSON = ARTIFACTS_DIR / "optimal_parlays.json"

with tab_parlays:
    st.subheader("⚡ Generator Optymalnych Dubli Inwestycyjnych (2-Leg Parlay)")
    st.caption("Łączenie dwóch niezależnych spotkań o bardzo wysokiej pewności (np. Arsenal + Leverkusen) w celu multiplikacji zysku przy zachowaniu szansy sukcesu > 75%")

    if PARLAYS_JSON.exists():
        with open(PARLAYS_JSON, "r", encoding="utf-8") as f:
            parlays_data = json.load(f)
    else:
        parlays_data = []

    if not parlays_data:
        st.info("Brak wygenerowanych parlayów. Kliknij 'Odśwież kursy z Betclic na żywo' w pasku bocznym.")
    else:
        for idx, p in enumerate(parlays_data, start=1):
            with st.container():
                st.markdown(f"#### **Dubel #{idx}: Kurs Łączny {p['combined_odds']:.2f}** | Pewność Kuponu: **{p['combined_prob']*100:.1f}%** | EV: **{p['ev']*100:+.1f}%**")
                pc1, pc2, pc3 = st.columns([4, 4, 3])
                with pc1:
                    st.info(f"**Noga 1:** [{p['leg1']['league']}] **{p['leg1']['match']}**\n\n🎯 Typ: **{p['leg1']['market']}** (Kurs: {p['leg1']['odd']:.2f} | Szansa: {p['leg1']['prob']*100:.1f}%)")
                with pc2:
                    st.info(f"**Noga 2:** [{p['leg2']['league']}] **{p['leg2']['match']}**\n\n🎯 Typ: **{p['leg2']['market']}** (Kurs: {p['leg2']['odd']:.2f} | Szansa: {p['leg2']['prob']*100:.1f}%)")
                with pc3:
                    parlay_pct = p['stake_pct']
                    if bankroll <= 100.0:
                        parlay_stake = max(2.0, round((parlay_pct / 100.0) * bankroll, 2))
                        parlay_pct = round((parlay_stake / bankroll) * 100.0, 1)
                    else:
                        parlay_stake = round((parlay_pct / 100.0) * bankroll, 2)
                    st.metric("Rekomendowana Stawka", f"{parlay_stake:.2f} PLN", f"{parlay_pct}% banku")
                    potential_win = round(parlay_stake * p['combined_odds'], 2)
                    st.caption(f"Potencjalna wygrana: **{potential_win:.2f} PLN**")
                st.markdown("---")

with tab_bets:
    st.subheader("🛡️ Portfel Inwestycyjny: Capital Preservation (1X / DNB / 1)")
    st.caption("Filtry: Pewność modelu ≥ 70%, kursy 1.15 - 1.85, minimalizacja wariancji kapitału")
    if actionable_df.empty:
        st.info("Brak spotkań spełniających zadane kryteria. Możesz poluzować suwaki w pasku bocznym.")
    else:
        for _, row in actionable_df.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                with c1:
                    league_badge = f"[{row.get('league_name', 'La Liga')}] "
                    st.markdown(f"### **{league_badge}{row['match']}**")
                    market_txt = row.get("market_chosen", "1X (Win or Draw)")
                    st.markdown(f"🎯 **Rekomendowany typ: <span style='color:#10b981; font-size:1.15em;'>{market_txt}</span>**", unsafe_allow_html=True)
                    st.caption(f"📅 Data: {row['match_date']} | Bilans xG 5m: {row['roll_xg_diff_5m']:+.2f} | Różnica Elo: {row['elo_diff']:+.1f}")
                with c2:
                    st.metric("Kurs Zakładu", f"{row.get('rec_odd', row['odds_home']):.2f}")
                    st.caption(f"1X: {row.get('odds_1x', 1.25)} | DNB: {row.get('odds_dnb', 1.30)} | 1: {row['odds_home']:.2f}")
                with c3:
                    rec_p = row.get("rec_prob", row["prob_home"]) * 100
                    rec_ev = row.get("rec_ev", row["current_ev"]) * 100
                    st.metric("Pewność Modelu", f"{rec_p:.1f}%", f"EV: {rec_ev:+.1f}%")
                    st.caption(f"Szansa 1X: {row.get('prob_1x', 0.8)*100:.1f}% | Szansa DNB: {row.get('prob_dnb', 0.8)*100:.1f}%")
                with c4:
                    st.metric("Stawka (Quarter-Kelly)", f"{row['stake_units_calc']:.2f} PLN", f"{row['stake_pct_calc']:.2f}% banku")
                st.markdown("---")

with tab_all:
    st.subheader("Wszystkie analizowane spotkania ze wszystkich lig")
    
    display_df = df.copy()
    display_df["Pewność (%)"] = (display_df["rec_prob"] * 100).round(1) if "rec_prob" in display_df.columns else (display_df["prob_home"] * 100).round(1)
    display_df["EV (%)"] = (display_df["rec_ev"] * 100).round(1) if "rec_ev" in display_df.columns else (display_df["current_ev"] * 100).round(1)
    display_df["Kurs Rekomendowany"] = display_df["rec_odd"] if "rec_odd" in display_df.columns else display_df["odds_home"]
    display_df["Rekomendowany Rynek"] = display_df["market_chosen"] if "market_chosen" in display_df.columns else "1X"
    display_df["Szansa 1X (%)"] = (display_df["prob_1x"] * 100).round(1) if "prob_1x" in display_df.columns else 0.0
    display_df["Szansa DNB (%)"] = (display_df["prob_dnb"] * 100).round(1) if "prob_dnb" in display_df.columns else 0.0
    display_df["Stawka (PLN)"] = display_df["stake_units_calc"]
    display_df["Status"] = display_df["is_actionable"].map({True: "✅ REKOMENDOWANY", False: "❌ ODPUŚĆ"})

    cols_to_show = [
        "league_name", "match_date", "match", "Rekomendowany Rynek", "Kurs Rekomendowany",
        "Pewność (%)", "EV (%)", "Szansa 1X (%)", "Szansa DNB (%)", "Status", "Stawka (PLN)"
    ]

    st.dataframe(
        display_df[cols_to_show].sort_values("Pewność (%)", ascending=False),
        use_container_width=True,
        hide_index=True
    )

with tab_visuals:
    st.subheader("Wizualizacja przewagi matematycznej (Model Prob vs Implied Odds)")
    
    # Wyliczamy uczciwą implikowaną szansę rynkową (usuwając overround) jeśli nie ma w ramce
    if "fair_prob_home" not in df.columns:
        inv_sum = (1.0 / df["odds_home"]) + (1.0 / df["odds_draw"]) + (1.0 / df["odds_away"])
        df["fair_prob_home"] = ((1.0 / df["odds_home"]) / inv_sum).round(4)
    
    hover_cols = ["match", "league_name", "market_chosen", "rec_odd", "rec_prob", "rec_ev"] if "rec_odd" in df.columns else ["match", "odds_home", "current_ev"]
    
    y_col = "rec_prob" if "rec_prob" in df.columns else "prob_home"
    size_col = "rec_odd" if "rec_odd" in df.columns else "odds_home"

    fig = px.scatter(
        df,
        x="fair_prob_home",
        y=y_col,
        color="is_actionable",
        color_discrete_map={True: "#10b981", False: "#6b7280"},
        size=size_col,
        hover_data=hover_cols,
        labels={
            "fair_prob_home": "Uczciwa szansa bukmachera (Implied Fair)",
            y_col: "Pewność wybranego rynku inwestycyjnego (1X / DNB / 1)"
        },
        title="Rozkład szans: Punkty powyżej linii przerywanej to dodatnia wartość (Edge > 0)"
    )
    # Linia neutralna (x = y)
    fig.add_shape(
        type="line", line=dict(dash="dash", color="white", width=1.5),
        x0=0, y0=0, x1=1, y1=1
    )
    fig.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Wykres EV wg meczów
    st.subheader("Wartość Oczekiwana (EV) dla wyselekcjonowanych rynków")
    bar_x = "rec_ev" if "rec_ev" in df.columns else "current_ev"
    
    fig_bar = px.bar(
        df.sort_values(bar_x, ascending=True),
        x=bar_x,
        y="match",
        orientation="h",
        color="is_actionable",
        color_discrete_map={True: "#10b981", False: "#ef4444"},
        labels={bar_x: "Expected Value (EV)", "match": "Mecz"},
        title="Wartość oczekiwana (Zielone: kwalifikuje się do portfela inwestycyjnego)"
    )
    fig_bar.add_vline(x=min_ev_input, line_dash="dash", line_color="yellow", annotation_text=f"Próg EV ({min_ev_input*100:.0f}%)")
    fig_bar.update_layout(height=650, template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)