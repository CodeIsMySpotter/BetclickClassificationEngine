"""Interactive Match Intelligence Dashboard (Streamlit).

Visualizes real-time Betclic fixtures, expected values (EV), Poisson score grids,
Quarter-Kelly capital allocations, and interactive bankroll risk controls.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Konfiguracja strony
st.set_page_config(
    page_title="Betclic Match Intelligence | La Liga Quant",
    page_icon="⚽",
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

default_bankroll = float(config.get("general", {}).get("default_bankroll", 10000.0))
bankroll = st.sidebar.number_input("Twój Bankroll (PLN)", min_value=100.0, max_value=1_000_000.0, value=default_bankroll, step=500.0)

min_ev_input = st.sidebar.slider("Minimalny Próg EV (%)", min_value=0.0, max_value=25.0, value=float(config.get("strategy", {}).get("min_ev", 0.05)) * 100, step=1.0) / 100.0

st.sidebar.markdown("### 🎯 Profil Ryzyka & Wariancja")
odds_range = st.sidebar.slider(
    "Zakres kursów (Widełki)",
    min_value=1.10,
    max_value=6.00,
    value=(float(config.get("strategy", {}).get("min_odds", 1.25)), float(config.get("strategy", {}).get("max_odds", 2.20))),
    step=0.05,
    help="Ogranicza zakłady do bezpieczniejszych kursów (np. 1.30 - 2.00) eliminując dużą zmienność portfela."
)

min_win_prob_input = st.sidebar.slider(
    "Minimalna szansa modelu (%)",
    min_value=30.0,
    max_value=85.0,
    value=float(config.get("strategy", {}).get("min_model_prob", 0.50)) * 100,
    step=5.0,
    help="Odrzuca typy o niskim prawdopodobieństwie sukcesu, gwarantując wysoki Win Rate."
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

# Przeliczenie stawek dla bieżącego bankrolla i filtrów profilu ryzyka
tax_multiplier = 1.0 if tax_free else 0.88

# Używamy wyselekcjonowanego rynku inwestycyjnego (1X / DNB / 1) jeśli istnieje w danych
eval_ev = df["rec_ev"] if "rec_ev" in df.columns else ((df["prob_home"] * (df["odds_home"] * tax_multiplier)) - 1.0)
eval_odd = df["rec_odd"] if "rec_odd" in df.columns else df["odds_home"]
eval_prob = df["rec_prob"] if "rec_prob" in df.columns else df["prob_home"]

df["current_ev"] = eval_ev

# Warunek decyzyjny łączący EV + Widełki kursowe + Minimalną szansę sukcesu
df["is_actionable"] = (
    (eval_ev >= min_ev_input) &
    (eval_odd >= odds_range[0]) &
    (eval_odd <= odds_range[1]) &
    (eval_prob >= min_win_prob_input)
)

# Dynamiczny Quarter Kelly pod wybrany rynek inwestycyjny
def calc_stake(row):
    if not row["is_actionable"]:
        return 0.0, 0.0
    ev = row["current_ev"]
    odd = row.get("rec_odd", row["odds_home"])
    b = (odd * tax_multiplier) - 1.0
    full_k = ev / b if b > 0 else 0.0
    adj_k = min(0.035, full_k * 0.25)
    return round(adj_k * 100, 2), round(adj_k * bankroll, 2)

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
tab_bets, tab_parlays, tab_all, tab_visuals = st.tabs([
    "🎯 Rekomendacje Pojedyncze (Single)",
    "⚡ Duble Inwestycyjne (2-Leg Parlay)",
    "📋 Pełna tabela spotkań",
    "📊 Wykresy i Rozkłady"
])

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
                    parlay_stake = round((p['stake_pct'] / 100.0) * bankroll, 2)
                    st.metric("Rekomendowana Stawka", f"{parlay_stake:.2f} PLN", f"{p['stake_pct']}% banku")
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