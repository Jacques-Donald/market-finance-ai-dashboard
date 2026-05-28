import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="IA Finance Dashboard Pro", layout="wide")

# ==============================================================================
# CONFIGURATION DYNAMIQUE DE L'URL (Alignement MLOps Cloud & Port 8006)
# ==============================================================================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8006")

st.title("📊 Dashboard Financier Prédictif & Analyse des Risques")
st.markdown("Architecture Synchrone : BigQuery ➔ FastAPI ➔ Streamlit (XGBoost Inference)")

# ==============================================================================
# 1. FONCTION DE REQUÊTAGE RÉSEAU SÉCURISÉE AVEC MISE EN CACHE
# ==============================================================================
@st.cache_data(ttl=300)  # Cache de 5 minutes pour éviter de surcharger FastAPI / BigQuery
def fetch_api_predictions(ticker: str):
    """Effectue l'appel HTTP vers l'API et met en cache le résultat JSON."""
    try:
        response = requests.get(f"{API_URL}/predictions/{ticker}", timeout=30)
        if response.status_code == 200:
            return response.json(), "OK"
        elif response.status_code == 404:
            return None, "NOT_FOUND"
        else:
            return None, f"ERROR_CODE_{response.status_code}"
    except requests.exceptions.RequestException:
        return None, "CONNECTION_FAILED"

# ==============================================================================
# 2. CONFIGURATION DES ACTIFS ET SÉLECTION
# ==============================================================================
assets = {
    "S&P 500 (Actions US)": "^GSPC",
    "Nasdaq 100 (Technologie)": "^IXIC",
    "Bitcoin (Crypto-actif)": "BTC-USD",
    "Or (Valeur Refuge)": "GC=F",
    "Euro / Dollar (Forex)": "EURUSD=X"
}

col_asset, col_horizon = st.columns([2, 2])

with col_asset:
    selected_label = st.selectbox("🎯 Choisissez l'actif à analyser via BigQuery & XGBoost :", list(assets.keys()))
    ticker = assets[selected_label]
    is_forex = (ticker == "EURUSD=X")

# Déclenchement de la requête via la fonction cachée
data, status = fetch_api_predictions(ticker)

# Routage des alertes en fonction du statut de la requête
if status == "NOT_FOUND":
    st.error(f"❌ Actif non trouvé dans la base BigQuery : {ticker}")
elif status == "CONNECTION_FAILED":
    st.warning("⚠️ Impossible de joindre l'API sur le port 8006. Assurez-vous d'avoir lancé votre serveur Uvicorn.")
elif status.startswith("ERROR_CODE_"):
    st.error(f"❌ L'API a répondu avec une erreur : {status}")

# ==============================================================================
# 3. TRAITEMENT APPLICATIF & ENGINE DE RISQUE
# ==============================================================================
if data:
    try:
        df_base = pd.DataFrame(data)
        df_base['date'] = pd.to_datetime(df_base['date'])
        df_base = df_base.sort_values('date').reset_index(drop=True)

        # Sélecteur d'horizon temporel (Ne déclenche plus d'appels HTTP grâce au cache)
        with col_horizon:
            horizon = st.radio(
                "⏱️ Sélectionnez l'horizon temporel de l'analyse :",
                ["Hebdomadaire (7j)", "Mensuel (30j)", "Trimestriel (90j)", "Semestriel (180j)", "Annuel (365j)"],
                horizontal=True
            )
        
        latest_date = df_base['date'].max()
        
        if horizon == "Hebdomadaire (7j)":
            df = df_base[df_base['date'] >= (latest_date - pd.Timedelta(days=7))].copy()
            horizon_text = "7 derniers jours"
        elif horizon == "Mensuel (30j)":
            df = df_base[df_base['date'] >= (latest_date - pd.Timedelta(days=30))].copy()
            horizon_text = "30 derniers jours"
        elif horizon == "Trimestriel (90j)":
            df = df_base[df_base['date'] >= (latest_date - pd.Timedelta(days=90))].copy()
            horizon_text = "90 derniers jours (Vue Trimestrielle)"
        elif horizon == "Semestriel (180j)":
            df = df_base[df_base['date'] >= (latest_date - pd.Timedelta(days=180))].copy()
            horizon_text = "180 derniers jours (Vue Semestrielle)"
        else:
            df = df_base[df_base['date'] >= (latest_date - pd.Timedelta(days=365))].copy()
            horizon_text = "365 derniers jours (Vue Annuelle)"

        # Protection anti-crash
        if df.empty or len(df) < 2:
            df = df_base.copy()
            horizon_text = "Historique Total (Données temporelles insuffisantes sur la sélection)"

        # Extraction des KPIs de fin de période
        latest_row = df.iloc[-1]
        current_price = latest_row['price']
        predicted_price = latest_row['predicted_next_day_price']
        r2_score = latest_row.get('model_r2', 0.0)

        # CALCUL ANALYTIQUE DES RISQUES
        df['returns'] = np.log(df['price'] / df['price'].shift(1))
        df['returns'] = df['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        std_returns = df['returns'].std()
        volatility = std_returns * np.sqrt(252) * 100 if not pd.isna(std_returns) else 0.0
        
        # Inversion du signe de la VaR pour l'affichage conventionnel positif de la perte potentielle
        var_95 = abs(np.percentile(df['returns'], 5) * 100) if len(df) > 1 else 0.0
        
        risk_free_rate = 0.02 / 252
        excess_returns = df['returns'] - risk_free_rate
        
        if std_returns > 0:
            sharpe_ratio = (excess_returns.mean() / std_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Affichage des KPIs principaux
        st.write("---")
        col1, col2, col3 = st.columns(3)
        if is_forex:
            col1.metric("Cours de Clôture Actuel", f"{current_price:.4f}")
            col2.metric("Prédiction XGBoost (J+1)", f"{predicted_price:.4f}")
        else:
            col1.metric("Prix de Clôture Actuel", f"${current_price:,.2f}")
            col2.metric("Prédiction XGBoost (J+1)", f"${predicted_price:,.2f}")
        col3.metric("Précision Globale du Modèle (R²)", f"{r2_score*100:.2f}%")

        # GRAPHIQUE PLOTLY
        st.subheader(f"📈 Analyse Comparative des Trajectoires ({horizon_text})")
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['price'],
            name="Prix / Cours Réel de Clôture",
            line=dict(color='#1f77b4', width=3),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['predicted_next_day_price'],
            name="Prédiction J+1 (XGBoost)",
            line=dict(color='#ff7f0e', width=2.5, dash='dash'),
            mode='lines'
        ))
        
        yaxis_title_text = "Taux de Change (EUR/USD)" if is_forex else "Cours de l'Actif (USD)"
        tick_format_type = ".4f" if is_forex else ",.2f"
        
        # --- BLOC ENTIÈREMENT SÉCURISÉ POUR UPDATE_LAYOUT ---
        fig.update_layout(
            xaxis_title="Axe Temporel",
            yaxis_title=dict(text=yaxis_title_text, font=dict(size=13)), 
            height=450,
            margin=dict(l=100, r=20, t=20, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(
                autorange=True, 
                fixedrange=False,
                tickformat=tick_format_type
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # METRIQUES PRUDENTIELLES CONTRAINTES
        st.subheader(f"🛡️ Métriques Prudentielles Risque & Performance ({horizon_text})")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Volatilité Réalisée (Ann.)", f"{volatility:.2f}%")
        rc2.metric("Value-at-Risk (VaR 95% - 1j)", f"{var_95:.2f}%")
        rc3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")

        # --- RAPPORT SYNTHÉTIQUE COMPILÉ SANS TRIPLE GUILLEMETS INFECTÉS ---
        st.info("### 📝 Rapport Synthétique d'Aide à la Décision")
        
        if volatility > 25:
            risk_profile = "fortement spéculatif (Forte variance)."
            action_plan = "L'algorithme XGBoost doit travailler au milieu d'un bruit de marché important."
        else:
            risk_profile = "stable et linéaire."
            action_plan = "La faible variance historique consolide la fiabilité statistique des vecteurs directionnels de l'IA."

        if sharpe_ratio > 1:
            sharpe_text = "excellent. Le rendement compense largement les risques de marché pris."
        elif sharpe_ratio > 0:
            sharpe_text = "favorable, mais l'efficience de la prime de risque reste modérée."
        else:
            sharpe_text = "défavorable (Rendement insuffisant au regard de la volatilité subie)."

        commentary = (
            f"* **Volatilité ({volatility:.2f}%)** : Sur la période choisie ({horizon_text}), l'actif montre un comportement {risk_profile} {action_plan}\n"
            f"* **Value-at-Risk (VaR 95% à {var_95:.2f}%)** : Cet indicateur montre qu'il y a 95% de chances mathématiques que la baisse maximale de l'actif ne dépasse pas **{var_95:.2f}%** sur une seule journée.\n"
            f"* **Ratio de Sharpe ({sharpe_ratio:.2f})** : Ce profil de performance ajustée du risque est jugé {sharpe_text}"
        )
        st.markdown(commentary)

        with st.expander("🔍 Structure Métier - Données Sources BigQuery"):
            st.dataframe(df)

    except Exception as internal_error:
        st.error("❌ Une erreur interne est survenue lors de l'analyse ou du calcul des indicateurs.")
        st.exception(internal_error)