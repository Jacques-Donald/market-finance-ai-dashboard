import os
import streamlit as st
import pandas as pd
import requests
from google.cloud import bigquery

# --- CONFIGURATION GCP DYNAMIQUE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
credentials_path = os.path.join(root_dir, "credentials.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
bq_client = bigquery.Client()

API_URL = "http://10.231.220.234:8080/predict"
HEADERS = {"Authorization": "Bearer secret_mlops_token_rncp6"}

# --- INTERFACE GRAPHIQUE STREAMLIT ---
st.set_page_config(page_title="Dashboard Prédictif V2", layout="wide")
st.title("📈 Écosystème Analytics Cloud & MLOps Financier (Version 2)")

asset_selected = st.selectbox(
    "Sélectionnez un actif de l'univers :", 
    ["S&P 500", "Nasdaq 100", "Bitcoin", "Or", "EUR/USD"]
)

query = f"SELECT * FROM `market_finance_enriched.analytics_features_v2` WHERE ticker = '{asset_selected}' ORDER BY date DESC LIMIT 1"

try:
    df_latest = bq_client.query(query).to_dataframe()
except Exception as e:
    st.error(f"❌ Erreur de connexion à BigQuery : {e}")
    df_latest = pd.DataFrame()

if not df_latest.empty:
    row = df_latest.iloc[0]
    st.subheader("⚓ Indicateurs de Risque Prudentiels & Sentiment Numérique")
    col1, col2, col3 = st.columns(3)
    vol_annualized = float(row['feature_log_return']) * (252 ** 0.5) * 100
    col1.metric(label="Dernier Prix de Clôture", value=f"{row['close']:.2f}")
    col2.metric(label="Volatilité Réalisée Instantanée", value=f"{vol_annualized:+.3f} %")
    col3.metric(label="Sentiment de Marché (FinBERT NLP)", value=f"{row['feature_news_sentiment']:.4f}")

    payload = {
        "ticker": str(asset_selected),
        "current_price": float(row['close']),
        "feature_log_return": float(row['feature_log_return']),
        "feature_lag_return": float(row['feature_lag_return']),
        "feature_fedfunds": float(row['feature_fedfunds']),
        "feature_dgs10": float(row['feature_dgs10']),
        "feature_t10y2y": float(row['feature_t10y2y']),
        "feature_walcl": float(row['feature_walcl']),
        "feature_stlfsi4": float(row['feature_stlfsi4']),
        "feature_vixcls": float(row['feature_vixcls']),
        "feature_news_sentiment": float(row['feature_news_sentiment'])
    }
    
    st.markdown("---")
    st.subheader("🔮 Prédiction Automatique Moteur ML (Horizon J+1)")
    
    try:
        res = requests.post(API_URL, json=payload, headers=HEADERS, timeout=5)
        if res.status_code == 200:
            data = res.json()
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label="Prix Théorique Reconstruit (J+1)", value=f"{data['predicted_next_price']:.2f}")
            col_res2.metric(label="Rendement Logarithmique Attendu", value=f"{data['predicted_log_return']:.6f}")
        else:
            st.error(f"Erreur API ({res.status_code}) : {res.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de joindre l'API FastAPI. Elle n'est pas lancée sur le port 8000.")
else:
    st.warning("⚠️ Aucune observation BigQuery disponible pour cet actif.")