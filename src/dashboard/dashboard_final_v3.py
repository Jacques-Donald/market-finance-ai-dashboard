import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from google.oauth2 import service_account
from google.cloud import bigquery


@st.cache_resource
def get_bq_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials, project=credentials.project_id)

bq_client = get_bq_client()

# --- CONFIGURATION GCP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
credentials_path = os.path.join(root_dir, "credentials.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
#bq_client = bigquery.Client()

#credentials = service_account.Credentials.from_service_account_info(
    #st.secrets["gcp_service_account"]

#bg_client = bigquery.Client(credentials=credentials, project=credentials.project_id)


# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Market Finance AI Dashboard", layout="wide")

st.title("📈 Market Finance AI Dashboard")
st.markdown("### Framework d'Analyse Quantitative & MLOps Institutionnel")

asset_selected = st.selectbox(
    "Sélectionnez un actif de l'univers :", 
    ["S&P 500", "Nasdaq 100", "Bitcoin", "Or", "EUR/USD"]
)

# --- EXTRACTION DE L'HISTORIQUE DEPUIS BIGQUERY ---
query = f"""
    SELECT * FROM `market_finance_enriched.analytics_features_v2` 
    WHERE ticker = '{asset_selected}' 
    ORDER BY date ASC
"""

try:
    with st.spinner("Extraction des données Cloud..."):
        df_history = bq_client.query(query).to_dataframe()
    df_history['date'] = pd.to_datetime(df_history['date'])
except Exception as e:
    st.error(f"❌ Erreur de connexion à BigQuery : {e}")
    df_history = pd.DataFrame()

if not df_history.empty:
    row_latest = df_history.iloc[-1]
    returns = df_history['feature_log_return'].dropna()
    
    # =========================================================================
    # 1. RISK ENGINE & INDICATEURS PRUDENTIELS
    # =========================================================================
    st.markdown("---")
    st.subheader("🛡️ Risk Engine & Indicateurs Prudentiels")
    
    vol_hist_ann = returns.std() * np.sqrt(252) * 100
    var_95_daily = np.percentile(returns, 5) * 100
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        st.metric(label="Dernier Prix de Clôture", value=f"{row_latest['close']:.2f}")
        st.caption(f"Dernier cours spot de clôture enregistré pour le **{asset_selected}**.")
        
    with col_r2:
        st.metric(label="Volatilité Réalisée (Ann.)", value=f"{vol_hist_ann:.2f} %")
        st.caption(f"Mesure l'amplitude des fluctuations de l'actif sur 1 an. Plus elle est élevée, plus le risque est grand.")
        
    with col_r3:
        st.metric(label="VaR Historique 95% (Jour)", value=f"{var_95_daily:.2f} %")
        st.caption(f"Perte quotidienne maximale statistique estimée dans 95% des cas, en conditions normales de marché.")
        
    with col_r4:
        st.metric(label="Ratio de Sharpe", value=f"{sharpe_ratio:.2f}")
        st.caption(f"Performance ajustée au risque. Un ratio supérieur à 1.0 indique une surperformance face au risque pris.")
    
    # =========================================================================
    # 2. ANALYSE DU SENTIMENT (FinBERT)
    # =========================================================================
    st.markdown("---")
    st.subheader("📰 Analyse du Sentiment Global (FinBERT NLP)")
    
    sentiment_score = float(row_latest['feature_news_sentiment'])
    
    if sentiment_score > 0.15:
        sentiment_status = "🟢 HAUSSIER (Bullish)"
        sentiment_desc = "Le flux d'actualités financières récent montre un optimisme marqué. Historiquement favorable à une pression acheteuse à court terme."
    elif sentiment_score < -0.15:
        sentiment_status = "🔴 BAISSIER (Bearish)"
        sentiment_desc = "Les signaux textuels extraits des actualités sont majoritairement négatifs. Risque de pression vendeuse accrue."
    else:
        sentiment_status = "🟡 NEUTRE"
        sentiment_desc = "L'actualité financière n'affiche pas de direction claire. Le marché est en phase d'attente ou d'équilibre."

    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        st.metric(label="Score de Sentiment Enregistré", value=f"{sentiment_score:+.4f}")
    with col_s2:
        st.markdown(f"**Orientation du marché :** `{sentiment_status}`")
        st.markdown(f"**Interprétation :** {sentiment_desc}")

    # =========================================================================
    # 3. PRÉDICTION UNIQUE J+1 (XGBoost) & QUALITÉ ML
    # =========================================================================
    st.markdown("---")
    st.subheader("🔮 Objectif & Prédiction J+1")
    
    asset_clean = asset_selected.replace(" ", "_").replace("&", "").replace("/", "")
    model_path = os.path.join(root_dir, f"models_v2/xgb_{asset_clean}_v2.pkl")
    if not os.path.exists(model_path):
        model_path = f"models_v2/xgb_{asset_clean}_v2.pkl"
        
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        # ALIGNEMENT : Strictement 8 features (exclusion de 'feature_log_return')
        X_cols = [
            'feature_lag_return', 'feature_fedfunds', 'feature_dgs10', 
            'feature_t10y2y', 'feature_walcl', 'feature_stlfsi4', 
            'feature_vixcls', 'feature_news_sentiment'
        ]
        
        input_vector = np.array([[float(row_latest[col]) for col in X_cols]])
        pred_log_return_daily = float(model.predict(input_vector)[0])
        current_price = float(row_latest['close'])
        
        # Calcul du prix cible J+1
        p_j1 = current_price * np.exp(pred_log_return_daily * 1)
        
        col_h1, col_perf1, col_perf2 = st.columns([2, 2, 2])
        
        with col_h1:
            st.metric(label="Prix Cible Target J+1", value=f"{p_j1:.2f}", delta=f"{pred_log_return_daily*100:+.3f}%")
            st.caption("Prix théorique estimé par XGBoost pour la prochaine séance de bourse.")
        
        # Évaluation MLOps historique alignée sur le décalage temporel t+1
        X_hist = df_history[X_cols].fillna(0).values
        
        # Pour l'évaluation historique, la vraie cible est le rendement décalé de -1 (shift -1)
        y_true = df_history['feature_log_return'].shift(-1).fillna(0).values
        y_pred = model.predict(X_hist)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse_score = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        with col_perf1:
            st.metric(label="Précision du Modèle (R²)", value=f"{r2_score:.4f}")
            st.caption("Part de la variance des rendements modélisée avec succès par l'algorithme.")
        with col_perf2:
            st.metric(label="Erreur Résiduelle (RMSE)", value=f"{rmse_score:.6f}")
            st.caption("Écart-type moyen des résidus de prédiction sur les rendements logs.")

        # =========================================================================
        # 4. FILTRES HORIZONS COMPLETS ET GRAPHIQUE ADAPTATIF
        # =========================================================================
        st.markdown("---")
        st.subheader("📈 Graphique Temporel Interactif")
        
        if 'horizon' not in st.session_state:
            st.session_state.horizon = "1 An"
            
        col_b1, col_b2, col_b3, col_b4, col_b5, col_b6, col_b7, col_b8 = st.columns(8)
        if col_b1.button("1 Jour", use_container_width=True): st.session_state.horizon = "1 Jour"
        if col_b2.button("1 Semaine", use_container_width=True): st.session_state.horizon = "1 Semaine"
        if col_b3.button("1 Mois", use_container_width=True): st.session_state.horizon = "1 Mois"
        if col_b4.button("3 Mois", use_container_width=True): st.session_state.horizon = "3 Mois"
        if col_b5.button("6 Mois", use_container_width=True): st.session_state.horizon = "6 Mois"
        if col_b6.button("1 An", use_container_width=True): st.session_state.horizon = "1 An"
        if col_b7.button("5 Ans", use_container_width=True): st.session_state.horizon = "5 Ans"
        if col_b8.button("Max", use_container_width=True): st.session_state.horizon = "Max"
        
        # Reconstruction historique des prédictions (Pour la courbe de validation)
        df_history['price_pred'] = df_history['close'] * np.exp(y_pred)
        # Shift d'affichage de 1 pour aligner la prédiction faite en t avec le prix réalisé en t+1
        df_history['price_pred'] = df_history['price_pred'].shift(1)
        
        if st.session_state.horizon == "1 Jour":
            df_filtered = df_history.iloc[-2:]
        elif st.session_state.horizon == "1 Semaine":
            df_filtered = df_history.iloc[-5:]
        elif st.session_state.horizon == "1 Mois":
            df_filtered = df_history.iloc[-21:]
        elif st.session_state.horizon == "3 Mois":
            df_filtered = df_history.iloc[-63:]
        elif st.session_state.horizon == "6 Mois":
            df_filtered = df_history.iloc[-126:]
        elif st.session_state.horizon == "1 An":
            df_filtered = df_history.iloc[-252:]
        elif st.session_state.horizon == "5 Ans":
            df_filtered = df_history.iloc[-1260:] if len(df_history) >= 1260 else df_history
        else:
            df_filtered = df_history

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['close'],
            mode='lines', name='Cours Réel', line=dict(color='#00CC96', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered['date'], y=df_filtered['price_pred'],
            mode='lines', name='Prédiction (XGBoost J+1)', line=dict(color='#636EFA', dash='dot', width=1.5)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title=f"Cours de Clôture - {asset_selected} (USD)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Filtre temporel actif : **{st.session_state.horizon}**")
        
    else:
        st.error(f"❌ Fichier de modèle introuvable : {model_path}")

# =========================================================================
# 5. AUDIT DES TABLES BIGQUERY
# =========================================================================
    st.markdown("---")
    with st.expander("🔍 Dérouler & Auditer la Table Enrichie BigQuery"):
        st.write(f"Données brutes de la table `market_finance_enriched.analytics_features_v2` pour **{asset_selected}** :")
        st.dataframe(df_history.sort_values(by='date', ascending=False), use_container_width=True)

else:
    st.warning("⚠️ Aucune donnée disponible dans BigQuery pour cet actif.")