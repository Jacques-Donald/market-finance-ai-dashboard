import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from google.cloud import bigquery
import pickle

# Configuration de la clé GCP à la racine
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
bq_client = bigquery.Client()

# --- RECTIFICATION DES FEATURES ---
# On supprime strictement 'feature_log_return' pour empêcher le modèle de tricher.
# On ne garde que le rendement passé (lag) et les indicateurs macro/sentiment.
FEATURE_COLS = [
    'feature_lag_return', 'feature_fedfunds', 'feature_dgs10', 
    'feature_t10y2y', 'feature_walcl', 'feature_stlfsi4', 
    'feature_vixcls', 'feature_news_sentiment'
]

def train_all_assets_v2():
    print("📥 Extraction de la couche analytique BigQuery (market_finance_enriched)...")
    query = "SELECT * FROM `market_finance_enriched.analytics_features_v2` ORDER BY date"
    df = bq_client.query(query).to_dataframe()
    
    assets = df['ticker'].unique()
    
    for asset in assets:
        print(f"\n🎯 BASE MACHINE LEARNING XGBOOST V2 (SANS BIAIS) : {asset.upper()}")
        
        # Isolation des données de l'actif
        df_asset = df[df['ticker'] == asset].copy()
        
        # --- CORRECTION DU DATA LEAKAGE ---
        # On décale les rendements de -1 jour pour créer la vraie cible J+1 :
        # Le rendement de la ligne 't+1' devient la cible de la ligne 't'.
        df_asset['next_day_log_return'] = df_asset['feature_log_return'].shift(-1)
        
        # On supprime la dernière ligne car son J+1 est inconnu (valeur NaN générée par le shift)
        df_asset = df_asset.dropna(subset=['next_day_log_return'])
        
        if len(df_asset) < 300:
            print(f"⚠️ Pas assez de données historiques pour {asset} (< 300 lignes). Ignoré.")
            continue
            
        # Split Chronologique (Séries Temporelles - 80% Train / 20% Test)
        split_idx = int(len(df_asset) * 0.8)
        train_df = df_asset.iloc[:split_idx]
        test_df = df_asset.iloc[split_idx:]
        
        # Application du fillna(0) pour la cohérence avec le Dashboard V3
        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df['next_day_log_return'] # Cible J+1
        
        X_test = test_df[FEATURE_COLS].fillna(0)
        y_test = test_df['next_day_log_return']   # Cible J+1
        
        # Hyperparamètres XGBoost régularisés anti-overfitting
        model = xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.03,
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            objective='reg:squarederror'
        )
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Évaluation sur l'ensemble de validation (le futur non vu par le modèle)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        # Affichage des métriques réelles (adieu les +0.99, bonjour la dure réalité de la finance !)
        print(f"📈 R² Rendements Validation : {r2:+.4f} | MAE Rendements : {mae:.5f}")
        
        # Sérialisation de l'artefact
        os.makedirs("models_v2", exist_ok=True)
        model_name = asset.replace(" ", "_").replace("&", "").replace("/", "")
        file_path = f"models_v2/xgb_{model_name}_v2.pkl"
        
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Modèle sain et sérialisé sous : {file_path}")

if __name__ == "__main__":
    train_all_assets_v2()