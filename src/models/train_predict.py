import os
import sys
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION ET ALIGNEMENT DE L'INFRASTRUCTURE BIGQUERY
# ==============================================================================
PROJECT_ID = "bigquery-finance-ia"
DATASET_ENRICHED = "market_finance_enriched"
DATASET_ANALYTICS = "market_finance_analytics"

# --- SYSTEME DE DETECTION DE RACINE IDENTIQUE AUX AUTRES FICHIERS ---
current_path = os.path.abspath(__file__)
project_root = current_path
while project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)
    if "market-finance-ai-dashboard" in os.path.basename(project_root):
        break

if project_root not in sys.path:
    sys.path.append(project_root)

def get_secure_bigquery_client():
    """Initialise le client BigQuery en forçant le chemin absolu calculé dynamiquement."""
    path_to_json = os.path.join(project_root, "google_credentials.json")

    if os.path.exists(path_to_json):
        print(f"✓ ML Engine : Authentification sécurisée via {path_to_json}")
        return bigquery.Client.from_service_account_json(path_to_json, project=PROJECT_ID)
    else:
        print(f"⚠ ML Engine : Clé JSON introuvable à l'adresse {path_to_json}. Tentative via ADC/Repli...")
        return bigquery.Client(project=PROJECT_ID)

def train_and_evaluate():
    print("🧠 Initialisation de l'entraînement Machine Learning...")
    
    try:
        client = get_secure_bigquery_client()
    except Exception as auth_err:
        print(f"❌ Erreur critique d'initialisation BigQuery : {auth_err}")
        return

    # 1. Chargement des données propres depuis la couche ENRICHED
    print("📥 Extraction des features depuis BigQuery...")
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ENRICHED}.features_market_ready` ORDER BY date ASC"
    
    try:
        df_all = client.query(query).to_dataframe()
    except Exception as query_err:
        print(f"❌ Impossible de lire la table source 'features_market_ready' : {query_err}")
        print("Vérifiez que vos scripts d'ingestion/ETL ont bien créé cette table dans BigQuery.")
        return

    if df_all.empty:
        print("❌ La table extraite de BigQuery est vide. Fin du processus.")
        return

    # Liste officielle des features macro-économiques et de marché
    features = ['price', 'volume', 'fed_rate', 'treasury_10y', 'yield_curve_spread', 'fed_balance_sheet', 'financial_stress_index', 'vix']
    final_dfs = []
    
    # Entraînement isolé par actif pour éviter le mélange des patterns
    tickers_presents = df_all['ticker'].unique()
    print(f"📈 Actifs détectés dans le dataset : {list(tickers_presents)}")

    for ticker in tickers_presents:
        print(f"\nModélisation pour l'actif : {ticker}")
        df = df_all[df_all['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        
        # Séparation Train complet et Ligne future (Jour J qui n'a pas encore de prix J+1)
        df_train_set = df[df['target_next_day_price'].notnull()]
        if len(df_train_set) < 20: 
            print(f"  ⚠ Pas assez de données historiques valides pour {ticker} (minimum requis : 20 lignes).")
            continue
        
        X = df_train_set[features]
        y = df_train_set['target_next_day_price']
        
        # --- SPLIT TEMPOREL STRUCTURÉ (Validation rigoureuse pour le jury) ---
        split_idx = int(len(df_train_set) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Initialisation et entraînement du Regressor XGBoost
        model = XGBRegressor(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.05, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # --- ÉVALUATION DES PERFORMANCES ---
        preds_test = model.predict(X_test)
        r2 = r2_score(y_test, preds_test)
        mae = mean_absolute_error(y_test, preds_test)
        
        # Formatage dynamique de l'erreur selon la nature de l'actif (Forex vs Crypto/Actions)
        is_forex = (ticker == "EURUSD=X")
        error_symbol = "" if is_forex else "$"
        precision_format = ".4f" if is_forex else ".2f"
        
        print(f"  -> R² Score (Précision) : {r2 * 100:.2f}%")
        print(f"  -> MAE (Erreur moyenne) : {error_symbol}{mae:{precision_format}}")
        
        # Prédiction globale (Historique + Ligne future sans target)
        df['predicted_next_day_price'] = model.predict(df[features])
        
        # Ajout des métriques du modèle pour l'affichage direct sur le Dashboard
        df['model_r2'] = r2
        df['model_mae'] = mae
        
        # Extraction de la structure finale attendue par l'API et le Dashboard
        final_dfs.append(df[['date', 'ticker', 'price', 'predicted_next_day_price', 'model_r2', 'model_mae']])
        
    # 2. Sauvegarde et mise à jour de la couche ANALYTICS
    if final_dfs:
        df_predictions_all = pd.concat(final_dfs, ignore_index=True)
        
        # SÉCURISATION DU TYPE DATE POUR L'ALIMENTATION EN BASE
        # On force Pandas à typer la colonne en datetime. BigQuery la convertira automatiquement en type DATE/TIMESTAMP.
        df_predictions_all['date'] = pd.to_datetime(df_predictions_all['date'])
            
        table_ref = f"{PROJECT_ID}.{DATASET_ANALYTICS}.market_predictions"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        
        print(f"\n📤 Écriture de {len(df_predictions_all)} lignes vers la table : {table_ref}...")
        try:
            client.load_table_from_dataframe(df_predictions_all, table_ref, job_config=job_config).result()
            print("✓ Prédictions injectées avec succès dans la couche ANALYTICS (Format Temporel Préservé).")
            print("🚀 Pipeline MLOps opérationnel à 100% !")
        except Exception as upload_err:
            print(f"❌ Erreur lors du chargement des données vers BigQuery : {upload_err}")
    else:
        print("\n❌ Aucun actif n'a pu être modélisé. Table 'market_predictions' non mise à jour.")

if __name__ == "__main__":
    train_and_evaluate()