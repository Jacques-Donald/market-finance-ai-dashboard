import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# ==============================================================================
# CONFIGURATION, SÉCURITÉ & ARBORESCENCE BIGQUERY
# ==============================================================================
PROJECT_ID = "bigquery-finance-ia"
LOCATION = "US"

DATASET_RAW = "market_finance_raw"
DATASET_ENRICHED = "market_finance_enriched"
DATASET_ANALYTICS = "market_finance_analytics"

# Les 3 couches de l'architecture indispensables au projet
ALL_DATASETS = [DATASET_RAW, DATASET_ENRICHED, DATASET_ANALYTICS]

# Périmètre des actifs à collecter pour le Dashboard et l'IA
TICKERS = ["^GSPC", "^IXIC", "BTC-USD", "GC=F", "EURUSD=X"]

# --- DÉTECTION AUTOMATIQUE DE LA CLÉ JSON (Système Robuste) ---
current_path = os.path.abspath(__file__)
project_root = current_path
while project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)
    if "market-finance-ai-dashboard" in os.path.basename(project_root):
        break

path_to_json = os.path.join(project_root, "google_credentials.json")

if os.path.exists(path_to_json):
    print(f"✓ Clé d'authentification détectée : {path_to_json}")
    client = bigquery.Client.from_service_account_json(path_to_json, project=PROJECT_ID)
    print(f"📧 Compte configuré : {client.project}")
else:
    print(f"⚠ Clé introuvable au chemin {path_to_json}. Tentative de repli système...")
    client = bigquery.Client(project=PROJECT_ID)

# Fenêtre temporelle : 1 an d'historique glissant
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# ==============================================================================
# FONCTIONS DE GESTION DE L'INFRASTRUCTURE ET COLLECTE
# ==============================================================================
def ensure_architecture_exists():
    """Scanne Google Cloud et s'assure que les 3 couches d'architecture existent."""
    print("\n--- Vérification et initialisation de l'architecture des Datasets ---")
    for dataset_name in ALL_DATASETS:
        dataset_ref = bigquery.DatasetReference(PROJECT_ID, dataset_name)
        try:
            client.get_dataset(dataset_ref)
            print(f"✓ Dataset '{dataset_name}' en ligne.")
        except NotFound:
            print(f"⚠ Dataset '{dataset_name}' absent. Création dans le cloud...")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = LOCATION
            client.create_dataset(dataset)
            print(f"✓ Dataset '{dataset_name}' créé avec succès ({LOCATION}).")

def fetch_all_yahoo_data():
    """Télécharge et unifie l'historique de l'ensemble des Tickers configurés."""
    print("\n--- Démarrage de la collecte Yahoo Finance ---")
    all_dfs = []
    for ticker in TICKERS:
        print(f"Téléchargement en cours pour : {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if df.empty: 
                continue
            df = df.reset_index()
            # Nettoyage anti-bug pour les MultiIndex générés par yfinance
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = df[['Date', 'Close', 'Volume']].rename(columns={'Date': 'date', 'Close': 'price', 'Volume': 'volume'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['ticker'] = ticker
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Erreur lors de la collecte de {ticker}: {e}")
            
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def fetch_fred_data():
    """Récupère l'ensemble des indicateurs macroéconomiques d'environnement de marché."""
    print("\n--- Démarrage de la collecte FRED (Macroéconomie) ---")
    metrics = {
        'FEDFUNDS': 'fed_rate',
        'DGS10': 'treasury_10y',
        'T10Y2Y': 'yield_curve_spread',
        'WALCL': 'fed_balance_sheet',
        'STLFSI4': 'financial_stress_index',
        'VIXCLS': 'vix'
    }
    try:
        df_macro = web.DataReader(list(metrics.keys()), 'fred', start_date, end_date)
        df_macro = df_macro.reset_index().rename(columns={'DATE': 'date'})
        df_macro = df_macro.rename(columns=metrics)
        df_macro['date'] = pd.to_datetime(df_macro['date']).dt.date
        return df_macro
    except Exception as e:
        print(f"❌ Erreur lors de la collecte des données FRED : {e}")
        return pd.DataFrame()

# ==============================================================================
# EXÉCUTION DU PIPELINE D'INGESTION RAW
# ==============================================================================
def main():
    # 1. Sécurité de l'infrastructure
    ensure_architecture_exists()

    # Configuration d'écriture : Remplace proprement les tables existantes
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

    # 2. Ingestion de la table de Marché
    df_market = fetch_all_yahoo_data()
    if not df_market.empty:
        table_ref = f"{PROJECT_ID}.{DATASET_RAW}.raw_market"
        print(f"Chargement de 'raw_market' ({len(df_market)} lignes) dans BigQuery...")
        client.load_table_from_dataframe(df_market, table_ref, job_config=job_config).result()
        print("✓ Table 'raw_market' alimentée avec succès pour les 5 actifs.")

    # 3. Ingestion de la table Macroéconomique
    df_macro = fetch_fred_data()
    if not df_macro.empty:
        table_ref = f"{PROJECT_ID}.{DATASET_RAW}.raw_macro"
        print(f"Chargement de 'raw_macro' ({len(df_macro)} lignes) dans BigQuery...")
        client.load_table_from_dataframe(df_macro, table_ref, job_config=job_config).result()
        print("✓ Table 'raw_macro' alimentée avec succès.")

if __name__ == "__main__":
    main()