import os
from google.cloud import bigquery

PROJECT_ID = "bigquery-finance-ia"
LOCATION = "US"
DATASET_RAW = "market_finance_raw"
DATASET_ENRICHED = "market_finance_enriched"

def run_data_engineering():
    """Aligne la macroéconomie sur chaque actif et génère la variable cible (Target)."""
    print("🚀 Initialisation du module Data Engineering...")
    
    current_path = os.path.abspath(__file__)
    project_root = current_path
    while project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)
        if "market-finance-ai-dashboard" in os.path.basename(project_root):
            break

    path_to_json = os.path.join(project_root, "google_credentials.json")

    if os.path.exists(path_to_json):
        client = bigquery.Client.from_service_account_json(path_to_json, project=PROJECT_ID)
        print("✓ Client BigQuery initialisé avec succès via la clé JSON.")
    else:
        print(f"⚠ Clé introuvable au chemin {path_to_json}. Tentative de repli système...")
        client = bigquery.Client(project=PROJECT_ID)
        
    print("⚡ Exécution du Data Engineering SQL sur BigQuery...")
    
    query = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ENRICHED}.features_market_ready` AS
    WITH dates_base AS (
      SELECT 
        CAST(date AS DATE) as date, 
        CAST(ticker AS STRING) as ticker, 
        CAST(price AS FLOAT64) as price, 
        CAST(volume AS FLOAT64) as volume 
      FROM `{PROJECT_ID}.{DATASET_RAW}.raw_market`
    ),
    macro_filled AS (
      SELECT 
        d.date,
        d.ticker,
        LAST_VALUE(CAST(m.fed_rate AS FLOAT64) IGNORE NULLS) OVER(PARTITION BY d.ticker ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as fed_rate,
        LAST_VALUE(CAST(m.treasury_10y AS FLOAT64) IGNORE NULLS) OVER(PARTITION BY d.ticker ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as treasury_10y,
        LAST_VALUE(CAST(m.yield_curve_spread AS FLOAT64) IGNORE NULLS) OVER(PARTITION BY d.ticker ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as yield_curve_spread,
        LAST_VALUE(CAST(m.fed_balance_sheet AS FLOAT64) IGNORE NULLS) OVER(PARTITION BY d.ticker ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as fed_balance_sheet,
        LAST_VALUE(CAST(m.financial_stress_index AS FLOAT64) IGNORE NULLS) OVER(PARTITION BY d.ticker ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as financial_stress_index,
        LAST_VALUE(CAST(m.vix AS FLOAT64) IGNORE NULLS) OVER(PARTITION BY d.ticker ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as vix
      FROM dates_base d
      LEFT JOIN `{PROJECT_ID}.{DATASET_RAW}.raw_macro` m ON d.date = CAST(m.date AS DATE)
    )
    SELECT 
      mkt.date, mkt.ticker, mkt.price, mkt.volume,
      mac.fed_rate, mac.treasury_10y, mac.yield_curve_spread, mac.fed_balance_sheet, mac.financial_stress_index, mac.vix,
      LEAD(mkt.price, 1) OVER(PARTITION BY mkt.ticker ORDER BY mkt.date) as target_next_day_price
    FROM dates_base mkt
    JOIN macro_filled mac ON mkt.date = mac.date AND mkt.ticker = mac.ticker;
    """
    
    try:
        query_job = client.query(query)
        query_job.result()  
        print("✓ Couche ENRICHED mise à jour avec succès dans BigQuery (Table: features_market_ready).")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution du Data Engineering SQL : {e}")

if __name__ == "__main__":
    run_data_engineering()