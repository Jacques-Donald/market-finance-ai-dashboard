import os
from google.cloud import bigquery

# --- CONFIGURATION GCP DYNAMIQUE ET SÉCURISÉE ---
# On localise le dossier où se trouve ce script (src/database)
current_dir = os.path.dirname(os.path.abspath(__file__))

# CORRECTION CRUCIALE : On remonte de 2 niveaux pour sortir de 'database' ET de 'src'
root_dir = os.path.dirname(os.path.dirname(current_dir))

# On construit le chemin vers le fichier credentials.json stocké à la racine du projet
credentials_path = os.path.join(root_dir, "credentials.json")

print(f"🔍 Tentative de chargement de la clé GCP à l'adresse : {credentials_path}")

# Injection du chemin absolu dans l'environnement système de l'application
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Initialisation sécurisée du client BigQuery
bq_client = bigquery.Client()

# --- LA REQUÊTE SQL COMPLÈTE DE DATA ENGINEERING ---
sql_query = """
CREATE OR REPLACE TABLE `market_finance_enriched.analytics_features_v2` AS
WITH market_calc AS (
  SELECT 
    date, ticker, close, volume,
    LN(close / LAG(close, 1) OVER(PARTITION BY ticker ORDER BY date)) AS log_return,
    LN(LAG(close, 1) OVER(PARTITION BY ticker ORDER BY date) / LAG(close, 2) OVER(PARTITION BY ticker ORDER BY date)) AS lag_return,
    LN(LEAD(close, 1) OVER(PARTITION BY ticker ORDER BY date) / close) AS target_log_return
  FROM `market_finance_enriched.raw_market`
),
macro_pivot AS (
  SELECT 
    PARSE_DATE('%Y-%m-%d', date) AS date,
    MAX(CASE WHEN series_id = 'FEDFUNDS' THEN value END) AS fedfunds,
    MAX(CASE WHEN series_id = 'DGS10' THEN value END) AS dgs10,
    MAX(CASE WHEN series_id = 'T10Y2Y' THEN value END) AS t10y2y,
    MAX(CASE WHEN series_id = 'WALCL' THEN value END) AS walcl,
    MAX(CASE WHEN series_id = 'STLFSI4' THEN value END) AS stlfsi4,
    MAX(CASE WHEN series_id = 'VIXCLS' THEN value END) AS vixcls,
    MAX(CASE WHEN series_id = 'NEWS_SENTIMENT' THEN value END) AS news_sentiment
  FROM `market_finance_enriched.raw_macro`
  GROUP BY date
),
macro_cleaned AS (
  SELECT
    d.date,
    LAST_VALUE(m.fedfunds IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS fedfunds,
    LAST_VALUE(m.dgs10 IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS dgs10,
    LAST_VALUE(m.t10y2y IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS t10y2y,
    LAST_VALUE(m.walcl IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS walcl,
    LAST_VALUE(m.stlfsi4 IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS stlfsi4,
    LAST_VALUE(m.vixcls IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS vixcls,
    COALESCE(LAST_VALUE(m.news_sentiment IGNORE NULLS) OVER(ORDER BY d.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0.0) AS news_sentiment
  FROM (SELECT DISTINCT EXTRACT(DATE FROM date) AS date FROM market_calc) d
  LEFT JOIN macro_pivot m ON d.date = m.date
)
SELECT 
  m.date, m.ticker, m.close, m.volume,
  COALESCE(m.log_return, 0.0) AS feature_log_return,
  COALESCE(m.lag_return, 0.0) AS feature_lag_return,
  COALESCE(c.fedfunds, 0.0) AS feature_fedfunds,
  COALESCE(c.dgs10, 0.0) AS feature_dgs10,
  COALESCE(c.t10y2y, 0.0) AS feature_t10y2y,
  COALESCE(c.walcl, 0.0) AS feature_walcl,
  COALESCE(c.stlfsi4, 0.0) AS feature_stlfsi4,
  COALESCE(c.vixcls, 0.0) AS feature_vixcls,
  c.news_sentiment AS feature_news_sentiment,
  m.target_log_return AS target
FROM market_calc m
JOIN macro_cleaned c ON EXTRACT(DATE FROM m.date) = c.date
WHERE m.log_return IS NOT NULL;
"""

if __name__ == "__main__":
    print("⏳ Envoi de la requête de Data Engineering à Google BigQuery...")
    query_job = bq_client.query(sql_query)
    query_job.result()  # Bloque le terminal jusqu'à ce que BigQuery finisse l'agrégation
    print("✅ Table 'analytics_features_v2' créée avec succès dans BigQuery via le terminal !")