import sys
import os

# --- INITIALISATION DE L'ENVIRONNEMENT SYSTÈME ---
# Calcule dynamiquement la racine du projet, peu importe l'outil qui lance le script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
import pandas as pd

app = FastAPI(
    title="API de Prévisions Financières IA", 
    description="Back-end de distribution des données analytiques pour le projet de certification"
)

PROJECT_ID = "bigquery-finance-ia"
DATASET_ANALYTICS = "market_finance_analytics"

# ==============================================================================
# CONFIGURATION DE L'AUTHENTIFICATION DIRECTE DÉCOUPÉE
# ==============================================================================
# Recherche de la clé JSON à la racine réelle du projet
path_to_json = os.path.join(project_root, "google_credentials.json")

if os.path.exists(path_to_json):
    # On force BigQuery à charger DIRECTEMENT la clé locale
    global_bq_client = bigquery.Client.from_service_account_json(path_to_json, project=PROJECT_ID)
    print("✓ API : Client BigQuery initialisé en DIRECT via clé JSON.")
else:
    print(f"⚠ API : Fichier JSON introuvable au chemin {path_to_json}. Repli environnement...")
    # Secours automatique pour l'environnement de test CI/CD ou Cloud Run
    global_bq_client = bigquery.Client(project=PROJECT_ID)


# ==============================================================================
# ROUTES DE L'API REST
# ==============================================================================
@app.get("/")
def read_root():
    return {
        "status": "online", 
        "project": "Market Finance AI",
        "auth_configured": os.path.exists(path_to_json)
    }

@app.get("/predictions/{ticker}")
def get_predictions(ticker: str):
    """Expose les prédictions d'un actif au format JSON pour le Front-end."""
    
    # Requête SQL sécurisée et alignée sur la couche Analytics de l'architecture
    query = f"""
        SELECT date, ticker, price, predicted_next_day_price, model_r2, model_mae
        FROM `{PROJECT_ID}.{DATASET_ANALYTICS}.market_predictions`
        WHERE ticker = @ticker
        ORDER BY date ASC
    """
    
    # Utilisation de paramètres typés pour immuniser l'API contre les injections SQL
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
    )
    
    try:
        # Exécution de la requête et conversion en DataFrame
        df = global_bq_client.query(query, job_config=job_config).to_dataframe()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Actif '{ticker}' introuvable dans la base analytics.")
        
        # Sécurisation du formatage pour une sérialisation JSON stable sans NaN/Null distants
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['price'] = df['price'].astype(float)
        df['predicted_next_day_price'] = df['predicted_next_day_price'].astype(float)
        df['model_r2'] = df['model_r2'].astype(float)
        df['model_mae'] = df['model_mae'].astype(float)
        
        return df.to_dict(orient="records")
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        # Capture propre des exceptions pour éviter de crash l'API en production
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")


# ==============================================================================
# LANCEUR REST
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage de l'API en mode Production Stable sur le port 8005...")
    # reload=False empêche les boucles de rechargement infinies causées par le cache Streamlit
    uvicorn.run("src.api.main_api:app", host="127.0.0.1", port=8006, reload=False)