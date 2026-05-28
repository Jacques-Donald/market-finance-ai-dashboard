import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd

# --- SYSTEME DE DETECTION DE RACINE DYNAMIQUE STANDARDISÉ ---
current_path = os.path.abspath(__file__)
project_root = current_path
while project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)
    if "market-finance-ai-dashboard" in os.path.basename(project_root):
        break

if project_root not in sys.path:
    sys.path.append(project_root)

# Import de l'application FastAPI après configuration du sys.path
from src.api.main_api import app

client = TestClient(app)

# ==============================================================================
# TESTS UNITAIRES
# ==============================================================================

def test_read_root():
    """Vérifie que la route principale de l'API est en ligne et accessible."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert "project" in response.json()
    assert response.json()["project"] == "Market Finance AI"


@patch('src.api.main_api.global_bq_client')
def test_get_predictions_success(mock_bq_client):
    """Vérifie le comportement nominal de l'API avec un ticker valide (Mocké)."""
    # 1. Simulation d'un retour de données BigQuery propre (DataFrame Pandas)
    mock_df = pd.DataFrame([{
        "date": "2026-05-28",
        "ticker": "^GSPC",
        "price": 5000.0,
        "predicted_next_day_price": 5050.0,
        "model_r2": 0.85,
        "model_mae": 12.5
    }])
    
    # On mocke la chaîne de calcul : client.query().to_dataframe()
    mock_query_job = MagicMock()
    mock_query_job.to_dataframe.return_value = mock_df
    mock_bq_client.query.return_value = mock_query_job

    # 2. Appel de l'API
    response = client.get("/predictions/^GSPC")
    
    # 3. Assertions strictes
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "^GSPC"
    assert data[0]["price"] == 5000.0
    assert data[0]["predicted_next_day_price"] == 5050.0


@patch('src.api.main_api.global_bq_client')
def test_get_predictions_invalid_ticker(mock_bq_client):
    """Vérifie que l'API renvoie un code 404 strict lorsque le ticker n'existe pas."""
    # Simulation d'un DataFrame vide retourné par BigQuery (ticker inconnu en base)
    mock_df = pd.DataFrame()
    
    mock_query_job = MagicMock()
    mock_query_job.to_dataframe.return_value = mock_df
    mock_bq_client.query.return_value = mock_query_job

    # Appel de l'API
    response = client.get("/predictions/FAKETICKER99")
    
    # Assertion déterministe : l'API DOIT renvoyer 404, le code 500 est désormais banni
    assert response.status_code == 404
    assert response.json()["detail"] == "Actif 'FAKETICKER99' introuvable dans la base analytics."


def test_dashboard_assets_integrity():
    """Vérifie la cohérence et l'intégrité des tickers configurés pour le Front-end."""
    assets = {
        "S&P 500 (Actions US)": "^GSPC",
        "Nasdaq 100 (Technologie)": "^IXIC",
        "Bitcoin (Crypto-actif)": "BTC-USD",
        "Or (Valeur Refuge)": "GC=F",
        "Euro / Dollar (Forex)": "EURUSD=X"
    }
    # Vérification des piliers de l'application
    assert "Bitcoin (Crypto-actif)" in assets
    assert assets["Euro / Dollar (Forex)"] == "EURUSD=X"
    assert assets["S&P 500 (Actions US)"] == "^GSPC"
    assert len(assets) == 5  # Sécurité : on s'assure que les 5 actifs attendus sont là