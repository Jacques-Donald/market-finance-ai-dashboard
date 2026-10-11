import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from google.cloud import bigquery
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# --- CONFIGURATION GCP DYNAMIQUE ET SÉCURISÉE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
credentials_path = os.path.join(root_dir, "credentials.json")

print(f"🔍 Tentative de chargement de la clé GCP à l'adresse : {credentials_path}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Initialisation sécurisée du client BigQuery
bq_client = bigquery.Client()
DATASET_ID = "market_finance_enriched"

# --- PARAMÈTRES DU PÉRIMÈTRE COMPLETS ---
TICKERS_MAP = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^IXIC",
    "Bitcoin": "BTC-USD",
    "Or": "GC=F",
    "EUR/USD": "EURUSD=X"
}

FRED_SERIES = ["FEDFUNDS", "DGS10", "T10Y2Y", "WALCL", "STLFSI4", "VIXCLS"]

# Initialisation FinBERT (NLP Sentiment)
TOKENIZER = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MODEL = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def fetch_all_market_data(start_date="2010-01-01"):
    """Extraction de l'intégralité des 5 actifs avec gestion robuste des MultiIndex yfinance"""
    print("🚀 Ingestion de la couche marché (5 actifs)...")
    df_list = []
    for name, ticker in TICKERS_MAP.items():
        try:
            raw_df = yf.download(ticker, start=start_date, group_by="column")
            if not raw_df.empty:
                raw_df = raw_df.reset_index()
                
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = [col[0] if col[0] != "" else col[1] for col in raw_df.columns]
                
                raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]
                
                target_col = None
                for candidate in ['adj close', 'close']:
                    if candidate in raw_df.columns:
                        target_col = candidate
                        break
                
                if target_col is None:
                    print(f"❌ Impossible de trouver la colonne de prix pour {name}")
                    continue
                
                df = pd.DataFrame()
                df['date'] = pd.to_datetime(raw_df['date'])
                df['ticker'] = name
                df['close'] = raw_df[target_col].astype(float)
                df['volume'] = raw_df['volume'].astype(float).fillna(0) if 'volume' in raw_df.columns else 0.0
                
                df_list.append(df)
                print(f"✅ {name} ({ticker}) extrait avec succès.")
        except Exception as e:
            print(f"❌ Échec d'extraction pour {name} : {e}")
            
    if not df_list:
        raise ValueError("❌ Aucun actif n'a pu être téléchargé via yfinance.")
        
    return pd.concat(df_list, ignore_index=True)

def fetch_all_fred_macro(api_key, start_date="2010-01-01"):
    """Extraction complète des 6 métriques macroéconomiques de la FRED"""
    print("🚀 Ingestion de la couche macroéconomique (6 indicateurs)...")
    df_list = []
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    for series in FRED_SERIES:
        params = {
            "series_id": series,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date
        }
        for attempt in range(3):
            try:
                res = requests.get(base_url, params=params, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    obs = data.get("observations", [])
                    
                    dates = [o["date"] for o in obs]
                    values = [float(o["value"]) if o["value"] != "." else np.nan for o in obs]
                    
                    df = pd.DataFrame({"date": dates, "series_id": series, "value": values})
                    df = df.dropna()
                    df_list.append(df)
                    print(f"✅ Série FRED {series} ingérée.")
                    break
                elif res.status_code == 429:
                    time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    print(f"❌ Échec critique sur la série {series} : {e}")
                time.sleep(2 ** attempt)
                
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def scrape_news_sentiment():
    """Scraping et analyse NLP du sentiment de marché du jour (CNBC Business)"""
    print("🚀 Extraction des signaux faibles (Web Scraping + FinBERT)...")
    url = "https://www.cnbc.com/id/10001147/device/rss/rss.html"
    headers = {"User-Agent": "Mozilla/5.0 MLOpsAcademicPipeline/2.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, features="xml")
        headlines = [item.title.text for item in soup.find_all("item") if item.title][:30]
        
        if not headlines:
            return 0.0
            
        inputs = TOKENIZER(headlines, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = MODEL(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        pos_score = predictions[:, 0].mean().item()
        neg_score = predictions[:, 1].mean().item()
        return float(np.clip(pos_score - neg_score, -1.0, 1.0))
    except Exception as e:
        print(f"⚠️ Dégradation gracieuse du sentiment (Neutre par défaut) : {e}")
        return 0.0

def save_to_bigquery(df, table_name):
    table_ref = bq_client.dataset(DATASET_ID).table(table_name)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()

if __name__ == "__main__":
    FRED_API_KEY = os.environ.get("FRED_API_KEY")
    
    market_df = fetch_all_market_data()
    save_to_bigquery(market_df, "raw_market")
    
    macro_df = fetch_all_fred_macro(FRED_API_KEY)
    
    sentiment_val = scrape_news_sentiment()
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    sentiment_row = pd.DataFrame([{"date": today_str, "series_id": "NEWS_SENTIMENT", "value": sentiment_val}])
    
    if not macro_df.empty:
        macro_final = pd.concat([macro_df, sentiment_row], ignore_index=True)
    else:
        macro_final = sentiment_row
        
    save_to_bigquery(macro_final, "raw_macro")
    print("🎉 Pipeline d'ingestion global exécuté sur l'intégralité du périmètre.")