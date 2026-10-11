from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="MLOps Predictive Multi-Asset API - V2 Stationnaire", version="2.0")
security = HTTPBearer()

class InferenceRequest(BaseModel):
    ticker: str
    current_price: float
    feature_log_return: float
    feature_lag_return: float
    feature_fedfunds: float
    feature_dgs10: float
    feature_t10y2y: float
    feature_walcl: float
    feature_stlfsi4: float
    feature_vixcls: float
    feature_news_sentiment: float

class InferenceResponse(BaseModel):
    ticker: str
    predicted_log_return: float
    predicted_next_price: float

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "secret_mlops_token_rncp6":
        raise HTTPException(status_code=403, detail="Accès non autorisé")
    return token

@app.post("/predict", response_model=InferenceResponse)
def get_prediction(req: InferenceRequest, token: str = Depends(verify_token)):
    asset_clean = req.ticker.replace(" ", "_").replace("&", "").replace("/", "")
    model_path = f"models_v2/xgb_{asset_clean}_v2.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Modèle introuvable : {model_path}")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    input_vector = np.array([[
        req.feature_log_return, req.feature_lag_return, req.feature_fedfunds,
        req.feature_dgs10, req.feature_t10y2y, req.feature_walcl,
        req.feature_stlfsi4, req.feature_vixcls, req.feature_news_sentiment
    ]])
    
    pred_return = float(model.predict(input_vector)[0])
    predicted_price = float(req.current_price * np.exp(pred_return))
    
    return InferenceResponse(
        ticker=req.ticker,
        predicted_log_return=pred_return,
        predicted_next_price=predicted_price
    )