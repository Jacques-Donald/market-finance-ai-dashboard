📊 End-to-End Market Predictor : Data Engineering, NLP & MLOps (À but uniquement éducatif, ceci n’est pas un conseil en investissement).
Architecture Data & ML complète et industrialisée : ingestion automatisée de données financières multi-sources (marché + macroéconomie + actualités), analyse du sentiment de marché par NLP (FinBERT), modélisation prédictive par Machine Learning (XGBoost), distribution via une API REST (FastAPI) et restitution sur un Dashboard Streamlit orienté gestion des risques.
Actifs couverts : Bitcoin, EUR/USD, Nasdaq 100, S&P 500, Or.
🔗 Démo en ligne : déployée sur Streamlit Community Cloud (fichier principal : src/dashboard/dashboard_final_v3.py).

🏗️ Architecture globale & flux de données
Le projet est découpé en couches découplées, selon les standards du Data Engineering et du MLOps :
 Marché (prix)  ─┐
 Macroéconomie  ─┼─► RAW ─► ENRICHED (SQL) ─┐
 Actualités     ─┘            │              ├─► ANALYTICS (XGBoost) ─► FastAPI ─► Streamlit
        │                     │              │
        └─► FinBERT (NLP) ─► Sentiment ──────┘
Layer RAW & ENRICHED (Data Engineering) : extraction quotidienne synchrone (marché + macroéconomie) avec écrasement automatique (WRITE_TRUNCATE). Transformations SQL directement sous BigQuery (Forward Fill pour la complétion macroéconomique, alignement temporel strict à J+1).
Layer SENTIMENT (NLP / FinBERT) : collecte des titres d'actualités financières, scoring par FinBERT, agrégation quotidienne par actif et alignement temporel sur les données de marché (voir section dédiée ci-dessous).
Layer ANALYTICS (Machine Learning) : extraction des features (techniques, macro et sentiment) et entraînement isolé par actif d'un XGBoost Regressor. Validation par split temporel (80 % train chronologique / 20 % test) pour interdire tout data leakage, puis injection des inférences en base.
Layer RESTITUTION (Application) : consommation des prédictions BigQuery via FastAPI (port 8006), couplée à un dashboard Streamlit doté d'un Risk Engine temps réel (volatilité annualisée, Value-at-Risk 95 %, ratio de Sharpe) et d'un indicateur de sentiment de marché.
Pour le détail des choix d'architecture, de la conformité RGPD et des schémas de données, voir ARCHITECTURE.md.



🧠 Sentiment de marché avec FinBERT
FinBERT est un modèle BERT affiné sur du texte financier. Il classe chaque texte en positif / négatif / neutre et fournit une probabilité pour chaque classe.
Principe

Lecture du score


⚠️ Limites : FinBERT est entraîné sur de l'anglais financier ; le sentiment d'un titre n'est pas une prévision en soi et reste une feature parmi d'autres. À ré-évaluer régulièrement (gain réel en MAE/RMSE avec vs sans sentiment).


## 📂 Structure du Répertoire

market-finance-ai-dashboard/
├── .github/workflows/ci.yml        # CI (tests)
├── .streamlit/secrets.toml         # secrets locaux (NON versionné)
├── data/
├── models_v2/                      # modèles XGBoost entraînés (.pkl, un par actif)
├── src/
│   ├── api/                        # FastAPI (main_api.py, main_api_v2.py)
│   ├── dashboard/                  # Streamlit (dashboard_final_v3.py = fichier déployé)
│   ├── database/                   # ingestion, feature engineering, alignement SQL
│   └── models/                     # entraînement (train_predict_v2.py)
├── tests/test_api.py
├── .env.example
├── .gitignore
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
└── VEILLE_ET_BENCHMARK.md
