# 📊 End-to-End Market Predictor: Pipeline Data Engineering & MLOps

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![GCP BigQuery](https://img.shields.io/badge/GCP-BigQuery-orange.svg)](https://cloud.google.com/bigquery)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-green.svg)](https://fastapi.tiangolo.com/)

Ce projet implémente une architecture Data & ML complète et industrialisée, allant de l'ingestion automatisée de données financières multi-sources à la modélisation prédictive par Machine Learning (**XGBoost**), jusqu'à la distribution via une **API REST (FastAPI)** et la restitution graphique sur un **Dashboard décisionnel (Streamlit)** orienté gestion des risques.

---

## 🏗️ Architecture Globale & Flux de Données

Le projet est découpé en couches hermétiques et découplées, respectant scrupuleusement les standards du Data Engineering et du MLOps :

1. **Layer RAW & ENRICHED (Data Engineering)** : Pipeline d'extraction quotidien synchrone (Marché + Macroéconomie) avec gestion automatique de l'écrasement (`WRITE_TRUNCATE`). Transformation SQL avancée directe sous BigQuery (*Forward Fill* pour la complétion macroéconomique et alignement temporel strict à $J+1$).
2. **Layer ANALYTICS (Machine Learning)** : Extraction des features et entraînement isolé par actif d'un modèle **XGBoost Regressor**. Intégration d'une validation rigoureuse par **Split Temporel** (80% Train chronologique / 20% Test) pour interdire tout *data leakage* algorithmique, suivie de l'injection des inférences en base.
3. **Layer RESTITUTION (Application Node)** : Consommation asynchrone et sécurisée des prédictions BigQuery via **FastAPI** (servi sur le port stable `8006`), couplée à un **Dashboard Streamlit** hautes performances doté d'un Risk Engine temps réel (Calculs de Volatilité annualisée, Value-at-Risk 95% et Ratio de Sharpe).

Pour une analyse détaillée des choix d'architecture, de la conformité RGPD et des schémas de données, consultez le fichier [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## 🛡️ Configuration de la Sécurité & Authentification Cloud

Pour des raisons évidentes de sécurité, de gouvernance et de conformité, la clé d'accès privée à l'infrastructure Google Cloud Engine est exclue du suivi de version via le fichier `.gitignore`.

Pour répliquer, auditer ou exécuter ce projet localement, suivre ce protocole d'initialisation :

1. **Générer la clé GCP** : Rendez-vous sur la console Google Cloud, créer un compte de service dédié et téléchargez une clé d'accès privée au format **JSON** (Rôles requis : *BigQuery Admin*).
2. **Installation du secret** : Placer le fichier JSON téléchargé directement à la racine du projet et le renommer exactement : `google_credentials.json`.
3. **Variables d'environnement** : Un fichier `.env.example` est mis à disposition pour documenter les configurations du projet :
   ```bash
   cp .env.example .env


## 📂 Structure du Répertoire


market-finance-ai-dashboard/
├── .github/workflows/           # 🚀 INTÉGRATION CONTINUE (CI/CD via GitHub Actions)
│   └── ci.yml
├── tests/                       # 🧪 TESTS UNITAIRES AUTOMATISÉS & MOCKING
│   └── test_api.py              # Validation déterministe de l'API REST
├── VEILLE_ET_BENCHMARK.md       # 📑 Rapport de veille technologique et Benchmark d'outils IA
├── ARCHITECTURE.md              # 🛠️ Documentation technique avancée (RGPD, Monitorage, BigQuery)
├── README.md                    # 📖 Guide principal d'utilisation et de déploiement
├── .gitignore                   # 🛡️ Exclusion des secrets (google_credentials.json) et des caches
├── .env.example                 # Template d'illustration des variables d'environnement
├── requirements.txt             # Dépendances stables et figées du projet
└── src/
    ├── database/                # ⚙️ COUCHE DATA ENGINEERING (Ingestion & ETL SQL)
    │   ├── feeding.py           # Ingestion automatisée et nettoyage automatique (Layer RAW)
    │   └── data_engineering_alignement_temporel.py  # Orchestration du Feature Engineering SQL (Layer ENRICHED)
    ├── api/                     # 🌐 BACK-END REST (FastAPI - Port de Production 8006)
    │   └── main_api.py          # Distribution managée des prédictions et scores R²/MAE
    ├── models/                  # 🧠 COUCHE MLOPS (Entraînement & Inférence Machine Learning)
    │   └── train_predict.py     # Pipeline XGBoost, Split Temporel et alimentation de la table Analytics
    └── dashboard/               # 📊 FRONT-END VIZ & RISK ENGINE (Streamlit)
        └── dashboard_final.py   # Interface décisionnelle pro, Plotly dynamique et métriques prudentielles