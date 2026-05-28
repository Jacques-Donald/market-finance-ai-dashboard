# 🛠️ Documentation Technique & Architecture

Ce document détaille les spécifications techniques, les choix de conception, la modélisation complète des données et la stratégie de surveillance MLOps au sein du projet.

---

## 🗄️ 1. Schéma et Modélisation des Données BigQuery

Le projet orchestre le cycle de vie de la donnée à travers trois couches de données distinctes au sein du Data Warehouse Google BigQuery (`PROJECT_ID: bigquery-finance-ia`). Ce partitionnement garantit l'étanchéité des environnements et une traçabilité ascendante de l'information.

```text
  [ Couche RAW ]              [ Couche ENRICHED ]            [ Couche ANALYTICS ]
  ├── raw_market  ──┐
  └── raw_macro   ──┴──>  features_market_ready  ─────>  market_predictions
                         (ETL / Alignement Temporel)     (Inférence XGBoost)


### Couche Source : `market_finance_enriched.features_market_ready`
Cette table centralise les variables de marché et les indicateurs macro-économiques (FED, VIX, etc.) utilisés comme features par le modèle :
* `date` (STRING/DATE) : Clé temporelle principale.
* `ticker` (STRING) : Identifiant de l'actif (ex: `BTC-USD`, `^GSPC`).
* `price` (FLOAT) : Prix de clôture au jour J.
* `volume` (FLOAT) : Volume de transaction.
* `fed_rate` / `vix` / `treasury_10y` : Indicateurs macro-économiques d'environnement de marché.
* `target_next_day_price` (FLOAT) : Le prix au jour $J+1$ (Cible à prédire).

### Couche Analytique : `market_finance_analytics.market_predictions`
Cette table est générée et écrasée (`WRITE_TRUNCATE`) à chaque exécution du pipeline de Machine Learning :
* `date`, `ticker`, `price` : Données historiques réelles.
* `predicted_next_day_price` (FLOAT) : Prédiction calculée par le modèle XGBoost.
* `model_r2` (FLOAT) : Score de précision $R^2$ calculé sur le jeu de test isolé de cet actif.
* `model_mae` (FLOAT) : Erreur Moyenne Absolue (MAE) du modèle.

---

## 🧠 2. Stratégie de Machine Learning (XGBoost)

### Pourquoi XGBoost Regressor ?
Les données financières sont hautement non-linéaires et complexes. Les modèles de régression linéaire classiques échouent à capturer les ruptures de tendance. **XGBoost** (Gradient Boosting sur arbres de décision) offre un excellent compromis entre puissance prédictive, gestion des valeurs aberrantes et rapidité d'exécution, tout en évitant le surapprentissage grâce à ses paramètres de régularisation.

### Le Split Temporel : Une exigence financière
Pour évaluer le modèle sans tricher (*Data Leakage*), nous n'utilisons pas un découpage aléatoire (`train_test_split` classique). 

```text
[------ 80% Entraînement (Passé) ------][--- 20% Test (Futur) ---]
Chronologie ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔ ➔

## 🛡️ 3. Conformité RGPD & Modélisation Conceptuelle

### Respect de la vie privée dès la conception (Privacy by Design)
Le projet traite exclusivement de **données financières publiques de marchés** (cours de bourses, volumes) et d'**indicateurs macroéconomiques institutionnels** (taux de la FED, indice de stress financier, indice de volatilité VIX). 

* **Aucune donnée à caractère personnel (PII - Personally Identifiable Information)** n'est collectée, traitée ou stockée dans Google BigQuery.
* L'application ne gère aucun compte utilisateur, s'affranchissant ainsi de la gestion des cookies traceurs ou du stockage de données sensibles, garantissant une conformité native et absolue avec les directives du **RGPD**.

---

## 🪵 4. Stratégie de Monitorage, Journalisation & Feedback Loop (MLOps)

Pour assurer la surveillance en production de l'application et du modèle d'IA, les mécaniques suivantes sont préconisées :

1. **Journalisation applicative (Logging) :** L'API FastAPI intègre le module `logging` standard de Python au lieu de simples instructions `print()`. Chaque requête, temps de réponse et exception (ex: erreur 404 sur un actif, échec de connexion BigQuery) génère une ligne de log structurée.
2. **Supervision des pannes (Incident Detection) :** En environnement de production (Cloud Run / App Engine), ces logs sont capturés par **Google Cloud Logging**. Des alertes automatisées par e-mail ou webhook (Slack/Teams) sont configurées pour se déclencher si le taux d'erreurs HTTP 500 dépasse 1% sur une fenêtre de 5 minutes.
3. **Surveillance du modèle (Data Drift) :** Afin de maintenir la précision du modèle XGBoost dans le temps, un script planifié compare mensuellement la dérive des distributions des données d'entrée (ex : une hausse brutale et durable des taux de la FED) avec le jeu d'entraînement initial. Si le score $R^2$ chute en dessous d'un seuil critique (ex : 0.40), une alerte de réentraînement est émise pour réactualiser le modèle via la *Feedback Loop*.