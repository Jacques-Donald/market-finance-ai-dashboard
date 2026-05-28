# 📑 Rapport de Veille Technique & Benchmark d'Intelligence Artificielle

## 1. Organisation de la Veille Technique et Réglementaire (Compétence 6)
Dans le cadre de l'industrialisation de ce projet de prévisions financières, un protocole de veille systématique a été mis en place afin de garantir l'adéquation de la solution avec l'état de l'art et les réglementations en vigueur (notamment l'EU AI Act et les directives de l'AMF / ESMA sur l'utilisation des algorithmes sur les marchés).

### Sélection des sources et partage (Veille Collective)
* **Sources Techniques :** Suivi des publications scientifiques (arXiv - section *Computational Finance*), des blogs d'ingénierie (Google AI, Meta AI, AWS Architecture) et de la documentation officielle des frameworks (*XGBoost Documentation*, *Scikit-Learn Releases*).
* **Sources Réglementaires :** Bulletins de la CNIL (conformité RGPD), publications de l'ESMA (European Securities and Markets Authority) sur les risques liés au trading algorithmique et à l'intégrité des marchés.
* **Outils de partage :** Centralisation des flux RSS via un canal partagé (Slack/Teams) et revues techniques bi-mensuelles au sein de l'équipe pour formuler des recommandations d'évolution architecturales.

---

## 2. Benchmark de Services d'Intelligence Artificielle (Compétence 7)
À partir du besoin exprimé d'anticiper les prix de clôture à $J+1$ tout en fournissant des indicateurs de gestion des risques (Moteur prudentiel : Volatilité, VaR, Sharpe), une étude comparative (benchmark) a été menée entre trois grandes familles de solutions d'IA :


### Solution : Modèle Custom ML propriétaire (XGBoost Regressor via SKLearn)
* **Caractéristiques :** Algorithme de Gradient Boosting sur arbres de décision, état de l'art mondial sur les données tabulaires et les séries temporelles enrichies.
* **Avantages majeurs retenus :** Maîtrise totale de la validation chronologique (Split Temporel 80/20), capacité à isoler l'entraînement par actif financier, intégration ultra-légère et native dans FastAPI, coût d'infrastructure nul (calcul local ou sur conteneur minimal).


---

## 3. Paramétrage, Documentation & Connecteurs (Compétence 8)
Le modèle XGBoost Regressor a été configuré et encapsulé conformément aux spécifications de sa documentation technique officielle.

### Paramétrage de l'algorithme (`train_model.py`)
Le script de modélisation applique des hyperparamètres stricts pour limiter le surapprentissage (overfitting) inhérent aux données financières :
* `n_estimators=100` : Nombre d'arbres de décision limités pour éviter d'apprendre le "bruit" du marché.
* `learning_rate=0.05` : Pas d'apprentissage réduit favorisant une convergence stable.
* `max_depth=5` : Profondeur des arbres contrainte pour préserver la capacité de généralisation du modèle sur les données futures.

### Intégration des Connecteurs et Distribution
Le modèle extrait ses données en amont via le connecteur BigQuery (`google.cloud.bigquery`), puis expose ses résultats d'inférence en aval à travers une route d'API REST FastAPI (`/predictions/{ticker}`). 
La documentation interactive de ce connecteur est nativement accessible via Swagger UI (`http://127.0.0.1:8005/docs`), permettant une intégration fluide, découplée et standardisée avec l'interface Streamlit.