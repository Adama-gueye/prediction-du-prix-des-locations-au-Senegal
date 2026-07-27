# Prédiction du prix des locations au Sénégal

Projet M2 DSIA — mise en production d'un modèle de prédiction du prix de
location de biens immobiliers (appartement, maison) au Sénégal, de la collecte
des données jusqu'à l'exposition via une API conteneurisée, avec interface
front-end React.

![CI](https://github.com/Adama-gueye/prediction-du-prix-des-locations-au-Senegal/actions/workflows/ci.yml/badge.svg)

## Sommaire

- [Installation](#installation)
- [Récupérer et préparer les données](#récupérer-et-préparer-les-données)
- [Entraîner un modèle](#entraîner-un-modèle)
- [Suivre les expériences avec MLflow](#suivre-les-expériences-avec-mlflow)
- [Lancer l'API](#lancer-lapi)
- [Lancer le front-end](#lancer-le-front-end)
- [Lancer les tests](#lancer-les-tests)
- [Conteneurisation Docker](#conteneurisation-docker)
- [Images Docker Hub](#images-docker-hub)
- [Structure du projet](#structure-du-projet)

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -e ".[dev]"
```

Le `[dev]` installe en plus les outils de développement (pytest, mypy, black, ruff).

## Récupérer et préparer les données

Les données brutes et nettoyées ne sont pas versionnées (voir `.gitignore`) ;
elles se régénèrent avec les deux commandes suivantes.

**1. Scraping** (annonces NeoBien, usage strictement académique) :
```bash
python app.py
```
Génère `data/raw/locations.csv`.

**2. Nettoyage** :
```bash
python -m senegal_rental_price.data.preprocessing
```
Génère `data/processed/locations_clean.csv`.

Voir `data/README.md` pour le détail de la provenance et des choix de
nettoyage, et `notebooks/01_exploration.ipynb` pour l'analyse exploratoire
complète.

## Entraîner un modèle

Configuration gérée par Hydra — aucun hyperparamètre codé en dur.

```bash
# Modèle par défaut (Random Forest)
python -m senegal_rental_price.models.train

# Autre modèle
python -m senegal_rental_price.models.train model=ridge
python -m senegal_rental_price.models.train model=xgboost

# Changer un hyperparamètre en ligne de commande
python -m senegal_rental_price.models.train model=xgboost model.params.max_depth=8
```

Chaque exécution :
- entraîne le modèle sur `log1p(prix)` (stabilise l'apprentissage face aux prix extrêmes),
- calcule une validation croisée à 5 plis en plus du split train/test,
- logue métriques et hyperparamètres dans MLflow,
- sauvegarde `models/<nom_du_modele>.pkl` + ses métadonnées (`*_features.json`, `*_metadata.json`).

## Suivre les expériences avec MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```
Puis ouvrir `http://localhost:5000` — expérience **"senegal-rental-price"**,
avec les runs `ridge`, `random_forest`, `xgboost` comparables côte à côte.

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

Documentation interactive : `http://localhost:8000/docs`

Endpoints :
| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Vérification de l'état du service |
| GET | `/model/info` | Métadonnées du modèle chargé |
| POST | `/predict` | Prédiction du prix à partir des caractéristiques d'un bien |

## Lancer le front-end

```bash
cd frontend
npm install
npm run dev
```

Ouvre `http://localhost:5173`. Nécessite l'API lancée en parallèle (CORS déjà
configuré entre les deux, cf. `CORS_ORIGINS` dans `api/main.py`).

## Lancer les tests

```bash
pytest --cov=src --cov-report=term-missing
```

Couverture minimale exigée : 70 % sur `src/` (actuellement ~82 %).

Vérifications qualité complètes (identiques à la CI) :
```bash
ruff check .
black --check .
mypy src/
pytest --cov=src --cov-fail-under=70
```

## Conteneurisation Docker

```bash
cd docker
docker compose up --build
```

- API : `http://localhost:8000/docs`
- Front-end : `http://localhost:8080`

⚠️ Les modèles entraînés (`models/`) doivent déjà exister en local avant de
lancer Docker : ils sont montés en volume, pas copiés dans l'image (pour ne
pas avoir à rebuild à chaque nouvel entraînement).

## Images Docker Hub

Les images sont automatiquement construites et publiées par la CI à chaque
push sur `main` :

```bash
docker pull TON_USERNAME/senegal-rental-api:latest
docker pull TON_USERNAME/senegal-rental-frontend:latest
```

⚠️ L'image API seule ne suffit pas pour que `/predict` fonctionne : il faut
aussi fournir un dossier `models/` (contenant `random_forest.pkl` et ses
fichiers `_features.json`/`_metadata.json`) monté en volume, ces fichiers
n'étant pas inclus dans l'image (limitation connue, documentée dans le rapport).

## Structure du projet

```
prix_location_Senegal/
├── conf/                  # Configuration Hydra (modèles, données)
├── data/                  # Données brutes/nettoyées (non versionnées) + README
├── notebooks/              # Analyse exploratoire
├── src/senegal_rental_price/
│   ├── data/               # Nettoyage (preprocessing.py)
│   ├── features/           # Feature engineering (build_features.py)
│   ├── models/              # Entraînement, prédiction (train.py, predict.py)
│   └── utils/               # Logging centralisé
├── api/                    # API FastAPI (schemas, dependencies, main)
├── frontend/                # Application React (Vite)
├── tests/                   # Tests unitaires (pytest)
├── docker/                  # Dockerfiles + docker-compose
├── models/                  # Modèles entraînés sérialisés (non versionnés)
└── .github/workflows/       # CI (lint, typage, tests, build+push Docker)
```

## Licence

Usage académique — projet M2 DSIA.