# Raccourcis pour les tâches courantes du projet.
# Usage : make <cible>, par exemple « make test ».
# Les variables peuvent être modifiées : make train CONFIG=configs/svm.yaml

PYTHON = python
SOURCES = src tests
CONFIG = configs/config.yaml
N_TRIALS = 40

.PHONY: install lint format test coverage train evaluate pipeline compare tune ui

install:  ## Installe l'environnement et le package
	$(PYTHON) -m pip install -r requirements.txt

lint:  ## Vérifie le style et la qualité du code
	ruff check $(SOURCES)
	ruff format --check $(SOURCES)

format:  ## Corrige automatiquement la mise en forme
	ruff check --fix $(SOURCES)
	ruff format $(SOURCES)

test:  ## Lance tous les tests
	pytest

coverage:  ## Lance les tests et mesure la couverture du code
	pytest --cov=bank_marketing --cov-report=term-missing

train:  ## Entraîne le modèle et choisit son seuil
	$(PYTHON) -m bank_marketing.application.train --config $(CONFIG)

evaluate:  ## Évalue le modèle sur le jeu de test
	$(PYTHON) -m bank_marketing.application.evaluate --config $(CONFIG)

pipeline:  ## Entraîne et évalue un modèle suivi dans MLflow
	$(PYTHON) -m bank_marketing.application.run_pipeline --config $(CONFIG)

compare:  ## Enregistre les trois modèles candidats dans MLflow
	$(PYTHON) -m bank_marketing.application.run_pipeline --config configs/config.yaml
	$(PYTHON) -m bank_marketing.application.run_pipeline --config configs/random_forest.yaml
	$(PYTHON) -m bank_marketing.application.run_pipeline --config configs/svm.yaml

tune:  ## Optimise les hyperparamètres avec Optuna
	$(PYTHON) -m bank_marketing.application.tune --config $(CONFIG) --n-trials $(N_TRIALS)

ui:  ## Lance l'interface MLflow (http://127.0.0.1:5000)
	mlflow ui
