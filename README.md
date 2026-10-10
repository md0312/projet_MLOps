# Bank Marketing - prédiction de la souscription à un dépôt à terme

Projet MLOps du Master 2 SIAD : construire, de l'exploration au suivi des expériences,
un modèle qui aide une banque à **cibler les clients à appeler** lors d'une campagne de
télémarketing pour un dépôt à terme.

L'accent est mis sur la **démarche MLOps** : code structuré en couches, configuration
séparée du code, tests automatisés, reproductibilité et suivi des expériences avec MLflow.

## Résultats

Modèle retenu : **régression logistique équilibrée** (régularisation *elastic net*),
choisie pour son interprétabilité ([ADR 005](docs/decisions/005-choix-modele-final.md)).

| Jeu | ROC-AUC | PR-AUC | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| Entraînement (out-of-fold) | 0,772 | 0,423 | 0,418 | 0,582 | 0,487 |
| Test | 0,789 | 0,420 | 0,444 | 0,533 | 0,485 |

Seuil de décision : **0,589**, choisi sur les probabilités out-of-fold du jeu
d'entraînement.

En appelant uniquement les clients ciblés par le modèle (13 % du jeu de test), la banque
atteindrait 53 % des souscripteurs avec une précision de 44 %, soit environ **4 fois** le
taux de souscription d'un appel au hasard (11 %).

## Installation

Prérequis : **Python 3.14** et Git.

```bash
git clone https://github.com/md0312/projet_MLOps.git
cd projet_MLOps
python -m venv .venv
```

Activer l'environnement :

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Installer les dépendances et le package du projet :

```bash
pip install -r requirements.txt
```

`requirements.txt` fixe la version de chaque librairie et installe le package
`bank_marketing` en mode éditable (`-e .`).

## Utilisation

Les tâches courantes sont regroupées dans le `Makefile`. Chaque cible correspond à une
commande Python, utilisable directement si `make` n'est pas installé.

| Commande | Équivalent | Rôle |
|---|---|---|
| `make test` | `pytest` | Lance les tests unitaires et d'intégration |
| `make coverage` | `pytest --cov=bank_marketing --cov-report=term-missing` | Tests et couverture du code |
| `make lint` | `ruff check src tests` et `ruff format --check src tests` | Vérifie la qualité du code |
| `make format` | `ruff check --fix src tests` et `ruff format src tests` | Corrige la mise en forme |
| `make train` | `python -m bank_marketing.application.train` | Entraîne le modèle et choisit son seuil |
| `make evaluate` | `python -m bank_marketing.application.evaluate` | Évalue le modèle sur le jeu de test |
| `make pipeline` | `python -m bank_marketing.application.run_pipeline` | Entraîne et évalue, avec suivi MLflow |
| `make compare` | `run_pipeline` sur les trois configurations | Enregistre les trois modèles dans MLflow |
| `make tune` | `python -m bank_marketing.application.tune` | Optimise les hyperparamètres avec Optuna |
| `make ui` | `mlflow ui` | Ouvre l'interface MLflow sur http://127.0.0.1:5000 |

Les commandes utilisent `configs/config.yaml` par défaut. Pour une autre configuration :

```bash
make train CONFIG=configs/svm.yaml
make tune CONFIG=configs/random_forest.yaml N_TRIALS=20
```

### Reproduire les résultats

```bash
make train      # modèle final : seuil 0,589, F1 OOF 0,487
make evaluate   # jeu de test : F1 0,485
```

Le modèle est enregistré dans `models/model.joblib` et les métriques dans
`reports/results/`. Toutes les graines aléatoires sont fixées : les résultats sont
identiques d'une machine à l'autre.

### Suivi des expériences avec MLflow

```bash
make compare                                  # les trois modèles candidats
make tune                                     # optimisation de la régression logistique
make tune CONFIG=configs/random_forest.yaml   # optimisation de la forêt aléatoire
make ui
```

Dans l'interface, choisir le mode **Model training**, puis l'expérience
**bank-marketing**. Chaque run contient la configuration, les métriques out-of-fold et de
test, les fichiers de résultats et le modèle. Les optimisations Optuna apparaissent sous
la forme d'un run parent contenant un run par essai.

Les points d'entrée sont aussi déclarés dans le fichier `MLproject` :

```bash
mlflow run . -e run_pipeline -P config=configs/svm.yaml --env-manager local --experiment-name bank-marketing
```

## Configuration

Le modèle et ses paramètres sont décrits dans des fichiers YAML : on change de modèle
sans modifier le code.

| Fichier | Modèle |
|---|---|
| `configs/config.yaml` | Régression logistique, **modèle final** |
| `configs/random_forest.yaml` | Forêt aléatoire, candidat comparé |
| `configs/svm.yaml` | SVM à noyau RBF calibré, candidat comparé |

Chaque fichier précise la graine aléatoire, la part du jeu de test, le jeu de variables
(`complete` ou `interpretable`), le nombre de plis de validation croisée, le modèle et ses
hyperparamètres, et la stratégie de choix du seuil (`f1` ou `youden`).

## Structure du projet

```
projet_MLOps/
├── configs/                  Configurations des modèles (YAML)
├── data/raw/                 Données brutes, jamais modifiées
├── docs/
│   ├── data/                 Dictionnaire des données
│   └── decisions/            Décisions du projet (ADR)
├── models/                   Modèle entraîné (non versionné)
├── notebooks/
│   ├── 01_eda.ipynb          Analyse exploratoire
│   ├── 02_modeling.ipynb     Comparaison des modèles et optimisation
│   └── 03_final_model.ipynb  Modèle final, évaluation et interprétation
├── reports/
│   ├── figures/              Figures du rapport
│   └── results/              Métriques et configurations optimisées
├── src/bank_marketing/
│   ├── settings/             Chemins, logging, paramètres MLflow
│   ├── infrastructure/       Lecture et écriture des fichiers
│   ├── domain/               Logique métier : prétraitement, modèles, évaluation
│   └── application/          Scripts : train, evaluate, run_pipeline, tune
├── tests/
│   ├── unit/                 Tests d'une fonction isolée
│   └── integration/          Tests de plusieurs couches ensemble, sur les vraies données
├── Makefile                  Raccourcis des tâches courantes
├── MLproject                 Points d'entrée MLflow
├── pyproject.toml            Package, pytest et ruff
└── requirements.txt          Dépendances aux versions fixées
```

Le code suit une architecture en couches (*Domain Driven Design*) :

- **`domain/`** contient la logique métier, sans accès au disque : il manipule des
  DataFrames et des modèles ;
- **`infrastructure/`** gère les entrées et sorties (CSV, YAML, modèles, figures) ;
- **`application/`** orchestre les deux couches précédentes dans des scripts exécutables ;
- **`settings/`** centralise les chemins et la configuration du logging.

Les notebooks n'implémentent aucune logique : ils appellent les fonctions de `src/` et
documentent les analyses.

## Démarche et décisions

Les choix méthodologiques sont documentés dans [`docs/decisions/`](docs/decisions/README.md) :

- exclusion de `duration`, connue seulement après l'appel (fuite d'information) ;
- remplacement de `pdays` par l'indicateur `previously_contacted` ;
- conservation des modalités `unknown` ;
- évaluation par PR-AUC et F1, seuil choisi sur les probabilités out-of-fold, jeu de test
  utilisé une seule fois ;
- choix de la régression logistique pour son interprétabilité.

Les variables sont décrites dans le [dictionnaire des données](docs/data/dictionnaire_donnees.md).

## Qualité du code

- **Tests :** 60 tests unitaires et d'intégration, couverture de 93 % (`make coverage`).
- **Style :** `ruff` vérifie la conformité PEP 8, l'ordre des imports, les conventions de
  nommage et la présence de docstrings au format NumPy.
- **Logging :** le module `logging` remplace les `print`.
- **Reproductibilité :** versions des librairies fixées, graines aléatoires fixées,
  fins de ligne normalisées (`.gitattributes`).

## Conventions de travail

- Chaque développement se fait sur une branche dédiée (`experiment/...`, `feat/...`),
  jamais directement sur `main`.
- Les branches sont intégrées à `main` par Pull Request, relue par un membre de l'équipe.
- Les messages de commit suivent les [Conventional Commits](https://www.conventionalcommits.org/fr/) :
  `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- Avant chaque commit : `make lint` et `make test`.

## Données

*Bank Marketing* (version « additional »), UCI Machine Learning Repository :
S. Moro, P. Cortez et P. Rita (2014), *A Data-Driven Approach to Predict the Success of
Bank Telemarketing*, Decision Support Systems. Échantillon de 4 119 clients d'une banque
portugaise (2008-2010).

## Équipe

- **Dargus MWETE**
- **Josue KANTENG-A-MUKOJ**
- **MOUSSA MENHOUK**
