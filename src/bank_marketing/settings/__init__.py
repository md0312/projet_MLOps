"""Paramètres globaux du projet : chemins des dossiers et fichiers clés.

Tous les chemins sont construits à partir de la racine du dépôt, ce qui permet
d'exécuter le code depuis n'importe quel dossier (notebooks, scripts, tests).
"""

from pathlib import Path

# src/bank_marketing/settings/__init__.py -> remonter de trois niveaux
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank-additional.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Configuration
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "config.yaml"
LOGGING_CONFIG_PATH = Path(__file__).resolve().parent / "logging.yaml"

# Sorties
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
