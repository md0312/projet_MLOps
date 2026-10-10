# ADR 005 - Choix du modèle final : régression logistique

**Statut :** acceptée

## Contexte

Trois familles de modèles ont été comparées sur les probabilités out-of-fold du jeu
d'entraînement (voir `notebooks/02_modeling.ipynb`) :

| Modèle | PR-AUC (OOF) | Seuil F1 | F1 au seuil (OOF) |
|---|---|---|---|
| Régression logistique (elastic net) | 0,423 | 0,589 | 0,487 |
| Forêt aléatoire | 0,436 | 0,639 | 0,484 |
| SVM RBF calibré | 0,345 | 0,234 | 0,473 |

Les hyperparamètres de la régression logistique et de la forêt aléatoire ont été
optimisés avec Optuna (PR-AUC en validation croisée). Une nouvelle optimisation
(`application/tune.py`) confirme que la performance est sur un plateau : le meilleur
essai n'apporte que quelques millièmes de PR-AUC.

## Décision

Le modèle final est la **régression logistique équilibrée** (elastic net,
`C ≈ 0,043`, `l1_ratio ≈ 0,04`), sur le jeu de variables complet, avec un seuil de
0,589 (`configs/config.yaml`).

- Le **SVM** est écarté : PR-AUC nettement inférieur et coût d'entraînement élevé.
- La **forêt aléatoire** classe légèrement mieux les clients, mais l'écart est faible au
  regard de la variabilité entre plis, et son F1 au seuil optimal est équivalent.
- La **régression logistique** est retenue pour son **interprétabilité** : ses
  coefficients s'expliquent directement en termes métier.

## Conséquences

- Résultats sur le jeu de test : F1 0,485, précision 0,444, rappel 0,533, PR-AUC 0,420.
  Ils sont très proches des estimations out-of-fold : pas de surapprentissage.
- Les autres modèles restent disponibles via `configs/random_forest.yaml` et
  `configs/svm.yaml`, et sont comparables dans MLflow (`make compare`).
- L'interprétation des effets est détaillée dans `notebooks/03_final_model.ipynb`.
