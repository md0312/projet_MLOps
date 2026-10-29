# ADR 006 - Versionnement des données brutes dans Git

**Statut :** acceptée

## Contexte

Les données brutes ne sont généralement pas versionnées dans Git : elles peuvent être
volumineuses ou confidentielles. Ici, `data/raw/bank-additional.csv` est un jeu de
données public, de petite taille (4 119 lignes, environ 580 Ko).

## Décision

Le fichier brut est versionné dans le dépôt, sans aucune modification.

## Conséquences

- Le projet est reproductible immédiatement après un `git clone`, sans étape de
  téléchargement.
- Les tests d'intégration et la CI peuvent utiliser les vraies données.
- Les artefacts produits (modèles, base MLflow) ne sont **pas** versionnés : ils se
  régénèrent avec les commandes du projet (voir `.gitignore`).
