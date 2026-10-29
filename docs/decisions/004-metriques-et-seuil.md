# ADR 004 - Métriques d'évaluation et choix du seuil de décision

**Statut :** acceptée

## Contexte

La cible est déséquilibrée : 10,95 % de souscripteurs. Un modèle qui prédit « non »
pour tout le monde obtiendrait 89 % d'accuracy sans aucune utilité.

Par ailleurs, le seuil par défaut de 0,5 n'a pas le même sens pour tous les modèles :
`class_weight="balanced"` gonfle les probabilités, alors que le calibrage du SVM les
ramène vers le taux réel de souscription.

## Décision

- **Métriques :** ROC-AUC et surtout **PR-AUC** (indépendantes du seuil), puis
  précision, rappel et F1 au seuil retenu. L'accuracy n'est pas utilisée.
- **Pondération :** `class_weight="balanced"` pour tous les modèles.
- **Seuil :** choisi pour maximiser le **F1** sur les **probabilités out-of-fold** du jeu
  d'entraînement (validation croisée stratifiée à 5 plis).
- **Jeu de test :** utilisé **une seule fois**, pour l'évaluation finale.

## Conséquences

- Le seuil est recalculé à chaque entraînement (`threshold.strategy` dans
  `configs/config.yaml`) au lieu d'être figé.
- Le seuil de Youden reste disponible (`strategy: youden`) ; pour le modèle final, il
  coïncide avec le seuil F1 (0,589).
- Le seuil peut être ajusté selon le coût métier d'un appel : un seuil plus bas
  détecte plus de souscripteurs, au prix de plus d'appels inutiles.
