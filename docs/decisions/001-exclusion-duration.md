# ADR 001 - Exclusion de la variable `duration`

**Statut :** acceptée

## Contexte

`duration` est la durée du dernier appel. Elle est très liée à la souscription : la
durée médiane est de 458 secondes chez les souscripteurs contre 165 chez les autres.
Mais elle n'est connue **qu'après** l'appel, alors que le modèle doit aider à décider
**qui appeler**.

## Décision

`duration` est exclue des variables explicatives.

## Conséquences

- Le modèle n'utilise que des informations disponibles avant l'appel : ses performances
  sont réalistes en situation réelle.
- Les performances sont plus modestes qu'avec `duration` (PR-AUC d'environ 0,42), mais
  elles ne reposent sur aucune fuite d'information.
- La règle est appliquée dans `build_modeling_dataset` et vérifiée par le test
  `test_build_modeling_dataset_excludes_leakage_and_target`.
