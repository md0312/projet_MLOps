# ADR 003 - Conservation des modalités `unknown`

**Statut :** acceptée

## Contexte

Six variables catégorielles contiennent la modalité `unknown` (`default`, `education`,
`housing`, `loan`, `job`, `marital`). Ce n'est pas une valeur manquante classique :
l'information n'a pas été renseignée.

Pour `default`, 19,50 % des clients sont `unknown`, et leur taux de souscription est
nettement plus faible (6,1 %) que celui des clients dont l'information est connue
(12,1 %).

## Décision

Les modalités `unknown` sont conservées comme des catégories à part entière, sans
remplacement par la modalité la plus fréquente.

## Conséquences

- L'information portée par l'absence de réponse est conservée.
- Aucune hypothèse arbitraire n'est faite sur la vraie valeur.
- Pour le modèle d'inférence statistique (statsmodels), `loan` est retirée : ses
  clients `unknown` sont exactement ceux de `housing`, ce qui crée une colinéarité
  parfaite. Le modèle de prédiction, régularisé, n'est pas concerné.
