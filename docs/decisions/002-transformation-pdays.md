# ADR 002 — Remplacement de `pdays` par `previously_contacted`

**Statut :** acceptée

## Contexte

`pdays` donne le nombre de jours depuis le dernier contact d'une campagne précédente.
Pour 96,12 % des clients, elle vaut 999, ce qui signifie « jamais contacté » et non un
délai de 999 jours. Utilisée telle quelle, elle ferait croire au modèle que ces clients
ont été contactés il y a très longtemps.

Une variable `days_since_previous_contact` (délai réel, manquant sinon) a aussi été
envisagée, mais elle serait manquante pour 96 % des clients.

## Décision

- `pdays` est retirée des variables explicatives.
- Elle est remplacée par l'indicateur `previously_contacted` (1 si contacté lors d'une
  campagne précédente, 0 sinon).
- L'historique reste aussi décrit par `previous` et `poutcome`.

## Conséquences

- La valeur sentinelle 999 n'est plus interprétée comme un nombre.
- Le délai exact depuis le dernier contact n'est pas utilisé ; il ne concerne que
  160 clients sur 4 119.
- La transformation est implémentée dans `add_previous_contact_flag` et testée.
