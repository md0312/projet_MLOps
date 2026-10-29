# Dictionnaire des données

## Origine

| Élément | Valeur |
|---|---|
| Jeu de données | *Bank Marketing* (version « additional »), UCI Machine Learning Repository |
| Référence | S. Moro, P. Cortez et P. Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems |
| Contexte | Campagnes de télémarketing d'une banque portugaise, de mai 2008 à novembre 2010 |
| Fichier | `data/raw/bank-additional.csv` (séparateur `;`) |
| Taille | 4 119 clients, 21 colonnes (échantillon de 10 % du fichier complet de 41 188 lignes) |
| Cible | `y` : le client a-t-il souscrit un dépôt à terme ? (`yes` / `no`) |
| Taux de souscription | 10,95 % (cible déséquilibrée) |

Aucune valeur manquante technique n'est présente. Plusieurs variables catégorielles
contiennent en revanche la modalité `unknown` (information non renseignée).

## Variables

### Données du client

| Variable | Type | Description | Valeurs |
|---|---|---|---|
| `age` | numérique | Âge du client | 18 à 88 ans |
| `job` | catégorielle | Profession | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` |
| `marital` | catégorielle | Situation familiale (`divorced` inclut les veufs) | `divorced`, `married`, `single`, `unknown` |
| `education` | catégorielle | Niveau d'études | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` |
| `default` | catégorielle | Défaut de crédit en cours | `no`, `yes`, `unknown` |
| `housing` | catégorielle | Prêt immobilier en cours | `no`, `yes`, `unknown` |
| `loan` | catégorielle | Prêt personnel en cours | `no`, `yes`, `unknown` |

### Dernier contact de la campagne actuelle

| Variable | Type | Description | Valeurs |
|---|---|---|---|
| `contact` | catégorielle | Canal du contact | `cellular`, `telephone` |
| `month` | catégorielle | Mois du dernier contact | `mar` à `dec` (pas de contact en janvier ni février) |
| `day_of_week` | catégorielle | Jour du dernier contact | `mon` à `fri` |
| `duration` | numérique | Durée du dernier appel, en secondes | 0 à 3 643 |

### Historique des campagnes

| Variable | Type | Description | Valeurs |
|---|---|---|---|
| `campaign` | numérique | Nombre de contacts pendant la campagne actuelle, dernier contact inclus | 1 à 35 |
| `pdays` | numérique | Jours écoulés depuis le dernier contact d'une campagne précédente ; **999 = jamais contacté** | 0 à 21, ou 999 |
| `previous` | numérique | Nombre de contacts avant la campagne actuelle | 0 à 6 |
| `poutcome` | catégorielle | Résultat de la campagne précédente | `failure`, `nonexistent`, `success` |

### Contexte économique

| Variable | Type | Description | Fréquence |
|---|---|---|---|
| `emp.var.rate` | numérique | Taux de variation de l'emploi | trimestrielle |
| `cons.price.idx` | numérique | Indice des prix à la consommation | mensuelle |
| `cons.conf.idx` | numérique | Indice de confiance des consommateurs | mensuelle |
| `euribor3m` | numérique | Taux Euribor à 3 mois | quotidienne |
| `nr.employed` | numérique | Nombre de salariés (en milliers) | trimestrielle |

### Cible

| Variable | Type | Description | Valeurs |
|---|---|---|---|
| `y` | binaire | Souscription d'un dépôt à terme | `yes`, `no` |

## Variables créées ou exclues pour la modélisation

| Variable | Statut | Raison | Décision |
|---|---|---|---|
| `duration` | exclue | Connue seulement après l'appel : fuite d'information | [ADR 001](../decisions/001-exclusion-duration.md) |
| `pdays` | remplacée | La valeur 999 n'est pas un délai | [ADR 002](../decisions/002-transformation-pdays.md) |
| `previously_contacted` | créée | Indicateur 1/0 de contact lors d'une campagne précédente | [ADR 002](../decisions/002-transformation-pdays.md) |
| `y` | transformée | Encodée en 1 (`yes`) / 0 (`no`) | — |

Le code correspondant se trouve dans `src/bank_marketing/domain/preprocessing.py`.
