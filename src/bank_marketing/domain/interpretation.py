"""Interprétation de la régression logistique : coefficients et significativité.

Deux modèles répondent à deux questions différentes :

- le modèle de **prédiction** (régression logistique pénalisée) dit qui appeler ;
  ses coefficients, réduits vers zéro par la pénalité, donnent le sens et
  l'importance relative des effets ;
- un modèle d'**inférence** (régression logistique non pénalisée, statsmodels)
  fournit les p-values et intervalles de confiance, qui ne sont pas valides pour
  un modèle pénalisé.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables retirées du modèle d'inférence pour le rendre estimable :
# - loan : ses clients "unknown" sont exactement ceux de housing (colinéarité) ;
# - default : modalité "yes" quasi absente (quasi-séparation) ;
# - previously_contacted : information déjà portée par poutcome.
INFERENCE_EXCLUDED_FEATURES = ["loan", "default", "previously_contacted"]

# Modalités trop rares regroupées avec une modalité voisine.
RARE_CATEGORY_GROUPS = {"education": {"illiterate": "basic.4y"}}


def extract_logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extrait les coefficients d'une régression logistique entraînée.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline entraîné ``preprocessor -> model`` dont le modèle est une
        régression logistique.

    Returns
    -------
    pd.DataFrame
        Coefficient et odds ratio de chaque variable encodée, triés par
        valeur absolue décroissante du coefficient.
    """
    feature_names = [
        name.split("__", maxsplit=1)[-1]
        for name in pipeline["preprocessor"].get_feature_names_out()
    ]
    coefficients = pipeline["model"].coef_[0]
    return (
        pd.DataFrame(
            {"coefficient": coefficients, "odds_ratio": np.exp(coefficients)},
            index=pd.Index(feature_names, name="variable"),
        )
        .sort_values("coefficient", key=np.abs, ascending=False)
        .round(3)
    )


def prepare_inference_features(features: pd.DataFrame) -> pd.DataFrame:
    """Prépare les variables du modèle d'inférence.

    Parameters
    ----------
    features : pd.DataFrame
        Variables explicatives (idéalement le jeu interprétable).

    Returns
    -------
    pd.DataFrame
        Variables sans redondance, avec les modalités rares regroupées.
    """
    prepared = features.drop(columns=INFERENCE_EXCLUDED_FEATURES, errors="ignore")
    for column, groups in RARE_CATEGORY_GROUPS.items():
        if column in prepared.columns:
            prepared = prepared.assign(**{column: prepared[column].replace(groups)})
    return prepared


def fit_inference_model(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Estime une régression logistique non pénalisée et ses tests de significativité.

    Les variables numériques sont standardisées : leur odds ratio correspond à
    une hausse d'un écart-type. Chaque variable catégorielle perd sa première
    modalité (ordre alphabétique), qui sert de référence.

    Parameters
    ----------
    features : pd.DataFrame
        Variables explicatives d'entraînement.
    target : pd.Series
        Cible binaire d'entraînement.

    Returns
    -------
    pd.DataFrame
        Coefficient, odds ratio et son intervalle de confiance à 95 %, p-value
        et indicateur de significativité au seuil de 5 %, triés par p-value.
    """
    prepared = prepare_inference_features(features)
    encoder = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), prepared.select_dtypes(include="number").columns),
            (
                "categorical",
                OneHotEncoder(drop="first", sparse_output=False),
                prepared.select_dtypes(exclude="number").columns,
            ),
        ],
        verbose_feature_names_out=False,
    )
    design_matrix = pd.DataFrame(
        encoder.fit_transform(prepared),
        columns=encoder.get_feature_names_out(),
        index=prepared.index,
    )
    result = sm.Logit(target, sm.add_constant(design_matrix)).fit(disp=False)

    confidence_intervals = np.exp(result.conf_int())
    table = pd.DataFrame(
        {
            "coefficient": result.params,
            "odds_ratio": np.exp(result.params),
            "or_ic95_bas": confidence_intervals[0],
            "or_ic95_haut": confidence_intervals[1],
            "p_value": result.pvalues,
        }
    ).drop(index="const")
    return (
        table.assign(significatif_5pct=table["p_value"] < 0.05)
        .rename_axis("variable")
        .sort_values("p_value")
        .round(3)
    )
