"""Construction du jeu de modélisation et du prétraitement scikit-learn.

Les décisions de feature engineering prises lors de l'exploration sont
implémentées ici :

- ``duration`` est exclue, car elle n'est connue qu'après l'appel (fuite
  d'information) ;
- ``pdays`` est remplacée par l'indicateur ``previously_contacted``, car la
  valeur 999 signifie « jamais contacté » et non un délai réel ;
- les modalités ``unknown`` sont conservées comme des catégories à part entière.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"
NOT_PREVIOUSLY_CONTACTED = 999
EXCLUDED_COLUMNS = [TARGET_COLUMN, "duration", "pdays"]

# Jeu interprétable : une seule variable macroéconomique (euribor3m), car les
# autres lui sont très corrélées.
INTERPRETABLE_FEATURES = [
    "age",
    "campaign",
    "previous",
    "euribor3m",
    "previously_contacted",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def add_previous_contact_flag(data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'indicateur ``previously_contacted`` à partir de ``pdays``.

    Parameters
    ----------
    data : pd.DataFrame
        Données contenant la colonne ``pdays``.

    Returns
    -------
    pd.DataFrame
        Copie des données avec la colonne ``previously_contacted`` (1 si le
        client a été contacté lors d'une campagne précédente, 0 sinon).
    """
    return data.assign(
        previously_contacted=data["pdays"].ne(NOT_PREVIOUSLY_CONTACTED).astype("int8")
    )


def build_modeling_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construit les variables explicatives et la cible binaire.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes, telles que chargées depuis le CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Les variables explicatives (sans la cible, ``duration`` ni ``pdays``)
        et la cible (1 pour une souscription, 0 sinon).
    """
    transformed_data = add_previous_contact_flag(data)

    target = transformed_data[TARGET_COLUMN].eq(POSITIVE_LABEL).astype("int8")
    features = transformed_data.drop(columns=EXCLUDED_COLUMNS)

    logger.info(
        "Jeu de modélisation : %d lignes, %d variables, taux de souscription %.2f %%",
        len(features),
        features.shape[1],
        100 * target.mean(),
    )
    return features, target


def select_features(features: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Sélectionne un jeu de variables explicatives.

    Parameters
    ----------
    features : pd.DataFrame
        Variables explicatives produites par :func:`build_modeling_dataset`.
    feature_set : {"complete", "interpretable"}
        ``complete`` conserve toutes les variables ; ``interpretable`` retire
        les variables macroéconomiques redondantes et ``day_of_week``.

    Returns
    -------
    pd.DataFrame
        Les variables du jeu demandé.

    Raises
    ------
    ValueError
        Si le nom du jeu de variables est inconnu.
    """
    if feature_set == "complete":
        return features
    if feature_set == "interpretable":
        return features[INTERPRETABLE_FEATURES]
    raise ValueError(
        f"Jeu de variables inconnu : {feature_set!r} (attendu : 'complete' ou 'interpretable')"
    )


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare les données en jeux d'entraînement et de test stratifiés.

    Parameters
    ----------
    features : pd.DataFrame
        Variables explicatives.
    target : pd.Series
        Cible binaire.
    test_size : float
        Proportion des observations réservées au test (par exemple 0.2).
    random_state : int
        Graine aléatoire, pour un découpage reproductible.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``X_train``, ``X_test``, ``y_train`` et ``y_test``.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    logger.info("Découpage : %d lignes d'entraînement, %d de test", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def create_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Construit le prétraitement adapté aux colonnes fournies.

    Les variables numériques sont imputées par la médiane puis standardisées.
    Les variables catégorielles sont imputées par la modalité la plus
    fréquente puis encodées en one-hot. Le préprocesseur n'est pas ajusté :
    il le sera dans un pipeline, uniquement sur les données d'entraînement.

    Parameters
    ----------
    features : pd.DataFrame
        Variables explicatives, utilisées seulement pour identifier le type de
        chaque colonne.

    Returns
    -------
    ColumnTransformer
        Préprocesseur non ajusté.
    """
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
