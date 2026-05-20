"""
src/preprocessing.py
====================
Módulo de preprocesamiento reutilizable para el dataset COPD GOLD.
Contiene pipelines de limpieza con scikit-learn, detección de outliers
y funciones de carga/split de datos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib


# ── Rutas del proyecto ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "DB"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_raw_data(filename: str = "230PatientsCOPD.xlsx") -> pd.DataFrame:
    """
    Carga el dataset crudo desde la carpeta DB.

    Parameters
    ----------
    filename : str
        Nombre del archivo (xlsx o csv).

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos originales.
    """
    filepath = DATA_RAW / filename
    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        df = pd.read_csv(filepath)
    print(f"OK Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def get_feature_groups(df: pd.DataFrame, target_col: str = "GOLD_stage"):
    """
    Clasifica las columnas en numéricas y categóricas,
    excluyendo la variable objetivo.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Nombre de la columna objetivo.

    Returns
    -------
    tuple[list, list]
        (columnas_numéricas, columnas_categóricas)
    """
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def detect_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Detecta outliers usando el método IQR.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Columnas numéricas a analizar.
    factor : float
        Multiplicador del IQR (default 1.5).

    Returns
    -------
    pd.DataFrame
        DataFrame con conteo de outliers por columna.
    """
    outlier_info = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_info.append({
            "columna": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "límite_inferior": round(lower, 2),
            "límite_superior": round(upper, 2),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(df) * 100, 2),
        })
    return pd.DataFrame(outlier_info)


def cap_outliers(df: pd.DataFrame, columns: list,
                 lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Aplica capping de outliers usando percentiles.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Columnas numéricas.
    lower_pct, upper_pct : float
        Percentiles para el capping.

    Returns
    -------
    pd.DataFrame
        DataFrame con outliers capados.
    """
    df = df.copy()
    for col in columns:
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lower, upper)
    return df


def build_preprocessing_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Construye un ColumnTransformer con pipelines para features
    numéricas y categóricas.

    Numéricas: Imputación (mediana) → Escalado (StandardScaler)
    Categóricas: Imputación (moda) → Encoding (OrdinalEncoder)

    Parameters
    ----------
    num_cols : list
    cat_cols : list

    Returns
    -------
    ColumnTransformer
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def load_and_split_data(
    df: pd.DataFrame,
    target_col: str = "GOLD_stage",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Divide el dataset en train/val/test con estratificación.

    Split: 70% train / 15% val / 15% test

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    test_size, val_size : float
    random_state : int

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target si es string
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Primer split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size,
        random_state=random_state, stratify=y_encoded,
    )

    # Segundo split: train vs val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative,
        random_state=random_state, stratify=y_temp,
    )

    print(f"OK Split completado:")
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Val:   {X_val.shape[0]} muestras")
    print(f"   Test:  {X_test.shape[0]} muestras")

    return X_train, X_val, X_test, y_train, y_val, y_test, le


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                        feature_names, preprocessor, label_encoder):
    """
    Guarda los datos procesados y el pipeline en disco.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, X, y in [("train", X_train, y_train),
                        ("val", X_val, y_val),
                        ("test", X_test, y_test)]:
        df_out = pd.DataFrame(X, columns=feature_names)
        df_out["target"] = y
        df_out.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessing_pipeline.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    print(f"OK Datos guardados en {DATA_PROCESSED}")
    print(f"OK Pipeline guardado en {MODELS_DIR}")
