# %% [markdown]
# # 🔧 Notebook 02: Preprocesamiento
# ## Pipeline de Limpieza Reproducible con scikit-learn
#
# **Objetivo:** Construir un pipeline de preprocesamiento modular y reproducible
# para el dataset COPD GOLD, incluyendo imputación, encoding, escalado,
# manejo de outliers y división train/val/test estratificada.

# %%
# ── Imports ─────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Configurar rutas del proyecto
PROJECT_ROOT = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_raw_data,
    get_feature_groups,
    detect_outliers_iqr,
    cap_outliers,
    build_preprocessing_pipeline,
    load_and_split_data,
    save_processed_data,
    DATA_RAW,
    DATA_PROCESSED,
    MODELS_DIR,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
print("✅ Librerías cargadas correctamente")

# %% [markdown]
# ## 1. Carga de Datos

# %%
# Cargar dataset principal
df = load_raw_data("230PatientsCOPD.xlsx")
print(f"\nDimensiones originales: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")

# %% [markdown]
# ## 2. Limpieza de Nombres de Columnas

# %%
# Limpiar nombres de columnas (quitar whitespace, saltos de línea)
df.columns = df.columns.str.strip().str.replace("\n", "")
print("✅ Nombres de columnas limpiados:")
for col in df.columns:
    print(f"   • {col}")

# %% [markdown]
# ## 3. Eliminar Columnas Innecesarias

# %%
# Eliminar columna de ID (no es feature predictiva)
id_cols = [c for c in df.columns if "id" in c.lower() or "number" in c.lower()]
if id_cols:
    print(f"🗑️ Eliminando columnas de ID: {id_cols}")
    df = df.drop(columns=id_cols)

# Eliminar 'Location' si existe (puede ser identificador)
if "Location" in df.columns:
    print(f"🗑️ Eliminando 'Location' (potencial leak de información)")
    df = df.drop(columns=["Location"])

print(f"\nDimensiones tras limpieza: {df.shape}")

# %% [markdown]
# ## 4. Manejo de Valores Nulos

# %%
# Verificar nulos
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_summary = pd.DataFrame({"Nulos": null_counts, "%": null_pct})
null_summary = null_summary[null_summary["Nulos"] > 0].sort_values("%", ascending=False)

print("📋 Resumen de valores nulos:")
if null_summary.empty:
    print("   ✅ No hay valores nulos")
else:
    print(null_summary)

# %%
# Eliminar filas completamente nulas (las 10 filas vacías del dataset)
rows_before = len(df)
df = df.dropna(how="all")
rows_dropped = rows_before - len(df)
print(f"🗑️ Eliminadas {rows_dropped} filas completamente nulas")
print(f"   Dimensiones: {df.shape}")

# Verificar nulos restantes
remaining_nulls = df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if remaining_nulls.empty:
    print("✅ Sin valores nulos restantes")
else:
    print(f"⚠️ Nulos restantes:\n{remaining_nulls}")

# %% [markdown]
# ## 5. Corrección de Tipos de Datos

# %%
# Definir variable objetivo
TARGET_COL = "COPD GOLD"
print(f"🎯 Variable objetivo: '{TARGET_COL}'")
print(f"   Distribución: {df[TARGET_COL].value_counts().sort_index().to_dict()}")

# %%
# Convertir COPD GOLD a entero (es float por los NaN)
df[TARGET_COL] = df[TARGET_COL].astype(int)
print(f"✅ '{TARGET_COL}' convertida a int: {df[TARGET_COL].unique()}")

# %%
# Inspeccionar columnas 'object' que deberían ser numéricas
object_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\n📋 Columnas tipo 'object' ({len(object_cols)}):")
for col in object_cols:
    sample = df[col].dropna().head(5).tolist()
    print(f"   • {col}: {sample}")

# %%
# Intentar extraer valores numéricos de columnas de texto si es posible
# Si no es posible, se mantienen como categóricas
print("Manteniendo columnas de texto como categóricas en lugar de forzarlas a NaN")

# %%
# Reclasificar features
num_cols, cat_cols = get_feature_groups(df, TARGET_COL)
print(f"\n📊 Features numéricas ({len(num_cols)}): {num_cols}")
print(f"📊 Features categóricas ({len(cat_cols)}): {cat_cols}")

# %% [markdown]
# ## 6. Encoding de Variables Categóricas

# %%
# Analizar variables categóricas
for col in cat_cols:
    print(f"\n📋 {col}:")
    print(f"   Valores únicos: {df[col].nunique()}")
    print(f"   Distribución: {df[col].value_counts().to_dict()}")

# %%
# Encoding manual para variables categóricas binarias/ordinales conocidas
encoding_maps = {}

# Gender: Male/Female → 0/1
if "Gender" in df.columns:
    gender_map = {"Male": 0, "Female": 1}
    df["Gender"] = df["Gender"].str.strip().map(gender_map)
    encoding_maps["Gender"] = gender_map
    print(f"✅ Gender encoded: {gender_map}")

# History of Heart Failure: Yes/No → 1/0
if "History of Heart Failure" in df.columns:
    yn_map = {"No": 0, "Yes": 1}
    df["History of Heart Failure"] = df["History of Heart Failure"].str.strip().map(yn_map)
    encoding_maps["History of Heart Failure"] = yn_map
    print(f"✅ History of Heart Failure encoded: {yn_map}")

# Vaccination: Yes/No → 1/0
if "Vaccination" in df.columns:
    df["Vaccination"] = df["Vaccination"].str.strip().map({"No": 0, "Yes": 1})
    encoding_maps["Vaccination"] = {"No": 0, "Yes": 1}
    print(f"✅ Vaccination encoded")

# Depression: Yes/No → 1/0
if "Depression" in df.columns:
    df["Depression"] = df["Depression"].str.strip().map({"No": 0, "Yes": 1})
    encoding_maps["Depression"] = {"No": 0, "Yes": 1}
    print(f"✅ Depression encoded")

# Dependent: Yes/No → 1/0
if "Dependent" in df.columns:
    df["Dependent"] = df["Dependent"].str.strip().map({"No": 0, "Yes": 1})
    encoding_maps["Dependent"] = {"No": 0, "Yes": 1}
    print(f"✅ Dependent encoded")

# Sputum
if "Sputum" in df.columns and df["Sputum"].dtype == "object":
    df["Sputum"] = df["Sputum"].str.strip().map({"No": 0, "Yes": 1})
    encoding_maps["Sputum"] = {"No": 0, "Yes": 1}
    print(f"✅ Sputum encoded")

# Blood pressure
if "Blood pressure" in df.columns and df["Blood pressure"].dtype == "object":
    # Intentar extraer sistólica o convertir
    bp_unique = df["Blood pressure"].dropna().unique()
    print(f"\n📋 Blood pressure valores: {bp_unique[:10]}")
    # Si contiene '/', extraer sistólica
    if any("/" in str(v) for v in bp_unique):
        df["BP_systolic"] = df["Blood pressure"].str.split("/").str[0].astype(float)
        df["BP_diastolic"] = df["Blood pressure"].str.split("/").str[1].astype(float)
        df = df.drop(columns=["Blood pressure"])
        print("✅ Blood pressure → BP_systolic + BP_diastolic")
    else:
        df["Blood pressure"] = pd.to_numeric(df["Blood pressure"], errors="coerce")

print(f"\n📊 Encoding maps guardados: {list(encoding_maps.keys())}")

# %%
# Verificar que no quedan columnas object
remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
if remaining_obj:
    print(f"⚠️ Columnas object restantes: {remaining_obj}")
    # Aplicar OrdinalEncoder a las restantes
    for col in remaining_obj:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"   ✅ {col} encoded con LabelEncoder")
else:
    print("✅ Todas las columnas son numéricas")

# Reclasificar features
num_cols, cat_cols = get_feature_groups(df, TARGET_COL)
print(f"\n📊 Features finales: {len(num_cols)} numéricas, {len(cat_cols)} categóricas")

# %% [markdown]
# ## 7. Manejo de Outliers

# %%
# Detectar outliers en features numéricas
outlier_df = detect_outliers_iqr(df, num_cols)
print("📊 Detección de Outliers (IQR × 1.5):")
print(outlier_df.sort_values("n_outliers", ascending=False).to_string(index=False))

# %%
# Aplicar capping de outliers (percentil 1% - 99%)
cols_with_outliers = outlier_df[outlier_df["n_outliers"] > 0]["columna"].tolist()
if cols_with_outliers:
    df = cap_outliers(df, cols_with_outliers, lower_pct=0.01, upper_pct=0.99)
    print(f"✅ Outliers capados en {len(cols_with_outliers)} columnas: {cols_with_outliers}")

    # Verificar post-capping
    outlier_df_post = detect_outliers_iqr(df, cols_with_outliers)
    print("\n📊 Outliers post-capping:")
    print(outlier_df_post[["columna", "n_outliers"]].to_string(index=False))
else:
    print("✅ No se detectaron outliers significativos")

# %% [markdown]
# ## 8. Imputación de Valores Nulos Restantes

# %%
# Verificar nulos restantes tras conversiones
remaining_nulls = df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if not remaining_nulls.empty:
    print(f"⚠️ Nulos restantes tras conversiones:")
    print(remaining_nulls)

    # Imputar con mediana para numéricas
    for col in remaining_nulls.index:
        if df[col].dtype in ["float64", "int64"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   ✅ {col} imputado con mediana: {median_val:.2f}")
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   ✅ {col} imputado con moda: {mode_val}")

    print(f"\n✅ Nulos finales: {df.isnull().sum().sum()}")
else:
    print("✅ Sin valores nulos restantes")

# %% [markdown]
# ## 9. División Train / Val / Test (70/15/15)

# %%
# Eliminar filas donde el target sea nulo
df = df.dropna(subset=[TARGET_COL])
print(f"📊 Dataset final: {df.shape[0]} filas × {df.shape[1]} columnas")

# Split estratificado
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_split_data(
    df, target_col=TARGET_COL, test_size=0.15, val_size=0.15, random_state=42
)

# %%
# Verificar distribución de clases en cada split
print("\n📊 Distribución de clases por split:")
for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    unique, counts = np.unique(y, return_counts=True)
    dist = {f"GOLD {label_encoder.inverse_transform([u])[0]}": c
            for u, c in zip(unique, counts)}
    print(f"   {name}: {dist}")

# %% [markdown]
# ## 10. Escalado con Pipeline de scikit-learn

# %%
# Obtener features numéricas y categóricas del set de entrenamiento
feature_cols = [c for c in df.columns if c != TARGET_COL]
num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"📊 Features para pipeline:")
print(f"   Numéricas ({len(num_features)}): {num_features}")
print(f"   Categóricas ({len(cat_features)}): {cat_features}")

# %%
# Construir y ajustar el pipeline
preprocessor = build_preprocessing_pipeline(num_features, cat_features)
preprocessor.fit(X_train)

# Transformar
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Obtener nombres de features transformadas
try:
    feature_names_out = preprocessor.get_feature_names_out()
    # Limpiar prefijos de sklearn (ej. "num__Age" -> "Age")
    feature_names_out = [f.split("__")[-1] for f in feature_names_out]
except AttributeError:
    # Fallback genérico si falla
    feature_names_out = [f"Feature_{i}" for i in range(X_train_processed.shape[1])]

print(f"\n✅ Pipeline ajustado. Features de salida: {len(feature_names_out)}")
print(f"   X_train: {X_train_processed.shape}")
print(f"   X_val:   {X_val_processed.shape}")
print(f"   X_test:  {X_test_processed.shape}")

# %% [markdown]
# ## 11. Guardar Datos Procesados

# %%
# Guardar todo
save_processed_data(
    X_train_processed, X_val_processed, X_test_processed,
    y_train, y_val, y_test,
    feature_names=feature_names_out,
    preprocessor=preprocessor,
    label_encoder=label_encoder,
)

# Guardar encoding maps
joblib.dump(encoding_maps, MODELS_DIR / "encoding_maps.joblib")
print(f"✅ Encoding maps guardados")

# %%
# Verificar archivos guardados
print("\n📁 Archivos generados:")
for f in sorted(DATA_PROCESSED.glob("*.csv")):
    size = f.stat().st_size / 1024
    df_check = pd.read_csv(f)
    print(f"   📄 {f.name}: {df_check.shape} ({size:.1f} KB)")

for f in sorted(MODELS_DIR.glob("*.joblib")):
    size = f.stat().st_size / 1024
    print(f"   🔧 {f.name}: ({size:.1f} KB)")

# %% [markdown]
# ## 12. Visualización Post-Procesamiento

# %%
# Verificar que los datos procesados se ven bien
train_df = pd.read_csv(DATA_PROCESSED / "train.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribución del target en train
target_counts = train_df["target"].value_counts().sort_index()
axes[0].bar(target_counts.index.astype(str), target_counts.values,
            color=["#2E86AB", "#F18F01", "#A23B72", "#C73E1D"][:len(target_counts)])
axes[0].set_title("Distribución del Target (Train)", fontweight="bold")
axes[0].set_xlabel("Clase GOLD")
axes[0].set_ylabel("Conteo")

# Distribución de features escaladas
feature_means = train_df.drop(columns=["target"]).mean()
axes[1].barh(feature_means.index, feature_means.values, color="#2E86AB", alpha=0.7)
axes[1].set_title("Media de Features Escaladas (Train)", fontweight="bold")
axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_08_post_processing.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# %%
print("=" * 60)
print("  ✅ PREPROCESAMIENTO COMPLETADO")
print(f"  📁 Datos en: {DATA_PROCESSED}")
print(f"  🔧 Pipeline en: {MODELS_DIR}")
print("  → Siguiente: 03_modeling.py")
print("=" * 60)
