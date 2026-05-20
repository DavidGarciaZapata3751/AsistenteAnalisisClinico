# %% [markdown]
# # 📊 Notebook 01: Análisis Exploratorio de Datos (EDA)
# ## Dataset: COPD GOLD — 230 Pacientes
#
# **Objetivo:** Realizar un análisis exploratorio profundo del dataset clínico
# de pacientes con COPD, incluyendo manejo de valores nulos, detección de
# outliers, análisis de imbalance de clases, y al menos 4 visualizaciones
# informativas.

# %%
# ── Imports ─────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Configurar rutas del proyecto
PROJECT_ROOT = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_raw_data, get_feature_groups, detect_outliers_iqr

# Configuración de estilo de gráficos
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.family"] = "sans-serif"

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#2CA58D",
    "danger": "#C73E1D",
}
GOLD_PALETTE = ["#2E86AB", "#F18F01", "#A23B72", "#C73E1D"]

print("✅ Librerías cargadas correctamente")

# %% [markdown]
# ## 1. Carga y primera inspección del dataset

# %%
# Cargar dataset
df = load_raw_data("230PatientsCOPD.xlsx")

# %%
# Primeras filas
print("📋 Primeras 5 filas del dataset:")
print(df.head())

# %%
# Información general
print("📋 Información del dataset:")
print(f"   Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"   Memoria: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print()
df.info()

# %%
# Tipos de datos
print("\n📋 Tipos de datos por columna:")
print(df.dtypes.value_counts())
print()

# Columnas
print("📋 Lista de columnas:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col} ({df[col].dtype})")

# %%
# Estadísticas descriptivas - Variables numéricas
print("📋 Estadísticas descriptivas (numéricas):")
print(df.describe().T.round(2))

# %%
# Estadísticas descriptivas - Variables categóricas
cat_desc = df.describe(include=["object", "category", "bool"])
if not cat_desc.empty:
    print("📋 Estadísticas descriptivas (categóricas):")
    print(cat_desc.T)

# %% [markdown]
# ## 2. Identificación de la variable objetivo
#
# El dataset COPD GOLD clasifica pacientes en **4 niveles de severidad (GOLD 1-4)**.
# Identificaremos la columna objetivo.

# %%
# Detectar la columna objetivo
# Buscar columnas que contengan 'GOLD', 'stage', 'class', 'severity', 'label', 'target'
target_candidates = [c for c in df.columns
                     if any(kw in c.lower() for kw in
                            ["gold", "stage", "class", "severity", "label", "target", "copd"])]

print("🎯 Posibles columnas objetivo:")
for col in target_candidates:
    print(f"   → {col}: {df[col].nunique()} valores únicos")
    print(f"     Distribución: {df[col].value_counts().to_dict()}")
    print()

# Seleccionar la columna objetivo (ajustar según el dataset real)
# Intentar identificar automáticamente
TARGET_COL = None
for col in target_candidates:
    if df[col].nunique() <= 5 and df[col].nunique() >= 2:
        TARGET_COL = col
        break

if TARGET_COL is None:
    # Fallback: usar la última columna
    TARGET_COL = df.columns[-1]

print(f"✅ Variable objetivo seleccionada: '{TARGET_COL}'")
print(f"   Clases: {sorted(df[TARGET_COL].unique())}")

# %%
# Separar features
num_cols, cat_cols = get_feature_groups(df, TARGET_COL)
print(f"\n📊 Features numéricas ({len(num_cols)}):")
for col in num_cols:
    print(f"   • {col}")
print(f"\n📊 Features categóricas ({len(cat_cols)}):")
for col in cat_cols:
    print(f"   • {col}")

# %% [markdown]
# ## 3. Análisis de Valores Nulos

# %%
# Conteo de nulos
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_df = pd.DataFrame({"Nulos": null_counts, "Porcentaje (%)": null_pct})
null_df = null_df[null_df["Nulos"] > 0].sort_values("Porcentaje (%)", ascending=False)

if null_df.empty:
    print("✅ No hay valores nulos en el dataset")
else:
    print(f"⚠️ Columnas con valores nulos ({len(null_df)}):")
    print(null_df)

# %%
# Heatmap de nulidad
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(
    df.isnull().T,
    cbar=True,
    cmap="YlOrRd",
    yticklabels=True,
    ax=ax,
    cbar_kws={"label": "Nulo (1) / No nulo (0)", "shrink": 0.6},
)
ax.set_title("Mapa de Calor de Valores Nulos", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Muestras (pacientes)", fontsize=11)
ax.set_ylabel("Variables", fontsize=11)
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_01_nulls_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 4. Detección de Outliers (Método IQR)

# %%
# Detección de outliers por IQR
if num_cols:
    outlier_df = detect_outliers_iqr(df, num_cols)
    print("📊 Detección de Outliers (IQR × 1.5):")
    outlier_df_sorted = outlier_df.sort_values("n_outliers", ascending=False)
    print(outlier_df_sorted.to_string(index=False))

# %%
# Boxplots de features numéricas para visualizar outliers
if num_cols:
    n_num = len(num_cols)
    n_cols_plot = min(4, n_num)
    n_rows_plot = (n_num + n_cols_plot - 1) // n_cols_plot

    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                              figsize=(4 * n_cols_plot, 4 * n_rows_plot))
    axes = np.array(axes).flatten() if n_num > 1 else [axes]

    for i, col in enumerate(num_cols):
        if i < len(axes):
            sns.boxplot(data=df, y=col, ax=axes[i], color=COLORS["primary"],
                       width=0.4, flierprops={"marker": "o", "markerfacecolor": COLORS["danger"]})
            axes[i].set_title(col, fontsize=10, fontweight="bold")
            axes[i].set_ylabel("")

    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Boxplots — Detección de Outliers por Feature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_02_boxplots_outliers.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## 5. 📊 Visualización 1: Balance de la Clase Objetivo

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Barplot
class_counts = df[TARGET_COL].value_counts().sort_index()
bars = ax1.bar(class_counts.index.astype(str), class_counts.values,
               color=GOLD_PALETTE[:len(class_counts)],
               edgecolor="white", linewidth=1.5)

# Añadir etiquetas
for bar, count in zip(bars, class_counts.values):
    pct = count / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

ax1.set_xlabel("Clase GOLD", fontsize=12)
ax1.set_ylabel("Número de pacientes", fontsize=12)
ax1.set_title("Distribución de Clases GOLD", fontsize=13, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# Pieplot
wedges, texts, autotexts = ax2.pie(
    class_counts.values,
    labels=[f"GOLD {c}" for c in class_counts.index],
    autopct="%1.1f%%",
    colors=GOLD_PALETTE[:len(class_counts)],
    startangle=90,
    explode=[0.05] * len(class_counts),
    shadow=True,
    textprops={"fontsize": 11},
)
ax2.set_title("Proporción de Clases", fontsize=13, fontweight="bold")

plt.suptitle("⚖️ Análisis de Balance de Clases", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_03_class_balance.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# Ratio de desbalance
majority = class_counts.max()
minority = class_counts.min()
print(f"\n📊 Ratio de desbalance (mayoría/minoría): {majority/minority:.2f}:1")

# %% [markdown]
# ## 6. 📊 Visualización 2: Distribución de Edades por Clase GOLD

# %%
# Buscar columna de edad
age_col = None
for col in df.columns:
    if "age" in col.lower() or "edad" in col.lower():
        age_col = col
        break

if age_col is None:
    # Usar la primera columna numérica como proxy
    age_col = num_cols[0] if num_cols else None

if age_col:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma con KDE por clase
    for i, cls in enumerate(sorted(df[TARGET_COL].unique())):
        subset = df[df[TARGET_COL] == cls][age_col].dropna()
        ax1.hist(subset, bins=15, alpha=0.5, color=GOLD_PALETTE[i % len(GOLD_PALETTE)],
                 label=f"GOLD {cls}", edgecolor="white")

    ax1.set_xlabel(age_col, fontsize=12)
    ax1.set_ylabel("Frecuencia", fontsize=12)
    ax1.set_title(f"Distribución de {age_col} por Clase", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Violin plot
    sns.violinplot(
        data=df, x=TARGET_COL, y=age_col,
        palette=GOLD_PALETTE[:df[TARGET_COL].nunique()],
        ax=ax2, inner="quartile",
    )
    ax2.set_xlabel("Clase GOLD", fontsize=12)
    ax2.set_ylabel(age_col, fontsize=12)
    ax2.set_title(f"Violin Plot: {age_col} por Clase", fontsize=13, fontweight="bold")

    plt.suptitle(f"📊 Análisis de {age_col} por Severidad COPD",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_04_age_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("⚠️ No se encontró columna de edad")

# %% [markdown]
# ## 7. 📊 Visualización 3: Matriz de Correlación

# %%
if len(num_cols) > 1:
    # Calcular correlación solo para features numéricas
    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Correlación de Pearson"},
        annot_kws={"size": 8},
        vmin=-1, vmax=1,
    )
    ax.set_title("Matriz de Correlación — Features Numéricas",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_05_correlation_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Correlaciones más altas con la variable objetivo (si es numérica)
    if df[TARGET_COL].dtype in ["int64", "float64"]:
        target_corr = df[num_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
        target_corr = target_corr.abs().sort_values(ascending=False)
        print("\n📊 Top correlaciones con la variable objetivo:")
        for col, val in target_corr.head(10).items():
            print(f"   {col}: {val:.3f}")

# %% [markdown]
# ## 8. 📊 Visualización 4: Factores Ambientales vs Lifestyle

# %%
# Seleccionar subset de features más relevantes para visualización
if len(num_cols) >= 4:
    # Tomar las 6 features numéricas más relevantes
    top_features = num_cols[:6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        if i < len(axes):
            sns.boxplot(
                data=df, x=TARGET_COL, y=col,
                palette=GOLD_PALETTE[:df[TARGET_COL].nunique()],
                ax=axes[i], width=0.5,
            )
            axes[i].set_title(col, fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Clase GOLD")
            axes[i].grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("📊 Distribución de Features Principales por Clase GOLD",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_06_features_by_class.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## 9. 📊 Visualización 5 (Bonus): Pairplot de Features Principales

# %%
if len(num_cols) >= 3:
    # Seleccionar top 4 features para pairplot
    pair_features = num_cols[:4] + [TARGET_COL]

    g = sns.pairplot(
        df[pair_features].dropna(),
        hue=TARGET_COL,
        palette=GOLD_PALETTE[:df[TARGET_COL].nunique()],
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "white"},
        height=2.5,
    )
    g.figure.suptitle("Pairplot — Features Principales por Clase GOLD",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_07_pairplot.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

# %% [markdown]
# ## 10. Análisis Estadístico Adicional

# %%
# Test estadístico: ¿Las medias difieren significativamente entre clases?
if num_cols:
    print("📊 Test Kruskal-Wallis (H) por feature vs clase objetivo:")
    print("   (p < 0.05 indica diferencia significativa entre grupos)\n")
    kw_results = []
    for col in num_cols:
        groups = [group[col].dropna().values
                  for _, group in df.groupby(TARGET_COL)]
        if all(len(g) > 0 for g in groups):
            stat, p_value = stats.kruskal(*groups)
            sig = "✅ Sí" if p_value < 0.05 else "❌ No"
            kw_results.append({
                "Feature": col,
                "H-statistic": round(stat, 3),
                "p-value": round(p_value, 4),
                "Significativo": sig,
            })
    kw_df = pd.DataFrame(kw_results).sort_values("p-value")
    print(kw_df.to_string(index=False))

# %% [markdown]
# ## 11. Resumen del EDA
#
# ### Hallazgos clave:
# 1. **Dataset**: 230 pacientes con múltiples features ambientales y de estilo de vida
# 2. **Variable objetivo**: Clasificación GOLD (severidad COPD)
# 3. **Valores nulos**: Documentados arriba
# 4. **Outliers**: Detectados mediante método IQR
# 5. **Balance de clases**: Analizado con métricas de ratio
# 6. **Correlaciones**: Matriz de correlación completa calculada
#
# → El siguiente paso es el preprocesamiento en `02_preprocessing.py`

# %%
print("=" * 60)
print("  ✅ EDA COMPLETADO")
print("  → Siguiente: 02_preprocessing.py")
print("=" * 60)
