# %% [markdown]
# # 🧠 Notebook 03: Modelado de Machine Learning
# ## Clasificación de Severidad COPD GOLD
#
# **Objetivo:** Entrenar y evaluar modelos de clasificación (Baseline,
# Random Forest, XGBoost y un MLP con PyTorch) usando el dataset preprocesado.
# Evaluar con Precision, Recall, F1-Score y Matrices de Confusión.

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

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import torch

warnings.filterwarnings("ignore")

# Configurar rutas
PROJECT_ROOT = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import (
    MLPClassifier,
    create_dataloaders,
    train_mlp,
    predict_mlp,
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    compare_models,
)
from src.preprocessing import DATA_PROCESSED, MODELS_DIR

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 6)

# Dispositivo para PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Librerías cargadas. Usando dispositivo: {device}")

# %% [markdown]
# ## 1. Carga de Datos Procesados

# %%
print("🔄 Cargando datos procesados...")
try:
    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    val_df = pd.read_csv(DATA_PROCESSED / "val.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    
    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    
    X_val = val_df.drop(columns=["target"]).values
    y_val = val_df["target"].values
    
    X_test = test_df.drop(columns=["target"]).values
    y_test = test_df["target"].values
    
    feature_names = train_df.drop(columns=["target"]).columns.tolist()
    class_names = [f"GOLD {c}" for c in label_encoder.classes_]
    n_classes = len(class_names)
    
    print(f"✅ Datos cargados exitosamente:")
    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")
    print(f"   Clases ({n_classes}): {class_names}")

except FileNotFoundError:
    print("❌ ERROR: No se encontraron los datos procesados. Ejecuta 02_preprocessing.py primero.")
    sys.exit(1)

# %% [markdown]
# ## 2. Manejo de Desbalance (SMOTE en Train)

# %%
print("🔄 Aplicando SMOTE para balancear clases en el conjunto de entrenamiento...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"   Train original:  {np.bincount(y_train)}")
print(f"   Train balanceado: {np.bincount(y_train_res)}")

# Preparar lista para almacenar resultados
all_results = []

# %% [markdown]
# ## 3. Baseline Model (Dummy Classifier)

# %%
print("🔄 Entrenando Baseline (Stratified)...")
baseline = DummyClassifier(strategy="stratified", random_state=42)
baseline.fit(X_train_res, y_train_res)

# Predecir en TEST (no en validación para comparación final)
y_pred_base = baseline.predict(X_test)

res_base = evaluate_model(y_test, y_pred_base, "Baseline", class_names)
all_results.append(res_base)

fig, ax = plt.subplots()
plot_confusion_matrix(y_test, y_pred_base, class_names, "Matriz de Confusión - Baseline", ax)
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_cm_baseline.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 4. Random Forest Classifier

# %%
print("🔄 Entrenando Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)

y_pred_rf = rf_model.predict(X_test)

res_rf = evaluate_model(y_test, y_pred_rf, "Random Forest", class_names)
all_results.append(res_rf)

fig, ax = plt.subplots()
plot_confusion_matrix(y_test, y_pred_rf, class_names, "Matriz de Confusión - Random Forest", ax)
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_cm_rf.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 5. XGBoost Classifier

# %%
print("🔄 Entrenando XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=n_classes,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
)

# Entrenar usando validation set para early stopping (conceptualmente)
xgb_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_pred_xgb = xgb_model.predict(X_test)

res_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost", class_names)
all_results.append(res_xgb)

fig, ax = plt.subplots()
plot_confusion_matrix(y_test, y_pred_xgb, class_names, "Matriz de Confusión - XGBoost", ax)
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_cm_xgb.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 6. Multi-Layer Perceptron (PyTorch)

# %%
print("🔄 Preparando datos para PyTorch...")
train_loader, val_loader = create_dataloaders(
    X_train_res, y_train_res, X_val, y_val, batch_size=32
)

# Inicializar modelo
input_dim = X_train.shape[1]
mlp_model = MLPClassifier(input_dim=input_dim, n_classes=n_classes, dropout=0.3)
print(f"   Arquitectura MLP:\n{mlp_model}")

# %%
print("\n🔄 Entrenando MLP...")
# Calcular pesos de clase (inverso de frecuencias en val_df original) para focalizar
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
weights = weights / weights.sum() * n_classes

history = train_mlp(
    model=mlp_model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_classes=n_classes,
    epochs=150,
    lr=0.001,
    patience=20,
    class_weights=weights,
    device=device
)

# %%
# Visualizar entrenamiento
plot_training_history(history, "Entrenamiento MLP PyTorch")
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_09_mlp_history.png"), dpi=150)
plt.close()

# %%
print("\n🔄 Evaluando MLP en Test...")
y_pred_mlp = predict_mlp(mlp_model, X_test, device)

res_mlp = evaluate_model(y_test, y_pred_mlp, "MLP (PyTorch)", class_names)
all_results.append(res_mlp)

fig, ax = plt.subplots()
plot_confusion_matrix(y_test, y_pred_mlp, class_names, "Matriz de Confusión - MLP", ax)
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_cm_mlp.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 7. Comparación y Feature Importance

# %%
# Tabla comparativa
compare_df = compare_models(all_results)

# Visualizar F1-Weighted de todos los modelos
fig, ax = plt.subplots(figsize=(8, 4))
scores = [float(r["f1_weighted"]) for r in all_results]
names = [r["modelo"] for r in all_results]

sns.barplot(x=scores, y=names, palette="viridis", ax=ax)
for i, v in enumerate(scores):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    
ax.set_title("Comparación de F1-Score (Weighted)", fontweight="bold")
ax.set_xlim(0, 1.05)
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_10_model_comparison.png"), dpi=150)
plt.close()

# %%
# Feature Importance de Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=importances[indices][:15], 
    y=[feature_names[i] for i in indices][:15],
    palette="rocket",
    ax=ax
)
ax.set_title("Top 15 Features más importantes (Random Forest)", fontweight="bold")
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "notebooks" / "fig_11_feature_importance.png"), dpi=150)
plt.close()

# %% [markdown]
# ## 8. Guardar el mejor modelo

# %%
# Seleccionar el mejor modelo basado en f1_weighted
best_model_name = compare_df["f1_weighted"].astype(float).idxmax()
print(f"\n🏆 Mejor modelo: {best_model_name}")

# En un caso real guardaríamos el mejor. Aquí guardamos Random Forest y MLP
joblib.dump(rf_model, MODELS_DIR / "rf_model.joblib")
joblib.dump(xgb_model, MODELS_DIR / "xgb_model.joblib")
torch.save(mlp_model.state_dict(), MODELS_DIR / "mlp_weights.pth")

print(f"✅ Modelos guardados en {MODELS_DIR}")

# %%
print("=" * 60)
print("  ✅ MODELADO COMPLETADO")
print("  → Siguiente: 04_llm_rag.py")
print("=" * 60)
