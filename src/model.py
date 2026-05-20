"""
src/model.py
============
Módulo de modelado para clasificación de severidad COPD GOLD.
Incluye definición de MLP en PyTorch, funciones de entrenamiento,
evaluación y comparación de modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ── MLP en PyTorch ──────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Red neuronal MLP para clasificación multiclase de datos tabulares.

    Arquitectura:
        Input → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
              → Dense(32, ReLU)  → Dropout(0.2)
              → Dense(n_classes)
    """

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),

            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.network(x)


def create_dataloaders(X_train, y_train, X_val, y_val,
                       batch_size: int = 32):
    """
    Crea DataLoaders de PyTorch a partir de arrays NumPy.
    """
    train_ds = TensorDataset(
        torch.FloatTensor(np.array(X_train, dtype=np.float32)),
        torch.LongTensor(np.array(y_train, dtype=np.int64)),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(np.array(X_val, dtype=np.float32)),
        torch.LongTensor(np.array(y_val, dtype=np.int64)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    class_weights: np.ndarray = None,
    device: str = "cpu",
):
    """
    Entrena el MLP con early stopping.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    n_classes : int
    epochs : int
    lr : float
    patience : int
        Épocas sin mejora antes de detener.
    class_weights : np.ndarray, optional
        Pesos por clase para el loss.
    device : str

    Returns
    -------
    dict
        Historial de entrenamiento con train_loss, val_loss, val_acc.
    """
    model = model.to(device)

    if class_weights is not None:
        weight = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ──
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} │ "
                f"Train Loss: {train_loss:.4f} │ "
                f"Val Loss: {val_loss:.4f} │ "
                f"Val Acc: {val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"  ⏹ Early stopping en epoch {epoch+1}")
            break

    # Restaurar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_mlp(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Genera predicciones con el modelo MLP."""
    model.eval()
    model = model.to(device)
    X_tensor = torch.FloatTensor(np.array(X, dtype=np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
    return preds


# ── Evaluación ──────────────────────────────────────────────────────────────

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "Modelo",
                   class_names: list = None) -> dict:
    """
    Evalúa un modelo con métricas completas.

    Returns
    -------
    dict
        Diccionario con accuracy, precision, recall, f1 (macro y weighted).
    """
    metrics = {
        "modelo": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print(f"\n{'='*60}")
    print(f"  📊 Evaluación: {model_name}")
    print(f"{'='*60}")
    target_names = class_names if class_names else None
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list = None,
                          title: str = "Matriz de Confusión",
                          ax=None):
    """
    Visualiza la matriz de confusión como heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    labels = class_names if class_names else [f"Clase {i}" for i in range(cm.shape[0])]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"shrink": 0.8},
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    return ax


def plot_training_history(history: dict, title: str = "Historial de Entrenamiento"):
    """Grafica curvas de loss y accuracy durante el entrenamiento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Pérdida durante entrenamiento")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_acc"], label="Val Accuracy", linewidth=2, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy en validación")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def compare_models(results: list) -> pd.DataFrame:
    """
    Crea tabla comparativa de rendimiento entre modelos.

    Parameters
    ----------
    results : list[dict]
        Lista de diccionarios con métricas (output de evaluate_model).

    Returns
    -------
    pd.DataFrame
        Tabla formateada con comparación.
    """
    df = pd.DataFrame(results)
    df = df.set_index("modelo")
    # Formatear como porcentaje
    for col in df.columns:
        df[col] = df[col].map(lambda x: f"{x:.4f}")
    print("\n📊 Comparación de Modelos:")
    print(df.to_string())
    return df
