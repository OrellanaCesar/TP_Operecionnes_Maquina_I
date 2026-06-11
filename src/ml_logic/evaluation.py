"""
Shared evaluation utilities for binary classification models.

Provides consistent metrics and plots across all team members.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible estimator.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    model_name : str
        Label for display purposes.

    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, f1, and roc_auc.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # AUC requires predict_proba or decision_function
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, y_scores)
    else:
        metrics["roc_auc"] = None

    return metrics


def print_classification_report(model, X_test, y_test) -> None:
    """Print sklearn classification report."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def plot_confusion_matrix(model, X_test, y_test, ax=None) -> None:
    """Plot a confusion matrix heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()


def plot_roc_curve(model, X_test, y_test, ax=None) -> None:
    """Plot ROC curve with AUC score."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    if not (hasattr(model, "predict_proba") or hasattr(model, "decision_function")):
        ax.text(
            0.5,
            0.5,
            "ROC curve not available\n(model has no score method)",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_title("ROC Curve")
        ax.set_axis_off()
        return
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curve")
    plt.tight_layout()


def plot_full_evaluation(model, X_test, y_test, model_name: str = "Model"):
    """Generate a complete evaluation dashboard: metrics + confusion matrix + ROC.

    Returns the metrics dict.
    """
    metrics = evaluate_model(model, X_test, y_test, model_name=model_name)

    print(f"=== {model_name} ===")
    print_classification_report(model, X_test, y_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(model, X_test, y_test, ax=axes[0])
    plot_roc_curve(model, X_test, y_test, ax=axes[1])
    fig.suptitle(model_name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return metrics


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of metric dicts.

    Parameters
    ----------
    metrics_list : list[dict]
        Each dict is the output of evaluate_model().

    Returns
    -------
    pd.DataFrame
        Sorted by F1 descending.
    """
    df = pd.DataFrame(metrics_list)
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df
