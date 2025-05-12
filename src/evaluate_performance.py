import json
import logging
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, roc_auc_score
import pandas as pd

# Logger configuration
logger = logging.getLogger("evaluator")

def evaluate_performance(scores_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute accuracy, F1 score, and ROC AUC from score data.

    Args:
        scores_df: DataFrame with 'y_true', 'y_pred', 'y_proba'.
        config: Config dict (unused but kept for consistency).

    Returns:
        Dict of evaluation metrics.
    """
    try:
        logger.info("Evaluating performance.")
        metrics = {
            "accuracy": accuracy_score(scores_df["y_true"], scores_df["y_pred"]),
            "f1": f1_score(scores_df["y_true"], scores_df["y_pred"], average="macro"),
            "roc_auc": roc_auc_score(scores_df["y_true"], scores_df["y_proba"]),
        }
        logger.debug("Evaluation metrics: %s", metrics)
        return metrics
    except Exception as e:
        logger.exception("Error computing performance metrics.")
        raise RuntimeError(f"Evaluation failed: {e}")


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of metric scores.
        path: File path to save the JSON.
    """
    try:
        logger.info("Saving metrics to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved.")
    except Exception as e:
        logger.exception("Failed to save metrics.")
        raise IOError(f"Could not save metrics: {e}")


def plot_roc_curve(scores_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plot and save the ROC curve from prediction scores.

    Args:
        scores_df: DataFrame with 'y_true' and 'y_proba' columns.
        save_path: Path to save the ROC curve image.
    """
    try:
        logger.info("Plotting ROC curve.")
        RocCurveDisplay.from_predictions(
            y_true=scores_df["y_true"],
            y_pred=scores_df["y_proba"]
        )
        plt.title("ROC Curve")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info("ROC curve saved to %s", save_path)
    except Exception as e:
        logger.exception("Failed to plot/save ROC curve.")
        raise RuntimeError(f"ROC plot failed: {e}")
    