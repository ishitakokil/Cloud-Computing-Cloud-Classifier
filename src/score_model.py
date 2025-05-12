import logging
from typing import Dict, Any
from pathlib import Path
import pandas as pd

# Logging Configuration
logger = logging.getLogger("model_scorer")

def score_model(test_df: pd.DataFrame, model: Any, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Score the model using test data.

    Args:
        test_df: Test dataset with features and target.
        model: Trained model object.
        config: Dict with 'target_column' key.

    Returns:
        DataFrame with true labels, predictions, and probabilities.
    """
    try:
        logger.info("Scoring the model.")
        X_test = test_df.drop(columns=[config["target_column"]])
        y_test = test_df[config["target_column"]]
        y_pred = model.predict(X_test)

        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        scores = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        })

        logger.info("Scoring completed successfully.")
        return scores

    except KeyError as e:
        logger.error("Missing key in config: %s", e)
        raise ValueError(f"Missing key in config: {e}")
    except Exception as e:
        logger.exception("Error during model scoring.")
        raise RuntimeError(f"Scoring failed: {e}")


def save_scores(df: pd.DataFrame, path: Path) -> None:
    """
    Save scores DataFrame to CSV.

    Args:
        df: Scores DataFrame.
        path: File path to save.
    """
    try:
        logger.info("Saving scores to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Scores saved.")
    except Exception as e:
        logger.exception("Failed to save scores.")
        raise IOError(f"Could not save scores: {e}")
    