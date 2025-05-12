from pathlib import Path
from typing import Tuple, Dict, Any
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
import joblib


# Logging Configuration
logger = logging.getLogger("model_trainer")

def train_model(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[ClassifierMixin, pd.DataFrame, pd.DataFrame]:
    """
    Train a RandomForestClassifier using provided data and config.

    Args:
        df: Dataset containing features and target.
        config: Dict with 'target_column', 'test_size', and model 'params'.

    Returns:
        Tuple of (trained model, train DataFrame, test DataFrame).
    """
    try:
        logger.info("Starting model training...")
        X = df.drop(columns=[config["target_column"]])
        y = df[config["target_column"]]

        logger.debug("Splitting data.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=config["params"].get("random_state", 42),
        )

        logger.debug("Initializing model.")
        model = RandomForestClassifier(**config["params"])
        model.fit(X_train, y_train)
        logger.info("Model training done.")

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        return model, train_df, test_df

    except KeyError as e:
        logger.error("Missing config key: %s", e)
        raise ValueError(f"Missing config key: {e}")
    except Exception as e:
        logger.exception("Error during training.")
        raise RuntimeError(f"Training failed: {e}")


def save_data(train: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> None:
    """
    Save train and test DataFrames to CSV files.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        out_dir: Directory to save the files.
    """
    try:
        logger.info("Saving datasets to %s", out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(out_dir / "train.csv", index=False)
        test.to_csv(out_dir / "test.csv", index=False)
        logger.info("Datasets saved.")
    except Exception as e:
        logger.exception("Saving data failed.")
        raise IOError(f"Save error: {e}")


def save_model(model: ClassifierMixin, path: Path) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model to save.
        path: Destination file path.
    """
    try:
        logger.info("Saving model to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info("Model saved.")
    except Exception as e:
        logger.exception("Saving model failed.")
        raise IOError(f"Model save error: {e}")
