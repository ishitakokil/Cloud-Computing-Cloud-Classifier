import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Logger configuration
logger = logging.getLogger("feature_generator")

def generate_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate new features using normalization, log transforms, and multiplications.

    Args:
        df: Input DataFrame.
        config: Dict with keys for 'calculate_norm_range', 'log_transform', and 'multiply'.

    Returns:
        DataFrame with new features.
    """
    try:
        df = df.copy()
        df = df.apply(pd.to_numeric, errors="coerce")

        # Normalized range
        if "calculate_norm_range" in config:
            for new_col, params in config["calculate_norm_range"].items():
                df[new_col] = (df[params["max_col"]] - df[params["min_col"]]) / (df[params["mean_col"]] + 1e-5)

        # Log transforms
        if "log_transform" in config:
            for new_col, source_col in config["log_transform"].items():
                df[new_col] = np.log(df[source_col] + 1e-5)

        # Multiplications
        if "multiply" in config:
            for new_col, mult in config["multiply"].items():
                df[new_col] = df[mult["col_a"]] * df[mult["col_b"]]

        logger.info("Generated features. Final shape: %s", df.shape)
        return df

    except KeyError as e:
        logger.error("Missing key in config or dataframe: %s", e)
        raise ValueError(f"Missing key in config or dataframe: {e}")


def generate_labels(df: pd.DataFrame, method: str = "threshold", config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Generate cloud type labels using threshold or KMeans clustering.

    Args:
        df: Input DataFrame.
        method: 'threshold' or 'kmeans'.
        config: Config dict with 'threshold' or 'n_clusters' as needed.

    Returns:
        DataFrame with a new 'cloud_type' column.
    """
    try:
        if method == "threshold":
            threshold = config.get("threshold", 200)
            df["cloud_type"] = (df["IR_mean"] > threshold).astype(int)
            logger.info("Labels generated using threshold method.")

        elif method == "kmeans":
            n_clusters = config.get("n_clusters", 2)
            features = df.select_dtypes(include=["number"]).dropna()
            df = df.loc[features.index].copy()
            model = KMeans(n_clusters=n_clusters, random_state=42)
            df["cloud_type"] = model.fit_predict(features)
            logger.info("Labels generated using KMeans clustering.")

        else:
            raise ValueError(f"Unknown labeling method: {method}")

        return df

    except Exception as e:
        logger.exception("Failed to generate labels.")
        raise RuntimeError(f"Label generation failed: {e}")
