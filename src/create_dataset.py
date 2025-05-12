from typing import Dict
from pathlib import Path
import logging
import pandas as pd


# Logger configuration
logger = logging.getLogger("data_loader")



def detect_data_start(data_path: Path, expected_fields: int = 10) -> int:
    """
    Detect the first line in a file containing the expected number of numeric fields.

    Args:
        data_path: Path to the raw data file.
        expected_fields: Number of numeric fields expected per line.

    Returns:
        Line number (0-based) where valid data starts.

    Raises:
        ValueError: If no valid line is found.
    """
    try:
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                fields = line.strip().split()
                if len(fields) == expected_fields and all(
                    field.replace(".", "", 1).replace("-", "", 1).isdigit()
                    for field in fields
                ):
                    logger.info("Detected start of data at line %d", i)
                    return i
        raise ValueError("No valid data line found with expected number of fields.")
    except Exception as e:
        logger.exception("Error detecting data start.")
        raise RuntimeError(f"Data start detection failed: {e}")


def create_dataset(data_path: Path, config: Dict[str, any]) -> pd.DataFrame:
    """
    Create a DataFrame from a whitespace-delimited file after skipping metadata lines.

    Args:
        data_path: Path to the data file.
        config: Config dict containing 'columns' list.

    Returns:
        Loaded DataFrame with specified column names.
    """
    try:
        column_names = config["columns"]
        expected_cols = len(column_names)
        start_line = detect_data_start(data_path, expected_fields=expected_cols)

        df = pd.read_csv(
            data_path,
            sep=r"\s+",
            header=None,
            names=column_names,
            engine="python",
            skiprows=start_line,
        )

        logger.info("Dataset created with shape %s", df.shape)
        return df

    except KeyError as e:
        logger.error("Missing config key: %s", e)
        raise ValueError(f"Missing config key: {e}")
    except Exception as e:
        logger.exception("Failed to create dataset.")
        raise RuntimeError(f"Dataset creation failed: {e}")


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """
    try:
        logger.info("Saving dataset to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Dataset saved.")
    except Exception as e:
        logger.exception("Failed to save dataset.")
        raise IOError(f"Could not save dataset: {e}")
