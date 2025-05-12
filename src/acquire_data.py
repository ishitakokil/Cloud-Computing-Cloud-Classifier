import logging
import sys
import time
from pathlib import Path
import requests
from typing import Optional

# Logger configuration
logger = logging.getLogger("data_acquirer")

def get_data(url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2) -> bytes:
    """
    Download data from a URL with retry logic.

    Args:
        url: Download URL.
        attempts: Max number of retries.
        wait: Initial wait between retries (seconds).
        wait_multiple: Factor to increase wait each attempt.

    Returns:
        Content in bytes if successful.

    Raises:
        SystemExit: If all attempts fail.
    """
    for attempt in range(1, attempts + 1):
        try:
            logger.info("Attempt %d: Downloading from %s", attempt, url)
            response = requests.get(url)
            response.raise_for_status()
            logger.info("Download successful.")
            return response.content
        except requests.RequestException as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            time.sleep(wait)
            wait *= wait_multiple

    logger.error("All download attempts failed for URL: %s", url)
    sys.exit(1)


def write_data(content: bytes, path: Path) -> None:
    """
    Write byte content to a specified file path.

    Args:
        content: Data to write.
        path: Destination file path.
    """
    try:
        logger.info("Writing data to %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        logger.info("Write complete.")
    except Exception as e:
        logger.exception("Failed to write data to file.")
        raise IOError(f"Could not write data: {e}")


def acquire_data(url: str, save_path: Path) -> None:
    """
    Download and save data from a URL.

    Args:
        url: Download URL.
        save_path: Path to save the downloaded file.
    """
    try:
        content = get_data(url)
        write_data(content, save_path)
        logger.info("Data saved to %s", save_path)
    except Exception as e:
        logger.error("Failed to acquire or save data: %s", e)
        sys.exit(1)

