import os
from pathlib import Path
from typing import Dict
import logging
import boto3

# Logger configuration
logger = logging.getLogger("uploader")



def upload_artifacts(directory: Path, aws_config: Dict[str, str]) -> None:
    """
    Upload all files in a local directory to an S3 bucket.

    Args:
        directory: Path to the directory containing artifacts.
        aws_config: Dictionary with 'bucket_name' and optional 'prefix'.
    """
    try:
        s3 = boto3.client("s3")
        bucket = os.environ.get("AWS_BUCKET", aws_config["bucket_name"])
        prefix = aws_config.get("prefix", "")

        logger.info("Uploading artifacts from %s to bucket: %s", directory, bucket)

        for root, _, files in os.walk(directory):
            for name in files:
                local_path = os.path.join(root, name)
                relative_path = os.path.relpath(local_path, start=directory)
                s3_path = f"{prefix}/{relative_path}".lstrip("/")

                logger.debug("Uploading %s to s3://%s/%s", local_path, bucket, s3_path)
                s3.upload_file(local_path, bucket, s3_path)

        logger.info("Upload complete.")

    except KeyError as e:
        logger.error("Missing required AWS config key: %s", e)
        raise ValueError(f"Missing required AWS config key: {e}")
    except Exception as e:
        logger.exception("Failed to upload artifacts to S3.")
        raise RuntimeError(f"Upload failed: {e}")
