"""
Data Ingestion Module
----------------------
Loads raw customer churn data either from an S3 bucket (production) or
from a local path (development/testing). Designed to be the first stage
of the modular ML pipeline: ingestion -> preprocessing -> features ->
training -> evaluation -> inference.

Resume alignment:
"Developed reusable document ingestion, preprocessing, and retrieval
workflows for enterprise knowledge systems." (Accenture, AI Engineering Intern)
"Built modular ML pipelines for ingestion, preprocessing, feature
engineering, training, evaluation, and inference." (Accenture, AI Engineer)
"""

import argparse
import logging
import os
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles raw data retrieval from S3 or local filesystem."""

    def __init__(self, bucket_name: str | None = None, region: str = "us-east-1"):
        self.bucket_name = bucket_name or os.environ.get("CHURN_S3_BUCKET")
        self.region = region
        self._s3_client = None

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client("s3", region_name=self.region)
        return self._s3_client

    def load_from_s3(self, key: str, local_cache: str = "data/raw_churn.csv") -> pd.DataFrame:
        """Download a CSV object from S3 and load it into a DataFrame.

        Falls back to a clear error message if credentials or bucket access
        are not configured, rather than failing silently.
        """
        if not self.bucket_name:
            raise ValueError(
                "No S3 bucket configured. Set CHURN_S3_BUCKET env var or pass bucket_name."
            )

        Path(local_cache).parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{key} -> {local_cache}")
            self.s3_client.download_file(self.bucket_name, key, local_cache)
        except NoCredentialsError:
            logger.error("AWS credentials not found. Configure via `aws configure` or IAM role.")
            raise
        except ClientError as e:
            logger.error(f"S3 access failed: {e}")
            raise

        return pd.read_csv(local_cache)

    def load_local(self, path: str) -> pd.DataFrame:
        """Load raw data from a local CSV path (used for local dev/tests)."""
        logger.info(f"Loading local dataset from {path}")
        return pd.read_csv(path)

    def load(self, source: str, s3_key: str | None = None) -> pd.DataFrame:
        """Unified entrypoint: source is 's3' or 'local'."""
        if source == "s3":
            if not s3_key:
                raise ValueError("s3_key is required when source='s3'")
            return self.load_from_s3(s3_key)
        elif source == "local":
            return self.load_local("data/raw_churn.csv")
        else:
            raise ValueError(f"Unknown source: {source}. Use 's3' or 'local'.")


def main():
    parser = argparse.ArgumentParser(description="Ingest raw churn data")
    parser.add_argument("--source", choices=["s3", "local"], default="local")
    parser.add_argument("--s3-key", default="raw/telco_churn.csv")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--output", default="data/raw_churn.csv")
    args = parser.parse_args()

    ingestion = DataIngestion(bucket_name=args.bucket)
    df = ingestion.load(source=args.source, s3_key=args.s3_key)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Ingested {len(df)} rows, {len(df.columns)} columns -> {args.output}")


if __name__ == "__main__":
    main()
