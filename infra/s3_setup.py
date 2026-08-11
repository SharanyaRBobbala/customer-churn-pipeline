"""
S3 Infrastructure Setup
--------------------------
One-time setup script: creates the S3 bucket used for raw data, MLflow
artifacts, and trained model storage, with versioning enabled so model
artifacts are never silently overwritten.

Usage:
    python infra/s3_setup.py --bucket-name your-unique-bucket-name --region us-east-1
"""

import argparse
import json
import logging

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_bucket(bucket_name: str, region: str):
    s3 = boto3.client("s3", region_name=region)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        logger.info(f"Created bucket: {bucket_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
            logger.info(f"Bucket already exists and is owned by you: {bucket_name}")
        else:
            raise

    # enable versioning so a bad model push doesn't destroy the last good one
    s3.put_bucket_versioning(
        Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"}
    )
    logger.info("Versioning enabled")

    # block all public access — this bucket holds model artifacts, not public assets
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    logger.info("Public access blocked")

    # create logical folder structure
    for prefix in ["raw/", "models/", "mlflow-artifacts/", "predictions/"]:
        s3.put_object(Bucket=bucket_name, Key=prefix)
    logger.info("Created folder structure: raw/, models/, mlflow-artifacts/, predictions/")


def main():
    parser = argparse.ArgumentParser(description="Set up S3 bucket for churn pipeline")
    parser.add_argument("--bucket-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    create_bucket(args.bucket_name, args.region)
    logger.info(f"Set this in your environment: export CHURN_S3_BUCKET={args.bucket_name}")


if __name__ == "__main__":
    main()
