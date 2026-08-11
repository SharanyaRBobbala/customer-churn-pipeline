"""
Feature Engineering Module
----------------------------
Derives higher-signal features from the cleaned churn dataset (tenure
buckets, charge ratios, service-count aggregates). Third stage of the
pipeline (see preprocessing.py for stage 2).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


class FeatureEngineer:
    """Adds derived features on top of the cleaned/encoded dataset."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "tenure" in df.columns:
            df["tenure_years"] = df["tenure"] / 12.0
            df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

        if {"TotalCharges", "tenure"}.issubset(df.columns):
            # avoid divide-by-zero for tenure == 0
            df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

        if {"MonthlyCharges", "TotalCharges"}.issubset(df.columns):
            df["charge_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)

        present_service_cols = [c for c in SERVICE_COLS if c in df.columns]
        if present_service_cols:
            # after label encoding, "No"/"No service" typically map to 0 —
            # this counts how many services are actively subscribed
            df["active_service_count"] = (df[present_service_cols] > 0).sum(axis=1)

        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == 0).astype(int)  # encoded value

        return df


def main():
    parser = argparse.ArgumentParser(description="Engineer features for churn data")
    parser.add_argument("--train-input", default="data/train.csv")
    parser.add_argument("--test-input", default="data/test.csv")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    fe = FeatureEngineer()

    for split, path in [("train", args.train_input), ("test", args.test_input)]:
        df = pd.read_csv(path)
        df_transformed = fe.transform(df)
        out_path = f"{args.output_dir}/{split}_features.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_transformed.to_csv(out_path, index=False)
        logger.info(f"{split}: {df_transformed.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    main()
