"""
Preprocessing Module
---------------------
Cleans raw churn data: handles missing values, fixes dtypes, encodes
categoricals, and splits into train/test sets. Second stage of the
pipeline (see ingestion.py for stage 1).
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Churn"
ID_COL = "customerID"


class Preprocessor:
    """Cleans and encodes the raw churn dataset."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # TotalCharges arrives as a string with some blank entries in the
        # IBM Telco dataset (new customers with 0 tenure) -- coerce + fill.
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        # Drop rows missing the target -- can't train/evaluate on those.
        if TARGET_COL in df.columns:
            df = df.dropna(subset=[TARGET_COL])

        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])

        return df

    def encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if TARGET_COL in categorical_cols:
            categorical_cols.remove(TARGET_COL)

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str).map(
                    lambda v: le.transform([v])[0] if v in le.classes_ else -1
                )

        if TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).fillna(df[TARGET_COL])

        return df

    def save_encoders(self, path: str = "data/label_encoders.joblib"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoders, path)
        logger.info(f"Saved {len(self.label_encoders)} label encoders -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw churn data")
    parser.add_argument("--input", default="data/raw_churn.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} raw rows")

    pre = Preprocessor()
    df_clean = pre.clean(df)
    df_encoded = pre.encode(df_clean, fit=True)
    pre.save_encoders(f"{args.output_dir}/label_encoders.joblib")

    train_df, test_df = train_test_split(
        df_encoded, test_size=args.test_size, random_state=args.random_state,
        stratify=df_encoded[TARGET_COL] if TARGET_COL in df_encoded.columns else None,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/test.csv", index=False)

    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")


if __name__ == "__main__":
    main()
