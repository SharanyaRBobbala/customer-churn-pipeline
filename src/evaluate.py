"""
Evaluation Module
-------------------
Loads the trained model, evaluates it against the held-out test set,
and sweeps decision thresholds to find the operating point that best
balances precision/recall for a churn-retention use case (missing a
churner is usually costlier than a false alarm, so we bias toward recall).
"""

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Churn"


def sweep_thresholds(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Evaluate precision/recall/F1 across a range of thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate churn model + tune threshold")
    parser.add_argument("--model-path", default="data/best_model.joblib")
    parser.add_argument("--test-input", default="data/test_features.csv")
    parser.add_argument("--output-report", default="data/evaluation_report.json")
    parser.add_argument("--target-recall", type=float, default=0.75,
                         help="Pick the lowest threshold achieving at least this recall")
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    test_df = pd.read_csv(args.test_input)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    logger.info("=== Default threshold (0.5) ===")
    logger.info("\n" + classification_report(y_test, y_pred_default))

    sweep = sweep_thresholds(y_test, y_proba)
    candidates = sweep[sweep["recall"] >= args.target_recall]
    best_row = candidates.iloc[-1] if not candidates.empty else sweep.iloc[sweep["f1"].idxmax()]
    best_threshold = best_row["threshold"]

    y_pred_tuned = (y_proba >= best_threshold).astype(int)

    report = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "default_threshold_metrics": {
            "precision": precision_score(y_test, y_pred_default, zero_division=0),
            "recall": recall_score(y_test, y_pred_default, zero_division=0),
            "f1": f1_score(y_test, y_pred_default, zero_division=0),
        },
        "tuned_threshold": float(best_threshold),
        "tuned_threshold_metrics": {
            "precision": precision_score(y_test, y_pred_tuned, zero_division=0),
            "recall": recall_score(y_test, y_pred_tuned, zero_division=0),
            "f1": f1_score(y_test, y_pred_tuned, zero_division=0),
        },
        "confusion_matrix_tuned": confusion_matrix(y_test, y_pred_tuned).tolist(),
    }

    logger.info(f"Tuned threshold: {best_threshold} -> {report['tuned_threshold_metrics']}")

    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report -> {args.output_report}")


if __name__ == "__main__":
    main()
