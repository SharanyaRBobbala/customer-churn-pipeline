"""
Training Module
-----------------
Trains multiple classifiers on the engineered churn features and logs
every run to MLflow (params, metrics, model artifact). Automates
experiment tracking and model versioning per the resume bullet:

"Automated experiment tracking, model versioning, and deployment
workflows using MLOps best practices."

Also uploads the best model artifact to S3 for use by the deployment
pipeline (see infra/ and .github/workflows/ci.yml).
"""

import argparse
import logging
import os

import boto3
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Churn"

MODEL_REGISTRY = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    ),
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=42
    ),
    "adaboost": lambda: AdaBoostClassifier(n_estimators=150, random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": lambda: GaussianNB(),
}


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_log(model_name: str, X_train, y_train, X_test, y_test, experiment_name: str):
    """Train a single model and log it as an MLflow run."""
    with mlflow.start_run(run_name=model_name):
        model = MODEL_REGISTRY[model_name]()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_param("model_type", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # XGBoost's native booster isn't a plain sklearn estimator under the
        # hood, so mlflow.sklearn's serializer (skops) refuses to trust it.
        # Use the dedicated xgboost flavor for that model, sklearn for the rest.
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"{model_name}: {metrics}")
        return model, metrics


def upload_to_s3(local_path: str, bucket: str, key: str):
    """Push the winning model artifact to S3 for the deployment pipeline."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    logger.info(f"Uploaded best model -> s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Train churn classifiers with MLflow tracking")
    parser.add_argument("--train-input", default="data/train_features.csv")
    parser.add_argument("--test-input", default="data/test_features.csv")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="customer-churn-prediction")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--output-model", default="data/best_model.joblib")
    parser.add_argument("--upload-s3", action="store_true")
    parser.add_argument("--s3-bucket", default=os.environ.get("CHURN_S3_BUCKET"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    train_df = pd.read_csv(args.train_input)
    test_df = pd.read_csv(args.test_input)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    results = {}
    models = {}
    for name in args.models:
        model, metrics = train_and_log(name, X_train, y_train, X_test, y_test, args.experiment_name)
        results[name] = metrics
        models[name] = model

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    logger.info(f"Best model: {best_name} (ROC-AUC={results[best_name]['roc_auc']:.4f})")

    joblib.dump(models[best_name], args.output_model)
    logger.info(f"Saved best model -> {args.output_model}")

    if args.upload_s3 and args.s3_bucket:
        upload_to_s3(args.output_model, args.s3_bucket, f"models/{best_name}/model.joblib")


if __name__ == "__main__":
    main()
