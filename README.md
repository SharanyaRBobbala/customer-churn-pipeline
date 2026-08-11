# Customer Churn Prediction Pipeline

An end-to-end, production-style ML pipeline for predicting customer churn —
built to mirror real MLOps workflows: modular stages, experiment tracking,
containerized deployment, and CI/CD automation on AWS.

This project directly reflects hands-on work from my time as an **AI
Engineer at Accenture**, where I built modular ML pipelines (ingestion →
preprocessing → feature engineering → training → evaluation → inference)
and deployed containerized AI services on AWS and Azure.

## Architecture

```
                    ┌─────────────┐
   S3 (raw data) → │  Ingestion  │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │Preprocessing│  clean, encode, split
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │  Features   │  tenure buckets, spend ratios
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │  Training   │  7 classifiers, MLflow tracking
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Evaluation  │  threshold tuning for recall
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
   FastAPI /predict │  Inference  │ → S3 (model artifact, versioned)
                    └─────────────┘
                           ↓
                 Docker → ECR → ECS Fargate
                 (automated via GitHub Actions)
```

## What this demonstrates

| Stage | What it does | Resume alignment |
|---|---|---|
| `src/ingestion.py` | Pulls raw data from S3 (or local for dev) | Document ingestion workflows |
| `src/preprocessing.py` | Cleans, encodes, splits data | Data preparation |
| `src/features.py` | Derives tenure/spend/service features | Feature engineering |
| `src/train.py` | Trains 7 classifiers, logs every run to MLflow | Experiment tracking, model versioning |
| `src/evaluate.py` | Sweeps decision thresholds, optimizes for recall | Model validation |
| `src/inference.py` | Batch scoring + real-time FastAPI endpoint | REST APIs, production inference |
| `Dockerfile` + `.github/workflows/ci.yml` | Multi-stage build → ECR → ECS, automated on push | CI/CD, containerized AWS deployment |
| `infra/` | S3 bucket setup, least-privilege IAM policy | AWS infrastructure |

## Models trained

RandomForest, XGBoost, Logistic Regression, Gradient Boosting, AdaBoost,
KNN, and Naive Bayes — all trained in one run, all logged to MLflow with
full params + metrics, with the best (by ROC-AUC) automatically selected
and pushed to S3.

## Quickstart (local)

```bash
git clone https://github.com/SharanyaRBobbala/customer-churn-pipeline.git
cd customer-churn-pipeline
pip install -r requirements.txt

# Get the dataset: IBM Telco Customer Churn (Kaggle)
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Save as data/raw_churn.csv

python -m src.preprocessing --input data/raw_churn.csv
python -m src.features
python -m src.train --tracking-uri sqlite:///mlflow.db
python -m src.evaluate

# View experiment tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Serve predictions locally
uvicorn src.inference:app --reload
# then: curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```

## AWS Deployment

See [`infra/deploy_notes.md`](infra/deploy_notes.md) for full setup —
covers S3 bucket creation, IAM role setup, and both ECS Fargate and
SageMaker deployment paths.

```bash
python infra/s3_setup.py --bucket-name churn-pipeline-<yourname>
export CHURN_S3_BUCKET=churn-pipeline-<yourname>
```

Pushing to `main` triggers `.github/workflows/ci.yml`: lint → test →
build Docker image → push to ECR → redeploy ECS service.

## Testing

```bash
pytest tests/ -v
```

## Notes on model performance

Metrics will vary significantly depending on the dataset used. This repo
ships without the actual data file (see `.gitignore`) — pull the IBM
Telco dataset above for realistic, signal-rich results. The pipeline
logic itself (encoding, feature derivation, threshold tuning) is fully
tested and dataset-agnostic.

## Tech stack

Python · scikit-learn · XGBoost · MLflow · FastAPI · Docker · AWS (S3,
ECR, ECS, IAM) · GitHub Actions
