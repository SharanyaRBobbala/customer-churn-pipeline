# AWS Deployment Guide

This project supports two deployment paths depending on what you want to
demonstrate. Both build on the same S3 bucket + IAM role from
`s3_setup.py` and `iam_policy.json`.

## Prerequisites (both paths)

```bash
aws configure                     # set your access key, secret, region
python infra/s3_setup.py --bucket-name churn-pipeline-<yourname> --region us-east-1
export CHURN_S3_BUCKET=churn-pipeline-<yourname>
```

Create an IAM role (`churn-pipeline-role`) and attach `iam_policy.json`
(replace `CHURN_BUCKET_NAME` with your actual bucket name first).

---

## Path A: ECS Fargate (recommended — matches your CI/CD resume bullet)

This mirrors: *"Built CI/CD deployment pipelines enabling automated
testing, containerized deployment, and scalable production inference."*

1. **Push image to ECR:**
   ```bash
   aws ecr create-repository --repository-name customer-churn-api
   aws ecr get-login-password --region us-east-1 | \
     docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   docker build -t customer-churn-api .
   docker tag customer-churn-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/customer-churn-api:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/customer-churn-api:latest
   ```

2. **Create an ECS Fargate service** (via console or CLI) pointing at that
   image, with the IAM role attached, port 8000 exposed, and an
   Application Load Balancer in front of it.

3. The GitHub Actions workflow (`.github/workflows/ci.yml`) automates
   steps 1 on every push to `main` — you just need to add your AWS
   credentials as repo secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

## Path B: SageMaker (if you want SageMaker on your resume specifically)

1. Package `data/best_model.joblib` + an inference script into a
   `model.tar.gz` per SageMaker's scikit-learn container conventions.
2. Upload to `s3://<bucket>/models/sagemaker/model.tar.gz`.
3. Create a SageMaker Model → Endpoint Configuration → Endpoint using the
   built-in `sklearn` inference container.
4. Test with `boto3.client("sagemaker-runtime").invoke_endpoint(...)`.

This path costs more to keep running — tear the endpoint down when not
actively demoing it (`aws sagemaker delete-endpoint`).

---

## Cost control

- ECS Fargate: scale desired task count to 0 when not in use.
- Always delete the ALB if unused — it bills hourly regardless of traffic.
- Set a Budget alarm (see AWS Budgets console) at $5 to catch anything
  left running by accident.
