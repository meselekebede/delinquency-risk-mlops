# Delinquency Risk Prediction System

Predicts whether a borrower will become delinquent (late on payments) using synthetic microfinance data.

Built with MLOps best practices:
- MLflow for experiment tracking
- DVC for data versioning
- FastAPI for model serving
- Docker-ready

## Steps
1. Generate synthetic loan data
2. Preprocess and split
3. Train XGBoost classifier
4. Track metrics with MLflow
5. Serve predictions via API