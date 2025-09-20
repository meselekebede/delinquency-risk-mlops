# scripts/retrain.py
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MLflow
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("delinquency-prediction-retrain")

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_loan_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("🔄 Starting automatic retraining pipeline...")

# --- 1. Load existing data ---
print("📁 Loading existing data...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found at {DATA_PATH}")
df_existing = pd.read_csv(DATA_PATH)
print(f"✅ Existing dataset size: {len(df_existing)}")

# --- 2. Simulate New Incoming Data ---
print("🆕 Generating new synthetic data for retraining...")
n_new = 500  # Simulate 500 new loan records
np.random.seed(None)  # Allow randomness

new_data = {
    'age': np.random.randint(18, 70, n_new),
    'gender': np.random.choice(['Male', 'Female'], n_new),
    'education': np.random.choice(['Primary', 'Secondary', 'Tertiary'], n_new),
    'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_new),
    'monthly_income': np.round(np.random.lognormal(9.2, 0.8, n_new), 2),
    'loan_amount': np.round(np.random.uniform(5000, 150000, n_new), 2),
    'loan_tenure_months': np.random.choice([6, 12, 18, 24], n_new),
    'loan_purpose': np.random.choice(['Business', 'Emergency', 'Education', 'Home Improvement'], n_new),
    'region': np.random.choice(['Urban', 'Rural'], n_new),
    'previous_default': np.random.choice([0, 1], n_new, p=[0.75, 0.25]),
    'num_previous_loans': np.random.poisson(1.5, n_new),
    'credit_score': np.random.randint(300, 850, n_new)
}
df_new = pd.DataFrame(new_data)

# Recompute target with slight drift (e.g., more defaults due to economy)
risk_score = (
    (df_new['monthly_income'] < 10000).astype(int) * 0.3 +
    df_new['previous_default'] * 0.5 +
    (df_new['loan_amount'] / df_new['monthly_income'] > 6).astype(int) * 0.4 +
    (df_new['credit_score'] < 550).astype(int) * 0.3 +
    np.random.normal(0, 0.05, n_new)
)
df_new['delinquent'] = (risk_score > 0.5).astype(int)

# Combine old + new data
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
print(f"📊 Combined dataset size: {len(df_combined)}")

# Save updated data
updated_path = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_loan_data.csv")
df_combined.to_csv(updated_path, index=False)
print(f"💾 Updated data saved to {updated_path}")

# --- 3. Reuse Training Logic from src/train.py ---
# We could refactor into a module, but for now duplicate key logic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def preprocess_and_train(df, model_name, model_instance):
    # Feature engineering
    df = df.copy()
    df['income_to_loan_ratio'] = df['monthly_income'] / df['loan_amount']
    df['default_history_weight'] = df['previous_default'] * df['num_previous_loans']

    # Encode categoricals
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_employment = LabelEncoder()
    le_purpose = LabelEncoder()
    le_region = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['education'] = le_education.fit_transform(df['education'])
    df['employment_status'] = le_employment.fit_transform(df['employment_status'])
    df['loan_purpose'] = le_purpose.fit_transform(df['loan_purpose'])
    df['region'] = le_region.fit_transform(df['region'])

    feature_cols = [col for col in df.columns if col != 'delinquent']
    X = df[feature_cols]
    y = df['delinquent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    with mlflow.start_run(run_name=f"{model_name}_retrained_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        model_instance.fit(X_train, y_train)
        y_pred_proba = model_instance.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"📈 {model_name} - AUC: {auc:.4f}")

        # Log metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("dataset_size", len(df))
        mlflow.set_tag("retraining", "true")

        # Save model and encoders
        model_path = os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model_instance, model_path)
        mlflow.log_artifact(model_path)

        # Save encoders only once (in last model)
        if model_name == "Random Forest":
            for name, enc in [
                ("le_gender.pkl", le_gender),
                ("le_education.pkl", le_education),
                ("le_employment.pkl", le_employment),
                ("le_purpose.pkl", le_purpose),
                ("le_region.pkl", le_region),
            ]:
                path = os.path.join(MODELS_DIR, name)
                joblib.dump(enc, path)
                mlflow.log_artifact(path)

        return auc

# --- 4. Train Models ---
print("🚀 Retraining models on updated data...")

rf_auc = preprocess_and_train(
    df_combined,
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42)
)

xgb_auc = preprocess_and_train(
    df_combined,
    "XGBoost",
    XGBClassifier(random_state=42, eval_metric='logloss')
)

lr_auc = preprocess_and_train(
    df_combined,
    "Logistic Regression",
    LogisticRegression(max_iter=1000)
)

best_auc = max(rf_auc, xgb_auc, lr_auc)
best_model = "Random Forest" if rf_auc == best_auc else "XGBoost" if xgb_auc == best_auc else "Logistic Regression"

print(f"\n🎉 Retraining complete!")
print(f"🏆 Best model: {best_model} with AUC = {best_auc:.4f}")
print(f"📌 Check MLflow UI: http://localhost:5000")