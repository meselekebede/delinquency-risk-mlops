# src/train.py
import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# MLflow
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("delinquency-prediction")


def main():
    # --- 1. Get Project Root & Ensure Directories ---
    print("📁 Setting up project paths...")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "synthetic_loan_data.csv")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Create required directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"✅ Models directory: {MODELS_DIR}")

    # --- 2. Load Data ---
    print("🚀 Loading raw data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} records.")

    # --- 3. Feature Engineering ---
    print("🔧 Preprocessing data...")
    df['income_to_loan_ratio'] = df['monthly_income'] / df['loan_amount']
    df['default_history_weight'] = df['previous_default'] * df['num_previous_loans']

    # Encode categorical variables
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

    # Define features and target
    feature_cols = [col for col in df.columns if col != 'delinquent']
    X = df[feature_cols]
    y = df['delinquent']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Define Models ---
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train, X_test),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'), X_train, X_test)
    }

    best_metric = 0
    best_model_name = ""

    print("📊 Training and evaluating models...\n")

    for name, (model, X_tr, X_ts) in models.items():
        print(f"=== {name} ===")
        with mlflow.start_run(run_name=name):
            # Fit model
            model.fit(X_tr, y_train)

            # Predictions
            y_pred_proba = model.predict_proba(X_ts)[:, 1]
            y_pred = model.predict(X_ts)

            # Metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}\n")

            # Log to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)

            # --- 5. Save Model & Artifacts ---
            model_filename = f"{name.lower().replace(' ', '_')}.pkl"
            model_path = os.path.join(MODELS_DIR, model_filename)

            try:
                joblib.dump(model, model_path)
                print(f"✅ Saved model: {model_path}")
                mlflow.log_artifact(model_path)
            except Exception as e:
                print(f"❌ Failed to save {name}: {e}")

            # Only save scaler once (skip for tree-based models if needed)
            if name == "Logistic Regression":
                scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
                try:
                    joblib.dump(scaler, scaler_path)
                    print(f"✅ Saved scaler: {scaler_path}")
                    mlflow.log_artifact(scaler_path)
                except Exception as e:
                    print(f"❌ Failed to save scaler: {e}")

            # Save label encoders (once only — do it in last model loop)
            if name == "XGBoost":
                encoder_mapping = {
                    "le_gender.pkl": le_gender,
                    "le_education.pkl": le_education,
                    "le_employment.pkl": le_employment,
                    "le_purpose.pkl": le_purpose,
                    "le_region.pkl": le_region,
                }
                for enc_name, encoder in encoder_mapping.items():
                    enc_path = os.path.join(MODELS_DIR, enc_name)
                    try:
                        joblib.dump(encoder, enc_path)
                        print(f"✅ Saved encoder: {enc_path}")
                        mlflow.log_artifact(enc_path)
                    except Exception as e:
                        print(f"❌ Failed to save {enc_name}: {e}")

            # Track best model by AUC
            if auc > best_metric:
                best_metric = auc
                best_model_name = name

            mlflow.set_tag("best_model", str(name == best_model_name))

    print(f"🎉 Best model: {best_model_name} with AUC = {best_metric:.4f}")
    print("📌 Check MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    main()