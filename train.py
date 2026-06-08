"""
train.py — Trains the custom logistic regression model and logs everything to MLflow.
Run this once locally (with customer_churn.csv in the same folder) to generate
artifacts/ and register the model.
"""

import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Hyperparameters ────────────────────────────────────────────────────────────
ALPHA      = 0.01
ITERATIONS = 1000
TEST_SIZE  = 0.2
RANDOM_STATE = 42

# ── Core ML functions (same as your notebook) ─────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    error = y * np.log(h) + (1 - y) * np.log(1 - h)
    cost  = -1 / m * sum(error)
    grad  = 1 / m * X.T.dot(h - y)
    return cost, grad

def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        cost, grad = cost_function(X, y, theta)
        theta = theta - alpha * grad
        cost_history[i] = cost
    return theta, cost_history

def predict(X, theta):
    return sigmoid(X.dot(theta))

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# ── MLflow Python Model Wrapper ────────────────────────────────────────────────
class ChurnPyfuncModel(mlflow.pyfunc.PythonModel):
    """Wraps theta + scaler so MLflow can serve them as one unit."""

    def load_context(self, context):
        with open(context.artifacts["theta"],  "rb") as f:
            self.theta  = pickle.load(f)
        with open(context.artifacts["scaler"], "rb") as f:
            self.scaler = pickle.load(f)

    def predict(self, context, model_input):
        X = self.scaler.transform(model_input)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        probs = sigmoid(X.dot(self.theta))
        return (probs > 0.5).astype(int)


# ── Main training run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Point to local or remote MLflow tracking server
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("customer-churn-prediction")

    with mlflow.start_run() as run:
        # 1. Log hyperparameters
        mlflow.log_params({
            "alpha":        ALPHA,
            "iterations":   ITERATIONS,
            "test_size":    TEST_SIZE,
            "random_state": RANDOM_STATE,
        })

        # 2. Load & preprocess data
        df = pd.read_csv("customer_churn.csv")
        df.drop(["Names", "Location", "Company", "Onboard_date"], axis=1, inplace=True)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        X_train_f = np.c_[np.ones((X_train_sc.shape[0], 1)), X_train_sc]
        X_test_f  = np.c_[np.ones((X_test_sc.shape[0],  1)), X_test_sc]

        # 3. Train
        theta, cost_history = gradient_descent(
            X_train_f, y_train.values,
            np.zeros(X_train_f.shape[1]), ALPHA, ITERATIONS
        )

        # 4. Evaluate
        y_pred = np.array([1 if p > 0.5 else 0 for p in predict(X_test_f, theta)])
        acc    = accuracy(y_test.values, y_pred)
        final_cost = float(cost_history[-1])

        mlflow.log_metric("accuracy",   round(acc * 100, 4))
        mlflow.log_metric("final_cost", round(final_cost, 6))
        print(f"✅  Accuracy: {acc * 100:.2f}%  |  Final cost: {final_cost:.6f}")

        # 5. Persist artifacts locally (needed for Docker image)
        os.makedirs("artifacts", exist_ok=True)
        theta_path  = "artifacts/theta.pkl"
        scaler_path = "artifacts/scaler.pkl"

        with open(theta_path,  "wb") as f: pickle.dump(theta,  f)
        with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

        # 6. Log model to MLflow with artifact references
        mlflow.pyfunc.log_model(
            name="churn_model",
            python_model=ChurnPyfuncModel(),
            artifacts={
                "theta":  theta_path,
                "scaler": scaler_path,
            },
            registered_model_name="CustomerChurnModel",
            pip_requirements=["numpy", "scikit-learn", "pandas"],
        )

        print(f"📦  Run ID : {run.info.run_id}")
        print(f"📌  Model registered as: CustomerChurnModel")
