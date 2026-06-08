"""
main.py — FastAPI app serving the custom logistic regression churn model.
Loads theta + scaler from artifacts/ at startup (baked into Docker image).
"""

import os
import pickle
import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    CustomerData,
    CustomerBatch,
    PredictionResponse,
    BatchPredictionResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Model state (loaded once at startup) ──────────────────────────────────────
state: dict = {}

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")


# ── Helper functions ───────────────────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _risk(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


def _run_inference(customer: CustomerData) -> PredictionResponse:
    theta  = state["theta"]
    scaler = state["scaler"]

    features = np.array([[
        customer.age,
        customer.total_purchase,
        customer.account_manager,
        customer.years,
        customer.num_sites,
    ]])

    X_scaled = scaler.transform(features)
    X_final  = np.c_[np.ones((1, 1)), X_scaled]       # add bias term
    prob     = float(sigmoid(X_final.dot(theta))[0])
    label    = 1 if prob > 0.5 else 0

    return PredictionResponse(
        churn_prediction=label,
        churn_probability=round(prob, 4),
        risk_level=_risk(prob),
    )


# ── Lifespan (loads model on startup, cleans up on shutdown) ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    theta_path  = os.path.join(ARTIFACTS_DIR, "theta.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")

    try:
        with open(theta_path,  "rb") as f: state["theta"]  = pickle.load(f)
        with open(scaler_path, "rb") as f: state["scaler"] = pickle.load(f)
        logger.info("✅  Model artifacts loaded from %s", ARTIFACTS_DIR)
    except FileNotFoundError as e:
        logger.error("❌  Artifact not found: %s", e)
        raise RuntimeError(f"Model artifacts missing: {e}")

    yield  # app runs here

    # Shutdown
    state.clear()
    logger.info("🛑  Model unloaded — shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts whether a marketing-agency customer will churn "
        "using a custom logistic regression model trained from scratch."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
async def root():
    return {"service": "Churn Prediction API", "status": "running", "version": "1.0.0"}


@app.get("/health", tags=["meta"])
async def health():
    model_ready = "theta" in state and "scaler" in state
    if not model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict churn for a single customer",
)
async def predict_single(customer: CustomerData):
    """
    Returns:
    - **churn_prediction**: 1 = will churn, 0 = will stay
    - **churn_probability**: raw sigmoid score
    - **risk_level**: Low | Medium | High
    """
    if "theta" not in state:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        return _run_inference(customer)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    summary="Predict churn for a batch of customers",
)
async def predict_batch(data: CustomerBatch):
    if "theta" not in state:
        raise HTTPException(status_code=503, detail="Model not ready")
    try:
        preds       = [_run_inference(c) for c in data.customers]
        churn_count = sum(p.churn_prediction for p in preds)
        return BatchPredictionResponse(
            predictions=preds,
            total_customers=len(preds),
            churn_count=churn_count,
            churn_rate_pct=round(churn_count / len(preds) * 100, 2),
        )
    except Exception as exc:
        logger.exception("Batch inference error")
        raise HTTPException(status_code=500, detail=str(exc))
