from pydantic import BaseModel, Field
from typing import List


class CustomerData(BaseModel):
    age:             float = Field(..., ge=0,  le=120, example=37.0,    description="Customer age")
    total_purchase:  float = Field(..., ge=0,          example=9935.53, description="Total ads purchased ($)")
    account_manager: int   = Field(..., ge=0,  le=1,   example=1,       description="0 = no manager, 1 = assigned")
    years:           float = Field(..., ge=0,          example=7.71,    description="Years as a customer")
    num_sites:       float = Field(..., ge=0,          example=8.0,     description="Number of websites using the service")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 37.0,
                "total_purchase": 9935.53,
                "account_manager": 1,
                "years": 7.71,
                "num_sites": 8.0,
            }
        }


class CustomerBatch(BaseModel):
    customers: List[CustomerData] = Field(..., min_items=1)


class PredictionResponse(BaseModel):
    churn_prediction:  int   = Field(..., example=0,      description="1 = will churn, 0 = will stay")
    churn_probability: float = Field(..., example=0.2341, description="Probability score (0–1)")
    risk_level:        str   = Field(..., example="Low",  description="Low / Medium / High")


class BatchPredictionResponse(BaseModel):
    predictions:     List[PredictionResponse]
    total_customers: int
    churn_count:     int
    churn_rate_pct:  float
