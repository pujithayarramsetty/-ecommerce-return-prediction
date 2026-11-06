
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

# Initialize FastAPI
app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description="Predicts whether an e-commerce product will be returned",
    version="1.0.1"
)

# Load trained model and feature names
model = joblib.load("best_return_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# Input data structure
class ProductData(BaseModel):
    Product_Category: str
    Product_Price_INR: float = Field(gt=0)
    Discount_Percent: float = Field(ge=0, le=100)
    Expected_Delivery_Days: int = Field(gt=0)
    Actual_Delivery_Days: float = Field(gt=0)
    Customer_Rating_PreOrder: float = Field(ge=1, le=5)
    Previous_Returns: int = Field(ge=0)
    Previous_Orders: int = Field(ge=0)
    Payment_Method: str
    Damaged_Package_Flag: str
    Package_Size: str
    Promotion_Active: str


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Product Return Prediction API is running",
        "api_version": "1.0.1"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: ProductData):

    try:
        # Convert input into DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Calculate engineered features
        input_df["Discount_Amount"] = (
            input_df["Product_Price_INR"] *
            input_df["Discount_Percent"] / 100
        )

        input_df["Delivery_Delay"] = (
            input_df["Actual_Delivery_Days"] -
            input_df["Expected_Delivery_Days"]
        )

        input_df["Customer_Return_Frequency"] = np.where(
            input_df["Previous_Orders"] > 0,
            input_df["Previous_Returns"] /
            input_df["Previous_Orders"],
            0
        )

        # One-hot encode categorical columns
        categorical_columns = [
            "Product_Category",
            "Payment_Method",
            "Damaged_Package_Flag",
            "Package_Size",
            "Promotion_Active"
        ]

        input_df = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            drop_first=True,
            dtype=int
        )

        # Match training feature columns and order
        input_df = input_df.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Predict
        prediction = int(model.predict(input_df)[0])

        probability = model.predict_proba(input_df)[0][prediction]

        result = (
            "Returned" if prediction == 1
            else "Not Returned"
        )

        return {
            "prediction": result,
            "prediction_value": prediction,
            "confidence": round(float(probability) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
