
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    sample_data = {
        "Product_Category": "Electronics",
        "Product_Price_INR": 2500,
        "Discount_Percent": 10,
        "Expected_Delivery_Days": 3,
        "Actual_Delivery_Days": 4,
        "Customer_Rating_PreOrder": 4,
        "Previous_Returns": 1,
        "Previous_Orders": 5,
        "Payment_Method": "UPI",
        "Damaged_Package_Flag": "No",
        "Package_Size": "Medium",
        "Promotion_Active": "Yes"
    }

    response = client.post(
        "/predict",
        json=sample_data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [
        "Returned",
        "Not Returned"
    ]

    assert 0 <= result["confidence"] <= 100
