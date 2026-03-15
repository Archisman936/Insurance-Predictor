from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated, Dict
import pickle
import pandas as pd

# -------------------- Load Model --------------------

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get class labels for probability mapping
class_labels = model.classes_.tolist()

app = FastAPI()

# -------------------- City Tier Lists --------------------

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]

tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

# -------------------- User Input Validation --------------------

class UserInput(BaseModel):
    age: Annotated[int, Field(..., ge=0, le=120, description="Age must be between 0 and 120")]
    weight: Annotated[float, Field(..., ge=0, description="Weight must be greater than 0")]
    height: Annotated[float, Field(..., ge=0, le=2.5, description="Height must be between 0 and 2.5 meters")]
    income_lpa: Annotated[float, Field(..., description="Income in lakhs per annum")]
    smoker: Annotated[bool, Field(..., description="Is the person a smoker?")]
    city: Annotated[str, Field(..., description="City of residence")]
    occupation: Annotated[
        Literal['retired', 'freelancer', 'student', 'government_job',
                'business_owner', 'unemployed', 'private_job'],
        Field(..., description="Occupation of the person")
    ]

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker and self.bmi > 27:
            return "medium"
        return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        return 3

# -------------------- Output Model Validation --------------------

class PredictionResponse(BaseModel):
    predicted_category: str = Field(
        ...,
        description="The predicted insurance premium category",
        example="High"
    )
    confidence: float = Field(
        ...,
        description="Model's confidence score for the predicted class (range: 0 to 1)",
        example=0.8432
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all possible classes",
        example={"Low": 0.01, "Medium": 0.15, "High": 0.84}
    )

# -------------------- Prediction Logic --------------------

def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    # Predict the class
    predicted_class = model.predict(df)[0]

    # Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)

    # Create mapping: {class_name: probability}
    class_probs = dict(
        zip(class_labels, map(lambda p: round(p, 4), probabilities))
    )

    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs
    }

# -------------------- API Endpoint --------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_premium(data: UserInput):

    input_dict = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation,
    }

    result = predict_output(input_dict)

    return result