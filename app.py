from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Any, Optional
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from datetime import datetime
import os
from contextlib import asynccontextmanager

tracking_uri = "http://host.docker.internal:5000"

model_uri = "models:/ticket_price_prediction_model/4"

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        ml_models['pipeline'] = mlflow.pyfunc.load_model(model_uri)
        print("MLflow Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    ml_models.clear()


app = FastAPI(
    title='Ticket Forecast API',
    version='1.0.0',
    lifespan=lifespan
)
    
class PredictRequest(BaseModel):
    Route_ID: str
    Travel_Date: datetime
    Booking_Date: datetime
    Booking_Timestamp: datetime
    Seat_Class: Literal['Standard', 'Flex', 'First']
    Booking_Channel: Literal['Web', 'Mobile', 'Phone', 'Agent']
    Origin: str
    Destination: str
    Distance_km: int
    Route_Category: Literal['Short', 'Medium', 'Long']
    Customer_Segment: Literal['Leisure', 'Business', 'Commuter', 'Group']
    Loyalty_Status: Literal['Gold', 'Silver', 'Bronze', 'Not loyal']
    Booking_Frequency_Qtr: int
    Average_Spend_GBP: float
    Total_Seats: int
    Seats_Sold_Realized: int
    Remaining_Seats_Realized: int
    Demand_Index: float
    Base_Price_At_Booking: Optional[float] = None
    Days_Before_Travel: int
    Price_Premium: Optional[float] = None
    Load_Factor: float


class PredictResponse(BaseModel):
    ticket_price_gbp: float


@app.get('/health')
def read_root():
    return {"Status": "Healthy"}


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    model = ml_models.get('pipeline') # assigns None if model was not loaded
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_data = pd.DataFrame([request.model_dump()])
        prediction = model.predict(input_data)[0]
        return PredictResponse(ticket_price_gbp=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))