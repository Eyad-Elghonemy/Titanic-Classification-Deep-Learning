from pydantic import BaseModel
from typing import Literal, Optional, List


class PassengerPrediction(BaseModel):

    passenger_id: int
    predicted: Literal['Survived', 'Not Survived']
    probability: float  # model's raw survival probability (0.0 - 1.0)


class PredictionResponse(BaseModel):

    predictions: List[PassengerPrediction]