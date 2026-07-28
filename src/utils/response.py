from pydantic import BaseModel
from typing import Literal, Optional, List


class PassengerPrediction(BaseModel):
    
    passenger_id: int
    predicted: Literal['Survived', 'Not Survived']
    
    
class PredictionResponse(BaseModel):

    predictions: List[PassengerPrediction]
