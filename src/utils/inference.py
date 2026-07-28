from src.utils.requests import PassengerData
from src.utils.response import PassengerPrediction, PredictionResponse
import pandas as pd
from typing import List
from src.utils.config import preprocessor, model

def predict_survival(passengers: List[PassengerData]):
    
    df = pd.DataFrame([p.model_dump() for p in passengers])
    
    df['family_size'] = [p.family_size for p in passengers]
    df['is_alone'] = [p.is_alone for p in passengers]
        
    # Transform
    df_preprocessed = preprocessor.transform(df)
    
    # Predict
    predictions = (model.predict(df_preprocessed) > 0.5).astype('int32').flatten()
    
    pred_response = PredictionResponse(predictions=[
        PassengerPrediction(
            passenger_id=passengers[i].passenger_id,
            predicted="Survived" if pred == 1 else "Not Survived"
        )
        for i, pred in enumerate(predictions)
    ])
    
    return pred_response
    
    