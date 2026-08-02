from src.utils.requests import PassengerData
from src.utils.response import PassengerPrediction, PredictionResponse
import pandas as pd
import time
from typing import List
from src.utils.config import preprocessor, model, log_api_call


def predict_survival(passengers: List[PassengerData]):

    start_time = time.perf_counter()

    try:
        df = pd.DataFrame([p.model_dump() for p in passengers])

        # family_size and is_alone are @property fields on PassengerData,
        # so they are NOT included by model_dump() and must be added manually
        df['family_size'] = [p.family_size for p in passengers]
        df['is_alone'] = [p.is_alone for p in passengers]

        # Transform
        df_preprocessed = preprocessor.transform(df)

        # Predict — keep the raw probability, then threshold for the label
        raw_probabilities = model.predict(df_preprocessed).flatten()
        predictions = (raw_probabilities > 0.5).astype('int32')

        survived_count = int(predictions.sum())

        pred_response = PredictionResponse(predictions=[
            PassengerPrediction(
                passenger_id=passengers[i].passenger_id,
                predicted="Survived" if pred == 1 else "Not Survived",
                probability=round(float(raw_probabilities[i]), 4)
            )
            for i, pred in enumerate(predictions)
        ])

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        log_api_call(
            operation_type="classify",
            success=True,
            record_count=len(passengers),
            survived_count=survived_count,
            response_time_ms=elapsed_ms,
        )

        return pred_response

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        log_api_call(
            operation_type="classify",
            success=False,
            record_count=len(passengers),
            error_detail=str(e),
            response_time_ms=elapsed_ms,
        )
        raise