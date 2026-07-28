from fastapi import FastAPI, HTTPException, Depends
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from src.utils.requests import PassengerData
from src.utils.response import PredictionResponse
from src.utils.config import APP_NAME, VERSION, API_SECRET_KEY
from src.utils.inference import predict_survival 

app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
    
)

@app.get('/', tags=['check'])
async def home():
    
    return {
        'message': "Up & Running"
    }
    

api_key_header = APIKeyHeader(name="X-API-KEY")
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key

    
@app.post("/classify",tags=['NN'], response_model=PredictionResponse)
async def classify(passengers: List[PassengerData], api_key: str=Depends(verify_api_key)):
    
    try:
        response = predict_survival(passengers=passengers)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")