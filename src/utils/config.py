import os
from dotenv import load_dotenv
import joblib
from tensorflow.keras.models import load_model 

# load .env file
load_dotenv(override=True)

# get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

SRC_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load preprocessor and model
preprocessor = joblib.load(os.path.join(SRC_FOLDER_PATH, 'artifacts', 'preprocessor.joblib'))
model = load_model(os.path.join(SRC_FOLDER_PATH, 'artifacts', 'best_model.keras'))

