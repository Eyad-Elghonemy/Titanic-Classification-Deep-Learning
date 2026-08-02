import os
from dotenv import load_dotenv
import joblib
from tensorflow.keras.models import load_model
from supabase import create_client

# load .env file
load_dotenv(override=True)

# get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SRC_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load preprocessor and model
preprocessor = joblib.load(os.path.join(SRC_FOLDER_PATH, 'artifacts', 'preprocessor.joblib'))
model = load_model(os.path.join(SRC_FOLDER_PATH, 'artifacts', 'best_model.keras'))

# supabase client (used for logging API calls to the api_logs table)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def log_api_call(operation_type: str, success: bool, record_count: int = 0,
                  survived_count: int = 0, error_detail: str = None,
                  response_time_ms: int = None):
    """Insert one row into api_logs. Never raises — a logging failure must not break the API."""
    try:
        supabase.table("api_logs").insert({
            "operation_type": operation_type,
            "success": success,
            "record_count": record_count,
            "survived_count": survived_count,
            "error_detail": error_detail,
            "response_time_ms": response_time_ms,
        }).execute()
    except Exception as e:
        print(f"[log_api_call] Failed to write log: {e}")