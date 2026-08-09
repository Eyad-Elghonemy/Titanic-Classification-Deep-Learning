<div align="center">

<img src="logo_circle.png" alt="Titanic Survival Classification" width="200" />

# RMS · 1912 — Titanic Survival Classification
**A production-grade deep learning API that predicts passenger survival on the RMS Titanic, built with FastAPI and TensorFlow/Keras.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-2C3E50?logo=uvicorn&logoColor=white)](https://www.uvicorn.org/)
[![joblib](https://img.shields.io/badge/joblib-Serialization-4B8BBE?logo=python&logoColor=white)](https://joblib.readthedocs.io/)

</div>

---

## What is this?

A production-ready REST API that takes raw passenger data (age, fare, class, family info, embarkation port) and returns a survival prediction for each passenger using a trained **Keras neural network**. Designed to plug cleanly into any downstream dashboard, analytics pipeline, or application.

---
---

> ⚠️ **Security note:** the `X-API-Key` below is a **demo key** shared for evaluation purposes only. If this repository is public, rotate the key (set a new `SECRET_KEY_TOKEN` in your Hugging Face Space secrets) so this value stops working. Never rely on a key that has appeared in a public README for anything beyond a quick demo.

```
Demo X-API-Key: c0c2d9d05029aed5d5174ff5ff8e6d88
```



## Features

| Feature | Detail |
|---|---|
| ⚡ **Fast inference** | Model and preprocessor loaded once at startup — never per request |
| 📦 **Batch predictions** | Classify a single passenger or an entire list in one call |
| 🧩 **Clean architecture** | Config, inference logic, and request/response schemas fully separated |
| 🔒 **Env-based config** | All secrets loaded from `.env`, never hardcoded |
| 📖 **Auto-generated docs** | Swagger UI & ReDoc available out of the box |
| 🛡️ **Robust error handling** | Descriptive HTTP error responses on any inference failure |

---

## Project Structure

```
.
├── main.py                       # FastAPI application entry point
├── requirements.txt              # Dependencies
├── .env.example                  # Environment variables template
├── .env                          # Environment variables (never committed)
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── artifacts/
    │   ├── best_model.keras      # Trained Keras neural network
    │   └── preprocessor.joblib   # Fitted preprocessing pipeline
    ├── notebooks/
    │   └── notebook.ipynb        # Model training & experimentation
    └── utils/
        ├── __init__.py
        ├── config.py             # App config, model & preprocessor loading
        ├── inference.py          # Prediction logic
        ├── requests.py           # Pydantic request schemas
        └── response.py           # Pydantic response schemas
```

---

## Model & Preprocessing Pipeline

The model is a neural network trained on the classic Titanic dataset. See `src/notebooks/notebook.ipynb` for the full training pipeline including feature engineering.

**Engineered features:**

| Feature | Formula | Meaning |
|---|---|---|
| `family_size` | `parch + sibsp + 1` | Total family members aboard |
| `is_alone` | `1 if family_size == 1` | Whether the passenger traveled alone |

**Inference flow:**

```
Raw passenger JSON
      ↓ Pydantic validation
Feature engineering (family_size, is_alone)
      ↓ preprocessor.joblib (fitted scaler + encoder)
Transformed feature matrix
      ↓ best_model.keras
Raw predictions
      ↓ label mapping
"Survived" / "Not Survived"
```

---

## Requirements

- Python 3.12+

**Key dependencies:**

```
uvicorn==0.51.0
fastapi==0.139.2
tensorflow==2.21.0
pydantic==2.13.4
joblib==1.5.3
python-dotenv==1.2.2
python-multipart==0.0.32
```

---

## Installation & Running

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
APP_NAME="Titanic-Survived-Classification"
VERSION="1.0"
API_SECRET_KEY=your-secret-key-here
```

### 5. Run the API

```bash
uvicorn main:app --reload
```

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000` | API base URL |
| `http://127.0.0.1:8000/docs` | Swagger UI (interactive docs) |
| `http://127.0.0.1:8000/redoc` | ReDoc (alternative docs) |

---

## API Reference

### `GET /`
Health check — no authentication required.

**Response**
```json
{ "message": "Up & Running" }
```

---

### `POST /classify`
Predict survival for one or more passengers in a single call.

**Request body** — a JSON array of passenger objects:

```json
[
  {
    "passenger_id": 1,
    "age": 22.0,
    "fare": 7.25,
    "sex": "female",
    "embarked": "S",
    "parch": 0,
    "sibsp": 1,
    "pclass": 3
  },
  {
    "passenger_id": 2,
    "age": 12.0,
    "fare": 8.2,
    "sex": "male",
    "embarked": "Q",
    "parch": 3,
    "sibsp": 2,
    "pclass": 3
  }
]
```

**Field reference**

| Field | Type | Description |
|---|---|---|
| `passenger_id` | int | Unique passenger identifier |
| `age` | float | Passenger age |
| `fare` | float | Ticket fare paid |
| `sex` | string | `male` or `female` |
| `embarked` | string | Port of embarkation: `S` (Southampton), `C` (Cherbourg), `Q` (Queenstown) |
| `parch` | int | Number of parents / children aboard |
| `sibsp` | int | Number of siblings / spouses aboard |
| `pclass` | int | Passenger class: `1`, `2`, or `3` |

**Success response**

```json
{
  "predictions": [
    { "passenger_id": 1, "predicted": "Survived" },
    { "passenger_id": 2, "predicted": "Not Survived" }
  ]
}
```

**Error response**

```json
{ "detail": "Error making predictions: <error message>" }
```

Any inference failure returns HTTP `500` with a descriptive message — never a raw stack trace.

**cURL example**

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '[
        {
          "passenger_id": 1,
          "age": 22.0, "fare": 7.25, "sex": "female",
          "embarked": "S", "parch": 0, "sibsp": 1, "pclass": 3
        }
      ]'
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `APP_NAME` | App title shown in FastAPI docs and the `/` welcome message |
| `VERSION` | API version shown in FastAPI docs |
| `API_SECRET_KEY` | Secret key for future auth middleware |

`.env` is listed in `.gitignore` and is never committed.

---

## Roadmap

- [ ] Add `X-API-Key` authentication middleware using `API_SECRET_KEY`
- [ ] Add model versioning & A/B testing support
- [ ] Add request/response logging & monitoring
- [ ] Containerize with Docker for easier deployment
- [ ] Add unit & integration tests

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML** | TensorFlow / Keras, scikit-learn, joblib |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Config** | python-dotenv |

---

<div align="center">
<sub>RMS · 1912 · built with FastAPI, TensorFlow &amp; Pydantic</sub>
</div>