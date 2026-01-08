# Treatment Recommendation API Documentation

## Overview

This API provides patient vital signs monitoring and AI-powered treatment recommendations for sepsis patients in ICU settings. It uses a Soft Actor-Critic (SAC) deep reinforcement learning model to generate personalized IV fluid and vasopressor dosage recommendations.

**Base URL:** `http://localhost:8000`

**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Endpoints

### 1. Health Check

Check if the API is running.

```
GET /
```

**Response:**
```json
{
  "status": "ok",
  "message": "Treatment Recommendation API is running"
}
```

---

### 2. Get All Patients

Retrieve a list of all patient IDs in the database.

```
GET /patients
```

**Response:**
```json
[50100001, 50100002, 50100003, ...]
```

---

### 3. Get Patient Vitals

Retrieve all vital signs for a specific patient across all recorded states.

```
GET /patient/{patient_id}/vitals
```

**Parameters:**
| Name | Type | Location | Description |
|------|------|----------|-------------|
| patient_id | integer | path | Patient's ICU stay ID |

**Response:**
```json
{
  "patient_id": 50100001,
  "heart_rate": [74.6, 74.6, 64.85, 68.5, ...],
  "spo2": [99.0, 99.0, 99.4, 92.6, ...],
  "respiratory_rate": [18.8, 18.8, 21.8, 26.2, ...],
  "blood_pressure": [
    {"systolic": 135.4, "diastolic": 38.2},
    {"systolic": 135.4, "diastolic": 38.2},
    ...
  ],
  "temperature": [36.8, 36.8, 36.7, 36.9, ...],
  "timestamps": ["18:10:00", "22:10:00", "02:10:00", ...]
}
```

**Error Responses:**
- `404` - Patient not found
- `500` - Internal server error

---

### 4. Get Patient Summary

Get statistical summary of patient vitals (min, max, avg, latest values).

```
GET /patient/{patient_id}/summary
```

**Parameters:**
| Name | Type | Location | Description |
|------|------|----------|-------------|
| patient_id | integer | path | Patient's ICU stay ID |

**Response:**
```json
{
  "patient_id": 50100001,
  "record_count": 12,
  "vitals": {
    "heart_rate": {
      "min": 64.0,
      "max": 77.6,
      "avg": 71.2,
      "latest": 76.5
    },
    "spo2": {
      "min": 92.6,
      "max": 99.4,
      "avg": 95.8,
      "latest": 95.0
    },
    "respiratory_rate": {
      "min": 18.8,
      "max": 32.1,
      "avg": 25.3,
      "latest": 23.0
    },
    "systolic_bp": {
      "min": 135.4,
      "max": 180.2,
      "avg": 154.7,
      "latest": 167.0
    },
    "diastolic_bp": {
      "min": 36.3,
      "max": 61.1,
      "avg": 45.2,
      "latest": 51.5
    },
    "temperature": {
      "min": 36.7,
      "max": 36.9,
      "avg": 36.8,
      "latest": 36.8
    }
  }
}
```

---

### 5. Baseline Treatment Prediction

Get baseline treatment recommendation based on historical physician actions (random sample from historical data).

```
POST /patient/{patient_id}/predict
```

**Parameters:**
| Name | Type | Location | Description |
|------|------|----------|-------------|
| patient_id | integer | path | Patient's ICU stay ID |

**Response:**
```json
{
  "iv_dose": "125.543 ml dose of iv fluid",
  "vasopressor_dose": "0.15 ug/kg/min dose of vasopressor"
}
```

**Description:**
This endpoint returns a randomly sampled physician action from historical data. It serves as a baseline/reference for comparison with the AI model.

---

### 6. Personalized AI Treatment Prediction

Get AI-optimized personalized treatment recommendations for ALL patient states using the SAC (Soft Actor-Critic) deep reinforcement learning model.

```
POST /patient/{patient_id}/predict-personalized
```

**Parameters:**
| Name | Type | Location | Description |
|------|------|----------|-------------|
| patient_id | integer | path | Patient's ICU stay ID |

**Response:**
```json
{
  "patient_id": 50100001,
  "predictions": [
    {
      "state": 1,
      "timestamp": "18:10:00",
      "iv_dose": "142.831 ml dose of iv fluid",
      "vasopressor_dose": "0.12 ug/kg/min dose of vasopressor"
    },
    {
      "state": 2,
      "timestamp": "22:10:00",
      "iv_dose": "138.215 ml dose of iv fluid",
      "vasopressor_dose": "0.14 ug/kg/min dose of vasopressor"
    },
    {
      "state": 3,
      "timestamp": "02:10:00",
      "iv_dose": "155.672 ml dose of iv fluid",
      "vasopressor_dose": "0.18 ug/kg/min dose of vasopressor"
    }
  ]
}
```

**Description:**
This endpoint uses the trained SAC ensemble model to generate personalized treatment recommendations based on each patient state. The model considers:
- Vital signs (HR, BP, SpO2, RR, Temperature)
- Lab values (Sodium, Glucose, Creatinine, etc.)
- Clinical scores (SOFA, GCS)

---

## Data Models

### PatientVitals
```typescript
{
  patient_id: number;
  heart_rate: number[];
  spo2: number[];
  respiratory_rate: number[];
  blood_pressure: Array<{systolic: number, diastolic: number}>;
  temperature: number[];
  timestamps: string[] | null;
}
```

### PredictionResponse
```typescript
{
  iv_dose: string;
  vasopressor_dose: string;
}
```

### PredictionListResponse
```typescript
{
  patient_id: number;
  predictions: Array<{
    state: number;
    timestamp: string | null;
    iv_dose: string;
    vasopressor_dose: string;
    error?: string;
  }>;
}
```

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Patient not found |
| 500 | Internal server error |

Error response format:
```json
{
  "detail": "Error message description"
}
```

---

## Quick Start

### Using cURL

```bash
# Health check
curl http://localhost:8000/

# Get all patients
curl http://localhost:8000/patients

# Get patient vitals
curl http://localhost:8000/patient/50100001/vitals

# Get patient summary
curl http://localhost:8000/patient/50100001/summary

# Get baseline prediction
curl -X POST http://localhost:8000/patient/50100001/predict

# Get AI personalized predictions
curl -X POST http://localhost:8000/patient/50100001/predict-personalized
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Get patient vitals
response = requests.get(f"{BASE_URL}/patient/50100001/vitals")
vitals = response.json()

# Get AI predictions
response = requests.post(f"{BASE_URL}/patient/50100001/predict-personalized")
predictions = response.json()

for pred in predictions["predictions"]:
    print(f"State {pred['state']}: IV={pred['iv_dose']}, Vaso={pred['vasopressor_dose']}")
```

---

## Running the Server

### Local Development
```bash
cd backend
uvicorn main:api --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t treatment-api .
docker run -p 8000:8000 treatment-api
```

---

## Model Information

The AI model uses a **Soft Actor-Critic (SAC) Ensemble** architecture with:
- 5 ensemble agents for robust predictions
- Autoencoder for state representation (37 → 24 dimensions)
- Trained on MIMIC-III ICU data

**Treatment outputs:**
- **IV Fluid Dose**: Intravenous fluid volume (ml)
- **Vasopressor Dose**: Vasopressor rate (μg/kg/min)

