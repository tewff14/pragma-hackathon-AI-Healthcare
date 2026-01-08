from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

from app import App

# Initialize FastAPI app
api = FastAPI(
    title="Treatment Recommendation API",
    description="API for patient vital signs monitoring and treatment recommendations",
    version="1.0.0"
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the App instance
app_instance = App()


# Response Models
class PatientVitals(BaseModel):
    patient_id: int
    heart_rate: List[float]
    spo2: List[float]
    respiratory_rate: List[float]
    blood_pressure: List[dict]
    temperature: List[float]
    timestamps: Optional[List[str]] = None


class PredictionRequest(BaseModel):
    patient_id: int


class PredictionResponse(BaseModel):
    iv_dose: str
    vasopressor_dose: str


class PredictionListResponse(BaseModel):
    patient_id: int
    predictions: List[dict]


@api.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Treatment Recommendation API is running"}


@api.get("/patients")
async def get_patients():
    """Get list of all patient IDs"""
    try:
        patient_list = app_instance.get_patient_list()
        return {"patient_ids": patient_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/patient/{patient_id}/vitals")
async def get_patient_vitals(patient_id: int):
    """
    Get all vital signs for a specific patient organized by state/record.
    Returns vitals as a list of records for easy iteration.
    """
    try:
        df = app_instance.df
        
        # Verify patient exists
        patient_list = app_instance.get_patient_list()
        if patient_id not in patient_list:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Filter data for specific patient
        patient_df = df[df['icustayid'] == patient_id]
        
        # Get timestamps if available
        timestamps = []
        if 'readable_charttime' in patient_df.columns:
            timestamps = patient_df['readable_charttime'].tolist()
        elif 'charttime' in patient_df.columns:
            timestamps = patient_df['charttime'].astype(str).tolist()
        
        # Build vitals list - one dict per state/record
        vitals = []
        for i in range(len(patient_df)):
            row = patient_df.iloc[i]
            vitals.append({
                "hr": float(row['HR']) if not pd.isna(row['HR']) else None,
                "rr": float(row['RR']) if not pd.isna(row['RR']) else None,
                "spo2": float(row['SpO2']) if not pd.isna(row['SpO2']) else None,
                "sbp": float(row['SysBP']) if not pd.isna(row['SysBP']) else None,
                "dbp": float(row['DiaBP']) if not pd.isna(row['DiaBP']) else None,
                "temp": float(row['Temp_C']) if not pd.isna(row['Temp_C']) else None,
                "timestamp": timestamps[i] if i < len(timestamps) else None
            })
        
        return {
            "patient_id": patient_id,
            "num_records": len(patient_df),
            "survive_status": app_instance.get_survive_status(patient_id),
            "vitals": vitals
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/patient/{patient_id}/summary")
async def get_patient_summary(patient_id: int):
    """
    Get a summary of patient vitals with statistics (min, max, avg, latest).
    """
    try:
        df = app_instance.df
        
        # Verify patient exists
        patient_list = app_instance.get_patient_list()
        if patient_id not in patient_list:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        patient_df = df[df['icustayid'] == patient_id]
        
        def get_stats(series):
            valid = series.dropna()
            if len(valid) == 0:
                return {"min": None, "max": None, "avg": None, "latest": None}
            return {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "avg": float(valid.mean()),
                "latest": float(valid.iloc[-1])
            }
        
        return {
            "patient_id": patient_id,
            "record_count": len(patient_df),
            "vitals": {
                "heart_rate": get_stats(patient_df['HR']),
                "spo2": get_stats(patient_df['SpO2']),
                "respiratory_rate": get_stats(patient_df['RR']),
                "systolic_bp": get_stats(patient_df['SysBP']),
                "diastolic_bp": get_stats(patient_df['DiaBP']),
                "temperature": get_stats(patient_df['Temp_C'])
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/patient/{patient_id}/predict", response_model=PredictionResponse)
async def predict_treatment(patient_id: int):
    """
    Get baseline clinician treatment recommendation (random historical physician action).
    Returns recommended IV fluid and vasopressor doses based on historical data.
    """
    try:
        df = app_instance.df
        
        # Verify patient exists
        patient_list = app_instance.get_patient_list()
        if patient_id not in patient_list:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Get latest patient data
        patient_df = df[df['icustayid'] == patient_id]
        
        # Use the latest record for prediction
        latest_record = patient_df.iloc[[-1]]
        
        # Call prediction (baseline - random physician action)
        result = app_instance.predict(latest_record)
        print(f"[DEBUG] Predict result: {result}")
        
        if "error" in result:
            print(f"[ERROR] Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        return PredictionResponse(
            iv_dose=result["iv_dose"],
            vasopressor_dose=result["vasopressor_dose"]
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in predict_treatment: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/patient/{patient_id}/predict-personalized", response_model=PredictionListResponse)
async def predict_treatment_personalized(patient_id: int):
    """
    Get AI-optimized personalized treatment recommendations using SAC model.
    Returns a list of recommended IV fluid and vasopressor doses for ALL patient states.
    """
    try:
        df = app_instance.df
        
        # Verify patient exists
        patient_list = app_instance.get_patient_list()
        if patient_id not in patient_list:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Get all patient records
        patient_df = df[df['icustayid'] == patient_id]
        
        # Get timestamps if available
        timestamps = []
        if 'readable_charttime' in patient_df.columns:
            timestamps = patient_df['readable_charttime'].tolist()
        elif 'charttime' in patient_df.columns:
            timestamps = patient_df['charttime'].astype(str).tolist()
        
        # Process each record and get predictions
        predictions = []
        for i in range(len(patient_df)):
            record = patient_df.iloc[[i]]
            
            # Call personalized prediction (AI model)
            result = app_instance.predict_personalized(record)
            
            if "error" in result:
                print(f"[ERROR] Prediction error at state {i+1}: {result['error']}")
                predictions.append({
                    "state": i + 1,
                    "timestamp": timestamps[i] if i < len(timestamps) else None,
                    "iv_dose": "N/A",
                    "vasopressor_dose": "N/A",
                    "error": result["error"]
                })
            else:
                predictions.append({
                    "state": i + 1,
                    "timestamp": timestamps[i] if i < len(timestamps) else None,
                    "iv_dose": result["iv_dose"],
                    "vasopressor_dose": result["vasopressor_dose"]
                })
        
        print(f"[DEBUG] Generated {len(predictions)} predictions for patient {patient_id}")
        
        return PredictionListResponse(
            patient_id=patient_id,
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in predict_treatment_personalized: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



