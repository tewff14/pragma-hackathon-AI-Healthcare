import requests
import sys

BASE_URL = "http://localhost:8000"


def test_patient_states(patient_id: int):
    """
    Fetch patient vitals and loop through each state,
    displaying vitals and prediction results.
    """
    print(f"\n{'='*50}")
    print(f"Testing Patient ID: {patient_id}")
    print(f"{'='*50}\n")
    
    # 1. Get patient vitals
    vitals_url = f"{BASE_URL}/patient/{patient_id}/vitals"
    try:
        response = requests.get(vitals_url)
        response.raise_for_status()
        vitals = response.json()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Make sure the server is running.")
        print("Run: uvicorn main:api --reload")
        return
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: {e}")
        print(f"Response: {response.json()}")
        return
    
    # 2. Get baseline prediction (random physician action)
    predict_url = f"{BASE_URL}/patient/{patient_id}/predict"
    try:
        predict_response = requests.post(predict_url)
        predict_response.raise_for_status()
        prediction = predict_response.json()
    except requests.exceptions.HTTPError as e:
        error_detail = predict_response.json().get("detail", "Unknown error")
        print(f"WARNING: Baseline prediction failed - {e}")
        print(f"         Error detail: {error_detail}")
        prediction = {"iv_dose": "N/A", "vasopressor_dose": "N/A"}
    
    # 3. Get personalized AI predictions (for ALL states)
    predict_ai_url = f"{BASE_URL}/patient/{patient_id}/predict-personalized"
    ai_predictions = []
    try:
        predict_ai_response = requests.post(predict_ai_url)
        predict_ai_response.raise_for_status()
        prediction_ai_data = predict_ai_response.json()
        ai_predictions = prediction_ai_data.get("predictions", [])
    except requests.exceptions.HTTPError as e:
        error_detail = predict_ai_response.json().get("detail", "Unknown error")
        print(f"WARNING: AI prediction failed - {e}")
        print(f"         Error detail: {error_detail}")
    
    # 4. Loop through each state
    hr_list = vitals.get("heart_rate", [])
    rr_list = vitals.get("respiratory_rate", [])
    spo2_list = vitals.get("spo2", [])
    bp_list = vitals.get("blood_pressure", [])
    timestamps = vitals.get("timestamps", [])
    
    num_states = len(hr_list)
    
    if num_states == 0:
        print("No data found for this patient.")
        return
    
    print(f"Found {num_states} states for patient {patient_id}\n")
    
    for i in range(num_states):
        hr = hr_list[i] if i < len(hr_list) else "N/A"
        rr = rr_list[i] if i < len(rr_list) else "N/A"
        spo2 = spo2_list[i] if i < len(spo2_list) else "N/A"
        
        if i < len(bp_list):
            sys_bp = bp_list[i].get("systolic", "N/A")
            dia_bp = bp_list[i].get("diastolic", "N/A")
            bp = f"{sys_bp}/{dia_bp}"
        else:
            bp = "N/A"
        
        timestamp = timestamps[i] if timestamps and i < len(timestamps) else ""
        
        # Get AI prediction for this state
        if i < len(ai_predictions):
            ai_pred = ai_predictions[i]
            ai_iv = ai_pred.get("iv_dose", "N/A")
            ai_vaso = ai_pred.get("vasopressor_dose", "N/A")
        else:
            ai_iv = "N/A"
            ai_vaso = "N/A"
        
        print(f"patient id: {patient_id}")
        print(f"state{i + 1}" + (f" ({timestamp})" if timestamp else ""))
        print(f"hr: {hr}")
        print(f"rr: {rr}")
        print(f"spo2: {spo2}")
        print(f"bp: {bp}")
        print(f"[Baseline] predict results: IV={prediction['iv_dose']}, Vaso={prediction['vasopressor_dose']}")
        print(f"[AI Model] predict results: IV={ai_iv}, Vaso={ai_vaso}")
        print("-" * 40)


def main():
    # Get patient ID from terminal input
    if len(sys.argv) > 1:
        try:
            patient_id = int(sys.argv[1])
        except ValueError:
            print("ERROR: Patient ID must be an integer")
            sys.exit(1)
    else:
        try:
            patient_id = int(input("Enter patient ID: "))
        except ValueError:
            print("ERROR: Patient ID must be an integer")
            sys.exit(1)
    
    test_patient_states(patient_id)


if __name__ == "__main__":
    main()

