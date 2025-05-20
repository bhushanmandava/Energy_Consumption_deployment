import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import random

# Set your FastAPI backend URL (use container name if using Docker Compose)
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.title("🔋 Energy Consumption Prediction")

def random_input():
    """Generate a random input dictionary for the prediction."""
    timestamp = (datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "Temperature": round(random.uniform(20, 30), 2),
        "Humidity": round(random.uniform(40, 60), 2),
        "SquareFootage": round(random.uniform(1000, 2000), 2),
        "Occupancy": random.randint(1, 10),
        "RenewableEnergy": round(random.uniform(1, 25), 2),
        "Timestamp": timestamp,
        "HVACUsage": random.choice(["On", "Off"]),
        "LightingUsage": random.choice(["On", "Off"]),
        "Holiday": random.choice(["Yes", "No"])
    }

def predict_single(data):
    response = requests.post(f"{API_URL}/predict", json=data)
    return response

def predict_batch(instances):
    response = requests.post(f"{API_URL}/batch-predict", json={"instances": instances})
    return response

st.header("Single Prediction")

with st.form("single_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.number_input("Temperature (°C)", 15.0, 40.0, 25.0)
        humidity = st.number_input("Humidity (%)", 10.0, 100.0, 50.0)
        square_footage = st.number_input("Square Footage", 500.0, 5000.0, 1500.0)
    with col2:
        occupancy = st.number_input("Occupancy", 1, 100, 5)
        renewable = st.number_input("Renewable Energy (%)", 0.0, 100.0, 10.0)
        timestamp = st.text_input("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with col3:
        hvac = st.selectbox("HVAC Usage", ["On", "Off"])
        lighting = st.selectbox("Lighting Usage", ["On", "Off"])
        holiday = st.selectbox("Holiday", ["Yes", "No"])
    submit_single = st.form_submit_button("Predict Energy Consumption")

if submit_single:
    input_data = {
        "Temperature": temperature,
        "Humidity": humidity,
        "SquareFootage": square_footage,
        "Occupancy": occupancy,
        "RenewableEnergy": renewable,
        "Timestamp": timestamp,
        "HVACUsage": hvac,
        "LightingUsage": lighting,
        "Holiday": holiday
    }
    with st.spinner("Predicting..."):
        resp = predict_single(input_data)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Predicted Energy Consumption: {result['energy_consumption']:.2f} units")
            st.write(f"Timestamp: {result['timestamp']}")
        else:
            st.error(f"Error: {resp.text}")

st.header("Batch Prediction")

batch_size = st.slider("Batch size", 2, 10, 3)
if st.button("Generate Random Batch & Predict"):
    batch = [random_input() for _ in range(batch_size)]
    st.write("Batch Input Data", pd.DataFrame(batch))
    with st.spinner("Predicting batch..."):
        resp = predict_batch(batch)
        if resp.status_code == 200:
            results = resp.json()["predictions"]
            df = pd.DataFrame([{
                **batch[i],
                "Predicted Energy Consumption": results[i]["energy_consumption"],
                "Timestamp": results[i]["timestamp"]
            } for i in range(len(results))])
            st.success("Batch predictions received!")
            st.dataframe(df)
        else:
            st.error(f"Error: {resp.text}")

st.markdown("---")
st.info("This app connects to your FastAPI backend for live predictions.")
