import os
import joblib
import wandb
import logging
import pandas as pd
import random
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List
import requests
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# Define request models
class PredictRequest(BaseModel):
    Temperature: float
    Humidity: float
    SquareFootage: float
    Occupancy: int
    RenewableEnergy: float
    Timestamp: str
    HVACUsage: str
    LightingUsage: str
    Holiday: str

    @field_validator('HVACUsage')
    def validate_hvac_usage(cls, v):
        if v not in ['On', 'Off']:
            raise ValueError('HVACUsage must be either "On" or "Off"')
        return v

    @field_validator('LightingUsage')
    def validate_lighting_usage(cls, v):
        if v not in ['On', 'Off']:
            raise ValueError('LightingUsage must be either "On" or "Off"')
        return v

    @field_validator('Holiday')
    def validate_holiday(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Holiday must be either "Yes" or "No"')
        return v

class BatchPredictRequest(BaseModel):
    instances: List[PredictRequest]

class PredictResponse(BaseModel):
    energy_consumption: float
    timestamp: str

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

def load_artifacts():
    try:
        if not os.getenv('WANDB_API_KEY'):
            raise ValueError("WANDB_API_KEY is not set in the environment variables")
        if not os.getenv("WANDB_PROJECT"):
            raise ValueError("WANDB_PROJECT is not set in the environment variables")
        
        wandb.login()
        run = wandb.init(project=os.getenv('WANDB_PROJECT'), job_type="inference")
        
        artifact = run.use_artifact('prophet_model:latest', type='model')
        model_dir = artifact.download()
        model_path = f"{model_dir}/prophet_model.joblib"
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        scaler_artifact = run.use_artifact('bhushanmandava16-personal/Energy-Consumption-Pred-101/scaler.pkl:v1')
        scaler_dir = scaler_artifact.download()
        scaler_path = os.path.join(scaler_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded successfully")


        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load necessary artifacts: {str(e)}")

try:
    model, scaler = load_artifacts()
    logger.info("Successfully loaded the artifacts")
except Exception as e:
    logger.error(f"Failed to load artifacts: {str(e)}")
    model = None
    scaler = None

def preprocess_data(data):
    """Preprocess input data to match the format expected by the model"""
    try:
        # Create DataFrame from input data
        df = pd.DataFrame([data] if not isinstance(data, list) else data)
        
        # Derive isWeekend from Timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['isWeekend'] = df['Timestamp'].dt.dayofweek.ge(5).astype(int)
        
        # Encode categorical features
        df['HVACUsage'] = df['HVACUsage'].map({'On': 1, 'Off': 0})
        df['LightingUsage'] = df['LightingUsage'].map({'On': 1, 'Off': 0})
        df['Holiday'] = df['Holiday'].map({'Yes': 1, 'No': 0})

        # Define feature groups
        numerical_features = [
            'Temperature', 'Humidity', 'SquareFootage',
            'Occupancy', 'RenewableEnergy'
        ]
        categorical_features = ['HVACUsage', 'LightingUsage', 'Holiday', 'isWeekend']
        
        # Validate feature existence
        missing_features = set(numerical_features + categorical_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Scale numerical features
        scaled_numerical = scaler.transform(df[numerical_features])
        scaled_df = pd.DataFrame(scaled_numerical, columns=numerical_features)
        
        # Combine features
        final_df = pd.concat([
            scaled_df,
            df[categorical_features].reset_index(drop=True)
        ], axis=1)

        # Enforce feature order
        required_order = numerical_features + categorical_features
        final_df = final_df[required_order]

        # Add Prophet's required datetime column
        final_df['ds'] = df['Timestamp']

        # Log final state
        logger.info(f"Final features:\n{final_df.columns.tolist()}")
        logger.info(f"Data sample:\n{final_df.head()}")
        logger.info(f"Data shape: {final_df.shape}")

        # Return DataFrame with columns instead of NumPy array
        return final_df, df['Timestamp']

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise HTTPException(400, detail=f"Data preprocessing failed: {str(e)}")


@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded properly")
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded properly")
    
    try:
        data = request.model_dump()
        processed_df, timestamps = preprocess_data(data)
        
        # Prophet requires the DataFrame with column names
        prediction = model.predict(processed_df)
        
        return PredictResponse(
            energy_consumption=float(prediction['yhat'].values[0]),
            timestamp=timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(request: BatchPredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded properly")
    
    try:
        # Convert instances to list of dictionaries
        data = [instance.model_dump() for instance in request.instances]
        
        # Preprocess data (returns DataFrame with 'ds' and other features)
        processed_df, timestamps = preprocess_data(data)
        
        # Make predictions with Prophet
        forecast = model.predict(processed_df)
        
        # Extract predictions and create responses
        responses = []
        for i, (ts, yhat) in enumerate(zip(timestamps, forecast['yhat'])):
            responses.append(
                PredictResponse(
                    energy_consumption=float(yhat),
                    timestamp=ts.strftime("%Y-%m-%d %H:%M:%S")
                )
            )
        
        return BatchPredictResponse(predictions=responses)
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Testing helpers
def test_health_check():
    response = requests.get("http://localhost:8000/health")
    print(f"Health check status code: {response.status_code}")
    print(f"Health check response: {response.json()}")
    return response.status_code == 200

def send_prediction(num_samples=1):
    test_data = []
    print(f"preparing the samples {num_samples}")
    for _ in range(num_samples):
        data = {
            "Temperature": random.uniform(20, 30),
            "Humidity": random.uniform(40, 60),
            "SquareFootage": random.uniform(1000, 2000),
            "Occupancy": random.randint(1, 10),
            "RenewableEnergy": random.uniform(1, 25),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "HVACUsage": random.choice(["On", "Off"]),
            "LightingUsage": random.choice(["On", "Off"]),
            "Holiday": random.choice(["Yes", "No"])
        }
        test_data.append(data)
        print(f"data is prepared {test_data}")
    
    if num_samples == 1:
        response = requests.post("http://localhost:8000/predict", json=test_data[0])
        print(f"Prediction status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Predicted energy consumption: {response.json()['energy_consumption']}")
        else:
            print(f"Error: {response.text}")
    else:
        batch_request = {"instances": test_data}
        response = requests.post("http://localhost:8000/batch-predict", json=batch_request)
        print(f"Batch prediction status code: {response.status_code}")
        if response.status_code == 200:
            print(f"Number of predictions: {len(response.json()['predictions'])}")
            for i, pred in enumerate(response.json()['predictions']):
                print(f"Prediction {i+1}: {pred['energy_consumption']}")
        else:
            print(f"Error: {response.text}")
    
    return response.status_code == 200

def simulate_user_traffic(num_requests=10, delay=1):
    print(f"Simulating {num_requests} user requests with {delay}s delay between requests")
    success_count = 0
    for i in range(num_requests):
        print(f"\nRequest {i+1}/{num_requests}")
        if random.choice([True, False]):
            success = send_prediction(num_samples=1)
        else:
            success = send_prediction(num_samples=random.randint(2, 5))
        
        if success:
            success_count += 1
        
        if i < num_requests - 1:
            time.sleep(delay)
    
    print(f"\nTraffic simulation complete. Success rate: {success_count}/{num_requests}")

if __name__ == "__main__":
    import uvicorn
    
    print("Testing prediction with sample data...")
    num_samples=1
    send_prediction(num_samples)
    
    print("\nStarting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
# clear
# http://0.0.0.0:8000 