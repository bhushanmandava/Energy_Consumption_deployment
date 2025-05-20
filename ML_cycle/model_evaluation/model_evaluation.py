import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import wandb 
import mlflow
import joblib

wandb.init(project="Energy-Consumption-Pred-101", name='model-evaluation-1')

def load_data():
    artifact = wandb.use_artifact(
        'bhushanmandava16-personal/Energy-Consumption-Pred-101/engineered_data:v0',
        type='dataset'
    )
    artifact_dir = artifact.download()
    data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
    print("Data loaded successfully")
    return data
def evaluate_model(data):
    # Define all regressors used in model training
    regressors = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
                  'HVACUsage', 'RenewableEnergy', 'Holiday', 'isWeekend']
    
    # Data preprocessing
    data.rename(columns={'Timestamp.1':'ds', 'EnergyConsumption':'y'}, inplace=True)
    data.drop('Timestamp', axis=1, inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Verify regressor presence
    missing_regressors = [col for col in regressors if col not in data.columns]
    if missing_regressors:
        raise ValueError(f"Missing regressors in data: {missing_regressors}")

    # Train-test split
    max_date = data['ds'].max()
    split_date = max_date - pd.Timedelta(hours=30)
    train_df = data[data['ds'] <= split_date]
    test_df = data[data['ds'] > split_date].copy()

    # Model loading
    artifact = wandb.use_artifact(
        'bhushanmandava16-personal/Energy-Consumption-Pred-101/prophet_model:v10', 
        type='model'
    )
    model_dir = artifact.download()
    model_path = f"{model_dir}/prophet_model.joblib"
    
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        if os.path.exists(model_path):
            print(f"Model file size: {os.path.getsize(model_path)} bytes")
        else:
            print(f"Model file not found at {model_path}")
        raise

    # Prepare future dataframe with all regressors
    future = test_df[['ds'] + regressors].copy()
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Align data for evaluation
    test_df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)
    
    combined = test_df.join(forecast[['yhat']], how='inner')
    
    if combined.empty:
        raise ValueError("No overlapping dates between forecast and test data")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(combined['y'], combined['yhat']))
    mae = mean_absolute_error(combined['y'], combined['yhat'])
    mape = mean_absolute_percentage_error(combined['y'], combined['yhat'])
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape}



def main():
    mlflow.start_run()
    try:
        data = load_data()
        metrics = evaluate_model(data)
        mlflow.log_param("model_type", "Prophet")
        for metric_name, metric_value in metrics.items():
            print(f"Logged metric: {metric_name} = {metric_value}")
            mlflow.log_metric(metric_name, metric_value)
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
