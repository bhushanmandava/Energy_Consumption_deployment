import os
import mlflow
import mlflow.prophet
import pandas as pd
import numpy as np
import wandb
import joblib
from prophet import Prophet
from mlflow.models.signature import infer_signature

# Enable W&B's new core engine for stability
wandb.require("core")

# Global variables
PROJECT_NAME = "Energy-Consumption-Pred-101"
REGRESSORS = [
    'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
    'HVACUsage', 'RenewableEnergy', 'Holiday', 'isWeekend'
]

def load_data():
    api = wandb.Api()
    artifact = api.artifact(
        'bhushanmandava16-personal/Energy-Consumption-Pred-101/engineered_data:v0',
        type='dataset'
    )
    artifact_dir = artifact.download()
    data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
    
    # Data preprocessing
    data.rename(columns={'Timestamp.1': 'ds', 'EnergyConsumption': 'y'}, inplace=True)
    data.drop('Timestamp', axis=1, inplace=True)
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Train-test split
    max_date = data['ds'].max()
    split_date = max_date - pd.Timedelta(hours=30)
    return (
        data[data['ds'] <= split_date],
        data[data['ds'] > split_date]
    )

def train_model():
    # Safely access config with default values
    config = wandb.config
    print(f"Current config: {dict(config)}")  # Debugging
    
    try:
        # Get config parameters with safe fallbacks
        cps = config.get('changepoint_prior_scale', 0.05)
        sps = config.get('seasonality_prior_scale', 0.05)
        smode = config.get('seasonality_mode', 'additive')
        
        # Model initialization
        model = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            seasonality_mode=smode
        )
        
        # Add regressors before fitting
        for reg in REGRESSORS:
            model.add_regressor(reg)
            
        # Load data and fit model
        train_df, test_df = load_data()
        model.fit(train_df)
        
        # Generate future dataframe
        future_df = model.make_future_dataframe(periods=30, freq='h')
        future_df = future_df.merge(
            test_df[['ds'] + REGRESSORS], 
            on='ds', 
            how='left'
        ).dropna(subset=REGRESSORS)
        
        # Make predictions
        forecast = model.predict(future_df)
        forecast = forecast.merge(test_df[['ds', 'y']], on='ds', how='left')
        forecast['error'] = (forecast['y'] - forecast['yhat']).abs()
        
        # Calculate metrics
        metrics = {
            'MAE': forecast['error'].mean(),
            'RMSE': np.sqrt((forecast['error'] ** 2).mean())
        }
        wandb.log(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        wandb.log({"error": 1})
        raise e

def main(run_type="FAST_RUN"):
    with wandb.init(project=PROJECT_NAME, 
                   name='hyperparameter_tuning',
                   job_type="hyperparameter-tuning",
                   allow_val_change=True) as run:
        
        mlflow.start_run(run_name='hyperparameter_tuning')
        
        # Sweep configuration
        sweep_config = {
            'method': 'random' if run_type == 'FAST_RUN' else 'bayes',
            'metric': {'name': 'RMSE', 'goal': 'minimize'},
            'parameters': {
                'changepoint_prior_scale': {
                    'values': [0.01, 0.1] if run_type == 'FAST_RUN' 
                    else [0.001, 0.01, 0.1, 0.5, 1.0]
                },
                'seasonality_prior_scale': {
                    'values': [0.1, 1.0] if run_type == 'FAST_RUN' 
                    else [0.01, 0.1, 1.0, 10.0]
                },
                'seasonality_mode': {
                    'values': ['additive'] if run_type == 'FAST_RUN' 
                    else ['additive', 'multiplicative']
                }
            }
        }
        
        # Run sweep
        sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
        wandb.agent(sweep_id, function=train_model, count=2 if run_type == "FAST_RUN" else 10)
        
        # Find best run
        api = wandb.Api()
        sweep = api.sweep(f"{run.entity}/{PROJECT_NAME}/{sweep_id}")
        best_run = sweep.best_run()
        print(f"Best run config: {best_run.config}")
        
        # Retrain best model
        best_config = best_run.config
        best_model = Prophet(
            changepoint_prior_scale=best_config['changepoint_prior_scale'],
            seasonality_prior_scale=best_config['seasonality_prior_scale'],
            seasonality_mode=best_config['seasonality_mode']
        )
        
        # Add regressors and fit
        for reg in REGRESSORS:
            best_model.add_regressor(reg)
        train_df, test_df = load_data()
        best_model.fit(train_df)
        
        # Load baseline model (WITHOUT adding regressors post-load)
        baseline_model = None
        try:
            artifact = api.artifact(
                'bhushanmandava16-personal/Energy-Consumption-Pred-101/prophet_model:v6',
                type='model'
            )
            baseline_model = joblib.load(artifact.file())
        except Exception as e:
            print(f"Baseline load error: {str(e)}")
        
        # Model comparison
        future_df = best_model.make_future_dataframe(periods=30, freq='h')
        future_df = future_df.merge(
            test_df[['ds'] + REGRESSORS], 
            on='ds', 
            how='left'
        ).dropna(subset=REGRESSORS)
        
        # Evaluate best model
        best_forecast = best_model.predict(future_df)
        best_metrics = {
            'MAE': (best_forecast['yhat'] - test_df['y']).abs().mean(),
            'RMSE': np.sqrt(((best_forecast['yhat'] - test_df['y']) ** 2).mean())
        }
        
        # Compare with baseline if available
        production_model = best_model
        if baseline_model:
            baseline_forecast = baseline_model.predict(future_df)
            baseline_metrics = {
                'MAE': (baseline_forecast['yhat'] - test_df['y']).abs().mean(),
                'RMSE': np.sqrt(((baseline_forecast['yhat'] - test_df['y']) ** 2).mean())
            }
            
            if best_metrics['RMSE'] < baseline_metrics['RMSE']:
                print(f"New model better: {best_metrics['RMSE']:.2f} vs {baseline_metrics['RMSE']:.2f}")
            else:
                print(f"Baseline better: {baseline_metrics['RMSE']:.2f} vs {best_metrics['RMSE']:.2f}")
                production_model = baseline_model
        
        # Log final model
        signature = infer_signature(
            future_df[['ds'] + REGRESSORS],
            best_forecast[['yhat']]
        )
        
        mlflow.prophet.log_model(
            production_model,
            "production_model",
            registered_model_name="prod_model",
            signature=signature,
            input_example=future_df[['ds'] + REGRESSORS].iloc[:5]
        )
        
        # Save to W&B
        model_artifact = wandb.Artifact("production_model", type="model")
        with model_artifact.new_file("model.joblib") as f:
            joblib.dump(production_model, f)
        wandb.log_artifact(model_artifact)
        
        mlflow.end_run()

if __name__ == "__main__":
    main(run_type="FAST_RUN")
