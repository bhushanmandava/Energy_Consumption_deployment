import pandas as pd
import numpy as np
import os
import mlflow
import wandb
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data():
    try:
        artifact = wandb.use_artifact(
            'bhushanmandava16-personal/Energy-Consumption-Pred-101/original_data:v0', 
            type='dataset'
        )
        artifact_dir = artifact.download()
        # artifact_dir = artifact_dir.replace(':', '-')

        data = pd.read_csv(os.path.join(artifact_dir, 'Energy_consumption.csv'))
        wandb.log({'raw_data_shape': data.shape})
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data from wandb: {e}")
        exit(1)

def save_data(data):
    cleaned_data_path = 'processed_data.csv'
    data.to_csv(cleaned_data_path, index=False)
    artifact = wandb.Artifact(name='processed_data.csv', type='dataset')  # Fixed typo
    artifact.add_file(cleaned_data_path)
    wandb.log_artifact(artifact)
    wandb.log({'cleaned_data_shape': data.shape})

def clean_data(data):
    # Separate target and features
    target = data['EnergyConsumption']
    timestamp = data['Timestamp']
    
    # Select numeric features excluding target and timestamp
    numeric_data = data.select_dtypes(include=[np.number]).drop(
        columns=['EnergyConsumption'], 
        errors='ignore'
    )
    
    # Imputation
    imputer = SimpleImputer(strategy='mean')
    imputed_data = pd.DataFrame(
        imputer.fit_transform(numeric_data),
        columns=numeric_data.columns
    )
    
    # Save imputer
    model_path = 'imputer.pkl'
    joblib.dump(imputer, 'imputer.pkl')
    artifact = wandb.Artifact('imputer.pkl', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    # Scaling
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(imputed_data),
        columns=imputed_data.columns
    )
    
    # Save scaler
    model_path = 'scaler.pkl'
    joblib.dump(scaler, 'scaler.pkl')
    artifact = wandb.Artifact('scaler.pkl', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)  
    #    model_path = "prophet_model.joblib"
    #     joblib.dump(model,model_path)
    #     #loging that model as an artifcat
    #     artifact = wandb.Artifact('prophet_model',type = 'model')
    #     artifact.add_file(model_path)
    
    # Reconstruct final dataframe
    non_numeric_data = data.select_dtypes(exclude=[np.number])
    cleaned_data = pd.concat([
        scaled_data,
        non_numeric_data.reset_index(drop=True),
        pd.DataFrame({'EnergyConsumption': target.values}),
        pd.DataFrame({'Timestamp': timestamp.values})
    ], axis=1)
    
    return cleaned_data

def main():
    wandb.init(project="Energy-Consumption-Pred-101", name='data-cleaning')
    mlflow.start_run(run_name ="data_cleaning")
    
    data = load_data()
    cleaned_data = clean_data(data)
    save_data(cleaned_data)
    
    mlflow.log_param("cleaned_data_shape", cleaned_data.shape)  # Fixed variable name
    mlflow.end_run()

if __name__ == "__main__":
    main()
