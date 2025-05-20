import os
import pandas as pd
import mlflow
import wandb

# Initialize wandb at the top (only once)
wandb.init(project="Energy-Consumption-Pred-101", name='feature-engineering')

def load_data():
    try:
        artifact = wandb.use_artifact(
            'bhushanmandava16-personal/Energy-Consumption-Pred-101/processed_data.csv:v0',
            type='dataset'
        )
        artifact_dir = artifact.download()
        data = pd.read_csv(os.path.join(artifact_dir, 'processed_data.csv'))
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def feature_engineering(data):
    data['HVACUsage'] = data['HVACUsage'].replace({'Off': 0, 'On': 1})
    data['LightingUsage'] = data['LightingUsage'].replace({'Off': 0, 'On': 1})
    data['Holiday'] = data['Holiday'].replace({'No': 0, 'Yes': 1})
    data['isWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    data.drop(columns=['DayOfWeek'], inplace=True)
    # wandb logging
    wandb.log({'transformed_features_count': data.shape[1]})  # logs number of columns
    wandb.log({'data_sample': wandb.Table(dataframe=data.head())})  # logs a sample as a table
    return data

def save_data(data):
    try:
        transformed_data_path = 'engineered_data.csv'
        data.to_csv(transformed_data_path, index=False)
        artifact = wandb.Artifact(name='engineered_data', type='dataset')
        artifact.add_file(transformed_data_path)
        wandb.log_artifact(artifact)
        wandb.log({'engineered_data_shape': data.shape})
        print("Engineered data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    mlflow.start_run(run_name="feature_engineering")
    try:
        data = load_data()
        if data is None:
            print("❌ Data loading failed. Exiting feature engineering.")
            return
        feature_engineered_data = feature_engineering(data)
        if feature_engineered_data is None:
            print("❌ Feature engineering failed. Exiting.")
            return
        save_data(feature_engineered_data)
        mlflow.log_param("feature_engineered_data_shape", str(feature_engineered_data.shape))
        print("✅ Feature engineering completed successfully.")
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
