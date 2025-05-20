import os
import mlflow.prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error,root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
import pandas as pd
import numpy as np
import mlflow
import wandb
# from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import joblib


wandb.init(project="Energy-Consumption-Pred-101", name='Model-training')
def load_data():
    try:
        artifact = wandb.use_artifact(
            'bhushanmandava16-personal/Energy-Consumption-Pred-101/engineered_data:v0',
            type='dataset'
        )
        artifact_dir = artifact.download()
        data = pd.read_csv(os.path.join(artifact_dir, 'engineered_data.csv'))
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_model(train_df,test_df):
    model = Prophet()
    model.add_regressor('Temperature')
    model.add_regressor('Humidity')
    model.add_regressor('SquareFootage')
    model.add_regressor('Occupancy')
    model.add_regressor('HVACUsage')
    model.add_regressor('RenewableEnergy')
    model.add_regressor('Holiday')
    model.add_regressor('isWeekend')
    # model.add_regressor('Temperature')
    model.fit(train_df)
    #till then we fit our model for the training data set now to get our forcast we need to create a dtaframe 
    future_df = model.make_future_dataframe(periods=30,freq='h')
    regressors = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
              'HVACUsage', 'RenewableEnergy', 'Holiday', 'isWeekend']
    future_df = future_df.merge(test_df[['ds'] + regressors], on='ds', how='left')
    # future_df = future_df.merge(train_df[['ds'] + regressors], on='ds', how='left')
    #after done with creating the dataframe we can get started with Forecast 
    # print(future_df.info())
    print(test_df.info())
    print(test_df.head())
    check_missing_values(future_df, test_df)
    future_df=future_df[future_df['ds'] > train_df['ds'].max()]
    forecast = model.predict(future_df)
    #now we got our forecast we can do our mretrics evaluation
    if test_df.index.name == 'ds':
        test_df = test_df.reset_index()
    forecast = forecast.merge(test_df[['ds', 'y']], on='ds', how='left')
    print(forecast)
    #now creating the forecast error
    forecast['error'] = (forecast['y']-forecast['yhat']).abs()
    metrics ={
        'MAE':forecast['error'].mean(),
        'RMSE':np.sqrt((forecast['error']**2).mean())
    }
    print(metrics)
    wandb.log(metrics)
    X_input = future_df[['ds']+regressors]
    y_output = forecast[['yhat']]
    signature =infer_signature(X_input,y_output)
    return model ,metrics, signature
def check_missing_values(train_df, test_df):
    # Count missing values in the training dataset
    missing_train = train_df[['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'HVACUsage', 'RenewableEnergy', 'Holiday', 'isWeekend']].isnull().sum()
    print(f"Missing values in training data:\n{missing_train}")
    
    # Count missing values in the test dataset
    missing_test = test_df[['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'HVACUsage', 'RenewableEnergy', 'Holiday', 'isWeekend']].isnull().sum()
    print(f"Missing values in test data:\n{missing_test}")
    
# Call this function before training the model to check for missing values


def main():
    mlflow.start_run(run_name='hyperparameter_tuning')
    data = load_data()
    data.rename(columns={'Timestamp.1':'ds','EnergyConsumption':'y'},inplace=True)
    data.drop('Timestamp',axis=1,inplace=True)
    #spliting the data
    data['ds'] = pd.to_datetime(data['ds'])
    max_date = data['ds'].max()
    split_date = max_date - pd.Timedelta(hours = 30)
    #sliting our test data to predict next 30 hours
    train_df = data[data['ds']<=split_date]
    test_df = data[data['ds']>split_date]
    check_missing_values(train_df, test_df)
    model , metrics , signature = train_model(train_df,test_df)
    mlflow.log_param("model_type","prophet")
    # params = model.get_params()
    model_params = {
    "growth": model.growth,
    "changepoint_range": model.changepoint_range,
    "n_changepoints": model.n_changepoints,
    "yearly_seasonality": model.yearly_seasonality,
    "weekly_seasonality": model.weekly_seasonality,
    "daily_seasonality": model.daily_seasonality,
    "seasonality_mode": model.seasonality_mode,
    "seasonality_prior_scale": model.seasonality_prior_scale,
    "changepoint_prior_scale": model.changepoint_prior_scale
}

    for key, value in model_params.items():
        mlflow.log_param(key, value)

        # for key , value in params.items():
        #     mlflow.log_param(key,value)
    for metric_name , metric_value in metrics.items():
        mlflow.log_metric(metric_name,metric_value)
        # model_path = "prophet_model.json"
        # model.save_model(model_path)
        model_path = "prophet_model.joblib"
        joblib.dump(model,model_path)
        #loging that model as an artifcat
        artifact = wandb.Artifact('prophet_model',type = 'model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        mlflow.log_artifact(model_path)

        mlflow.prophet.log_model(
            model,
            'prophet_model',
            signature = signature,
            input_example = train_df.iloc[:5],
            registered_model_name = "prophet Baseline"
        )
        mlflow.end_run()
if __name__ ==  "__main__":
    main()

