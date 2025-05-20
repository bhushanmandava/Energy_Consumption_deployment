import os
import mlflow
import wandb
import pandas as pd
#we want some thing to validate our schema
import pandera as pa
from pandera import Check,Column, DataFrameSchema
from pandera.errors import SchemaErrors

wandb.init(project="Energy-Consumption-Pred-101",name = 'data-Validation')
def load_data():
    #load our data from the processed data artifact
    artifact = wandb.use_artifact('bhushanmandava16-personal/Energy-Consumption-Pred-101/processed_data.csv:v0',type = 'dataset')
    artifact_dir = artifact.download()
    data = pd.read_csv(os.path.join(artifact_dir, 'processed_data.csv'))
    return data
def validate_data(data):
    energy_schema = DataFrameSchema(
    {
        "Timestamp": Column(pa.DateTime),
        "Temperature": Column(pa.Float),
        "Humidity": Column(pa.Float),
        "SquareFootage": Column(pa.Float),
        "Occupancy": Column(pa.Float),  # Was int, now float after scaling
        "RenewableEnergy": Column(pa.Float),
        "HVACUsage": Column(pa.String, checks=[
            Check.isin(["On", "Off"])
        ]),
        "LightingUsage": Column(pa.String, checks=[
            Check.isin(["On", "Off"])
        ]),
        "DayOfWeek": Column(pa.String, checks=[
            Check.isin([
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ])
        ]),
        "Holiday": Column(pa.String, checks=[
            Check.isin(["Yes", "No"])
        ]),
        "EnergyConsumption": Column(pa.Float),
    },
    coerce=True,
    strict='filter'
)
    try:
        energy_schema.validate(data)
        print("✅ Data validation passed.")
        all_checks_passed = True
        # return True
    except SchemaErrors as e:
        print("❌ Data validation failed.")
        print(e)
        # return False
        all_checks_passed = False
    wandb.log({'all_checks_passed': all_checks_passed})
    return all_checks_passed
def main():
    mlflow.start_run(run_name="data_validation")
    data = load_data()
    validation_result = validate_data(data)
    if validation_result:
        print("✅ All data validation checks passed. Proceeding to feature engineering.")
        mlflow.log_param("validation_passed", True)
    else:
        print("❌ Data validation failed. Please review the logs and fix the issues before proceeding.")
        mlflow.log_param("validation_passed", False)
    mlflow.end_run()

if __name__ == '__main__':
    main()
