import mlflow
import dagshub
import json
from mlflow import MlflowClient

import dagshub
dagshub.init(repo_owner='MohammoD2', repo_name='Uber_Demand_Pridiction-', mlflow=True)

# set the mlflow tracking uri
mlflow.set_tracking_uri("https://dagshub.com/MohammoD2/Uber_Demand_Pridiction-.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# get model name
registered_model_name = 'uber_demand_prediction_model'
stage = "Staging"

# get the latest version from staging stage
client = MlflowClient()

# get the latest version of model in staging
latest_versions = client.search_model_versions(
    f"name='{registered_model_name}' and current_stage='{stage}'",
    max_results=1
)

if not latest_versions:
    raise ValueError(f"No model versions found for {registered_model_name} in {stage} stage")

latest_model_version_staging = latest_versions[0].version

# promotion stage
promotion_stage = "Production"

model_version_prod = client.transition_model_version_stage(
                                                        name=registered_model_name,
                                                        version=latest_model_version_staging,
                                                        stage=promotion_stage,
                                                        archive_existing_versions=True
                                                    )

production_version = model_version_prod.version
new_stage = model_version_prod.current_stage

print(f"The model is moved to the {new_stage} stage having version number {production_version}")