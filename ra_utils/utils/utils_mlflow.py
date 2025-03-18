import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import mlflow.pytorch



def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def find_experiment_id(config):
    # Replace with your actual experiment name
    experiment_name = config["experiment_name"]

    # Get the experiment details
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        experiment_id = experiment.experiment_id
        print(f"Experiment ID: {experiment_id}")
    else:
        print("Experiment not found.")
        raise ValueError("Experiment not found.")
    return experiment_id
    
    

def find_run_id(config, experiment_id):
    # Initialize the MLflow client
    client = MlflowClient()

    # Replace with your actual experiment ID and run name
    run_name = config["run_name"]

    # Search for runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment_id])

    # Iterate through the runs to find the one with the matching run name
    run_id = None
    for run in runs:
        if run.data.tags.get('mlflow.runName') == run_name:
            run_id = run.info.run_id
            print(f"Found Run ID: {run_id}")
            break

    if run_id is None:
        raise ValueError("Run not found.")

    return run_id

