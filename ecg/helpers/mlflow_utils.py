"""MLflow utilities for experiment tracking."""

import mlflow
from pathlib import Path
from typing import Dict, Any


def setup_mlflow(
    tracking_uri: str = "./mlruns",
    experiment_name: str = "brugada-ecg-classification"
) -> str:
    """
    Initialize MLflow experiment tracking.
    
    Args:
        tracking_uri: URI for MLflow tracking server
        experiment_name: Name of the experiment
        
    Returns:
        experiment_id: ID of the created/loaded experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def log_params_from_config(config: Dict[str, Any], prefix: str = "") -> None:
    """Log parameters from config dict to MLflow."""
    for key, value in config.items():
        if isinstance(value, dict):
            log_params_from_config(value, prefix=f"{prefix}{key}.")
        else:
            mlflow.log_param(f"{prefix}{key}", value)


def log_dataset_info(statistics: Dict[str, Any]) -> None:
    """Log dataset statistics to MLflow."""
    mlflow.log_params({
        "total_samples": statistics['total_samples'],
        "normal_samples": statistics['normal_samples'],
        "brugada_samples": statistics['brugada_samples'],
        "class_ratio": statistics['class_balance_ratio'],
    })
    
    mlflow.log_metric("pos_weight", statistics['pos_weight'])
