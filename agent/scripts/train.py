#!/usr/bin/env python3
"""
Train Brugada ECG classifier.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/custom_training.yaml
    python scripts/train.py --experiment-name my-experiment
"""

import argparse
import sys
from pathlib import Path
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logfire
import mlflow
from data import DataConfig, AugmentationConfig, BrugadaDataModule
from models import create_inception1d, BrugadaClassifier
from helpers import setup_logfire, setup_mlflow, log_params_from_config


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Brugada ECG classifier')
    parser.add_argument(
        '--data-config',
        type=Path,
        default=Path('config/data.yaml'),
        help='Path to data config file'
    )
    parser.add_argument(
        '--training-config',
        type=Path,
        default=Path('config/train.yaml'),
        help='Path to training config file'
    )
    parser.add_argument(
        '--mlflow-config',
        type=Path,
        default=Path('config/mlflow.yaml'),
        help='Path to MLflow config file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name (overrides config)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name'
    )
    return parser.parse_args()


def create_callbacks(training_config: dict, checkpoint_dir: Path) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_params = training_config['training']['checkpoint']
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_params['filename'],
        monitor=checkpoint_params['monitor'],
        mode=checkpoint_params['mode'],
        save_top_k=checkpoint_params['save_top_k'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_params = training_config['training']['early_stopping']
    early_stop_callback = EarlyStopping(
        monitor=early_stop_params['monitor'],
        patience=early_stop_params['patience'],
        mode=early_stop_params['mode'],
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configurations
    print("=" * 80)
    print("BRUGADA ECG CLASSIFIER - TRAINING")
    print("=" * 80)
    
    print("\n[1/7] Loading configurations...")
    data_config_dict = load_config(args.data_config)
    training_config = load_config(args.training_config)
    mlflow_config = load_config(args.mlflow_config)
    
    # Set random seed
    seed = training_config.get('seed', 42)
    pl.seed_everything(seed, workers=True)
    print(f"✓ Random seed set to {seed}")
    
    # Setup logging
    print("\n[2/7] Initializing Logfire...")
    setup_logfire(send_to_logfire=False)
    logfire.info('Training started', seed=seed)
    
    # Setup MLflow
    print("\n[3/7] Setting up MLflow...")
    experiment_name = args.experiment_name or mlflow_config['experiment_name']
    setup_mlflow(
        tracking_uri=mlflow_config['tracking_uri'],
        experiment_name=experiment_name
    )
    
    # Create MLflow logger for Lightning
    run_name = args.run_name or f"{mlflow_config['run_name_prefix']}-train"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_config['tracking_uri'],
        run_name=run_name
    )
    
    # Log all configs to MLflow
    log_params_from_config(data_config_dict, prefix="data.")
    log_params_from_config(training_config, prefix="training.")
    
    print(f"✓ MLflow run: {run_name}")
    
    # Create data module
    print("\n[4/7] Preparing data...")
    data_config = DataConfig(**data_config_dict)
    
    # Create augmentation config
    aug_params = data_config_dict.get('augmentation', {})
    augmentation_config = AugmentationConfig(
        amplitude_scale_range=tuple(aug_params.get('amplitude_scale', {}).get('range', [0.8, 1.2])),
        amplitude_scale_prob=aug_params.get('amplitude_scale', {}).get('prob', 0.5),
        noise_std=aug_params.get('noise', {}).get('std', 0.05),
        noise_prob=aug_params.get('noise', {}).get('prob', 0.5),
        baseline_wander_amplitude=aug_params.get('baseline_wander', {}).get('amplitude', 0.1),
        baseline_wander_frequency=tuple(aug_params.get('baseline_wander', {}).get('frequency_range', [0.1, 0.5])),
        baseline_wander_prob=aug_params.get('baseline_wander', {}).get('prob', 0.3),
        time_warp_sigma=aug_params.get('time_warp', {}).get('sigma', 0.2),
        time_warp_knots=aug_params.get('time_warp', {}).get('knots', 4),
        time_warp_prob=aug_params.get('time_warp', {}).get('prob', 0.2),
        lead_scale_range=tuple(aug_params.get('lead_scale', {}).get('range', [0.9, 1.1])),
        lead_scale_prob=aug_params.get('lead_scale', {}).get('prob', 0.3),
    )
    
    datamodule = BrugadaDataModule(
        config=data_config,
        augmentation_config=augmentation_config
    )
    datamodule.setup(stage='fit')
    
    # Get pos_weight for loss
    pos_weight = datamodule.get_pos_weight()
    print(f"✓ Data loaded: {datamodule.statistics.total_samples} samples")
    print(f"  Train: {len(datamodule.train_metadata)}, Val: {len(datamodule.val_metadata)}")
    print(f"  Pos weight: {pos_weight:.3f}")
    
    # Log dataset stats
    mlflow.log_metric("pos_weight", pos_weight)
    mlflow.log_metric("total_samples", datamodule.statistics.total_samples)
    
    # Create model
    print("\n[5/7] Building model...")
    model_config = training_config['model']
    base_model = create_inception1d(
        model_size=model_config['size'],
        in_channels=12,
        num_classes=1,
        dropout=model_config['dropout']
    )
    
    # Setup loss function
    loss_config = training_config.get('loss', {})
    loss_type = loss_config.get('type', 'weighted_bce')
    
    # Prepare loss parameters based on type
    loss_params = {}
    if loss_type == 'weighted_bce':
        # Use pos_weight from config or compute from data
        loss_params['pos_weight'] = loss_config.get('pos_weight', pos_weight)
    elif loss_type == 'focal':
        loss_params['alpha'] = loss_config.get('alpha', 0.25)
        loss_params['gamma'] = loss_config.get('gamma', 2.0)
    # For 'bce', no parameters needed
    
    # Wrap in Lightning module
    scheduler_params = training_config['training'].get('scheduler', {})
    if scheduler_params.get('type') == 'none':
        scheduler_params = None
    
    lightning_model = BrugadaClassifier(
        model=base_model,
        loss_type=loss_type,
        loss_params=loss_params,
        learning_rate=training_config['training']['learning_rate'],
        weight_decay=training_config['training']['weight_decay'],
        scheduler_params=scheduler_params
    )
    
    num_params = base_model.get_num_params()
    print(f"✓ Model created: {model_config['architecture']} ({model_config['size']})")
    print(f"  Parameters: {num_params:,}")
    print(f"  Loss function: {loss_type}")
    if loss_type == 'weighted_bce':
        print(f"    pos_weight: {loss_params.get('pos_weight', 1.0):.3f}")
    elif loss_type == 'focal':
        print(f"    alpha: {loss_params.get('alpha', 0.25):.3f}")
        print(f"    gamma: {loss_params.get('gamma', 2.0):.3f}")
    
    mlflow.log_param("model_architecture", model_config['architecture'])
    mlflow.log_param("model_size", model_config['size'])
    mlflow.log_param("loss_type", loss_type)
    for param_name, param_value in loss_params.items():
        mlflow.log_param(f"loss_{param_name}", param_value)
    mlflow.log_metric("num_parameters", num_params)
    
    # Create callbacks
    print("\n[6/7] Setting up training callbacks...")
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    callbacks = create_callbacks(training_config, checkpoint_dir)
    print(f"✓ Callbacks created: checkpoint, early stopping, lr monitor")
    
    # Create trainer
    print("\n[7/7] Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=training_config['training']['max_epochs'],
        accelerator=training_config['hardware']['accelerator'],
        devices=training_config['hardware']['devices'],
        precision=training_config['hardware']['precision'],
        callbacks=callbacks,
        logger=mlf_logger,
        log_every_n_steps=training_config['logging']['log_every_n_steps'],
        deterministic=True,
        enable_progress_bar=True
    )
    
    print(f"✓ Trainer ready")
    print(f"  Max epochs: {training_config['training']['max_epochs']}")
    print(f"  Accelerator: {training_config['hardware']['accelerator']}")
    print(f"  Precision: {training_config['hardware']['precision']}")
    
    # Train!
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    logfire.info('Training starting', max_epochs=training_config['training']['max_epochs'])
    
    trainer.fit(
        model=lightning_model,
        datamodule=datamodule
    )
    
    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest model: {best_model_path}")
    print(f"Best {trainer.checkpoint_callback.monitor}: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    logfire.info('Training complete', best_model_path=best_model_path)
    
    # Log best model to MLflow
    if training_config['logging']['log_model']:
        mlflow.pytorch.log_model(lightning_model, "model")
        print("✓ Model logged to MLflow")
    
    print("\nView results:")
    print("  MLflow UI: mlflow ui")
    print(f"  Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logfire.error('Training failed', error=str(e), exc_info=True)
        raise
