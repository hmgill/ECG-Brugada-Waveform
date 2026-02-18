#!/usr/bin/env python3
"""
K-fold cross-validation for Brugada ECG classifier.

Usage:
    python scripts/cross_validate.py
    python scripts/cross_validate.py --n-folds 10
    python scripts/cross_validate.py --experiment-name kfold-experiment
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
from typing import List, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logfire
import mlflow
from data import DataConfig, AugmentationConfig, BrugadaDataModule, ECGMetadata
from models import BrugadaClassifier, create_inception1d, create_resnet1d, create_ecg_transformer
from helpers import setup_logfire, setup_mlflow, log_params_from_config


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='K-fold cross-validation for Brugada classifier')
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
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name (overrides config)'
    )
    return parser.parse_args()


def create_fold_datamodule(
    all_metadata: List[ECGMetadata],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    data_config: DataConfig,
    augmentation_config: AugmentationConfig
) -> BrugadaDataModule:
    """Create a datamodule for a specific fold."""
    # Create datamodule
    datamodule = BrugadaDataModule(
        config=data_config,
        augmentation_config=augmentation_config
    )
    
    # Manually set train/val splits based on fold indices
    datamodule.metadata_list = all_metadata
    datamodule.train_metadata = [all_metadata[i] for i in train_indices]
    datamodule.val_metadata = [all_metadata[i] for i in val_indices]
    
    # Compute statistics
    from data.models import DatasetStatistics
    datamodule.statistics = DatasetStatistics.from_metadata_list(datamodule.train_metadata)
    
    # Create datasets
    from data.dataset import BrugadaDataset
    datamodule.train_dataset = BrugadaDataset(
        datamodule.train_metadata,
        augmentation_config if data_config.augment_train else None,
        data_config.normalize,
        data_config.normalization_method
    )
    datamodule.val_dataset = BrugadaDataset(
        datamodule.val_metadata,
        None,
        data_config.normalize,
        data_config.normalization_method
    )
    
    return datamodule


def train_fold(
    fold: int,
    datamodule: BrugadaDataModule,
    training_config: dict,
    checkpoint_dir: Path,
    mlflow_run_id: str
) -> Dict[str, float]:
    """Train a single fold and return metrics."""
    print(f"\n{'='*80}")
    print(f"FOLD {fold + 1}")
    print(f"{'='*80}")
    print(f"Train samples: {len(datamodule.train_metadata)}")
    print(f"Val samples: {len(datamodule.val_metadata)}")
    
    # Create model
    model_config = training_config['model']

    if model_config['architecture'] == 'resnet1d':
        base_model = create_resnet1d(
            model_size=model_config['size'],
            in_channels=12,
            num_classes=1,
            dropout=model_config['dropout']
        )
    elif model_config['architecture'] == "transformer":
        base_model = create_ecg_transformer(
            model_size=model_config['size'],
            in_channels=12,
            num_classes=1,
            dropout=model_config['dropout']
        )
    else:  # inception1d
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
        pos_weight_config = loss_config.get('pos_weight', 'auto')
        if pos_weight_config == 'auto' or pos_weight_config is None:
            pos_weight = datamodule.get_pos_weight()
        else:
            pos_weight = float(pos_weight_config)
        loss_params['pos_weight'] = pos_weight
        print(f"Pos weight: {pos_weight:.3f}")
    elif loss_type == 'focal':
        loss_params['alpha'] = loss_config.get('alpha', 0.25)
        loss_params['gamma'] = loss_config.get('gamma', 2.0)
        print(f"Focal Loss - alpha: {loss_params['alpha']:.3f}, gamma: {loss_params['gamma']:.3f}")
    else:
        print(f"Standard BCE (no class weighting)")
    
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
    
    # Create callbacks
    fold_checkpoint_dir = checkpoint_dir / f"fold_{fold}"
    fold_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=fold_checkpoint_dir,
        filename=f"fold{fold}-{{epoch:02d}}-{{val_auroc:.3f}}",
        monitor='val_auroc',
        mode='max',
        save_top_k=1,
        verbose=False
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=training_config['training']['early_stopping']['patience'],
        mode='min',
        verbose=False
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config['training']['max_epochs'],
        accelerator=training_config['hardware']['accelerator'],
        devices=training_config['hardware']['devices'],
        precision=training_config['hardware']['precision'],
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=False,  # Disable for cleaner output
        logger=False,  # No per-fold logging
        deterministic=True,
        enable_model_summary=False
    )
    
    # Train
    trainer.fit(
        model=lightning_model,
        datamodule=datamodule
    )

    print(f"  Training completed at epoch: {trainer.current_epoch}")
    print(f"  Early stopped: {trainer.should_stop}")
    if hasattr(early_stop_callback, 'wait_count'):
        print(f"  Early stop wait count: {early_stop_callback.wait_count}/{early_stop_callback.patience}")
    
    
    # Get best metrics
    best_metrics = {
        'fold': fold,
        'best_epoch': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0,
        'val_auroc': checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 0,
    }
    
    # Load best model and compute all metrics
    if checkpoint_callback.best_model_path:
        best_model = BrugadaClassifier.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=base_model
        )
        
        # Run validation to get all metrics
        trainer_eval = pl.Trainer(
            accelerator=training_config['hardware']['accelerator'],
            devices=training_config['hardware']['devices'],
            logger=False,
            enable_progress_bar=False
        )
        
        # Validate
        val_results = trainer_eval.validate(best_model, datamodule=datamodule, verbose=False)
        
        if val_results:
            metrics = val_results[0]
            best_metrics.update({
                'val_loss': metrics.get('val_loss', 0),
                'val_acc': metrics.get('val_acc', 0),
                'val_precision': metrics.get('val_precision', 0),
                'val_recall': metrics.get('val_recall', 0),
                'val_sensitivity': metrics.get('val_sensitivity', 0),
                'val_f1': metrics.get('val_f1', 0),
                'val_auroc': metrics.get('val_auroc', 0),
                'val_auprc': metrics.get('val_auprc', 0),
            })
    
    # Log fold metrics to parent MLflow run (no nested run needed)
    for metric_name, metric_value in best_metrics.items():
        if metric_name != 'fold':
            mlflow.log_metric(f"fold_{fold}_{metric_name}", metric_value)
    
    print(f"\nFold {fold + 1} Results:")
    print(f"  Val AUROC: {best_metrics.get('val_auroc', 0):.4f}")
    print(f"  Val Accuracy: {best_metrics.get('val_acc', 0):.4f}")
    print(f"  Val Sensitivity: {best_metrics.get('val_sensitivity', 0):.4f}")
    print(f"  Val F1: {best_metrics.get('val_f1', 0):.4f}")
    
    return best_metrics


def main():
    """Main cross-validation function."""
    args = parse_args()
    
    print("=" * 80)
    print("BRUGADA ECG CLASSIFIER - K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    # Load configurations
    print("\n[1/6] Loading configurations...")
    data_config_dict = load_config(args.data_config)
    training_config = load_config(args.training_config)
    mlflow_config = load_config(args.mlflow_config)
    
    # Set random seed
    seed = training_config.get('seed', 42)
    pl.seed_everything(seed, workers=True)
    print(f"✓ Random seed set to {seed}")
    print(f"✓ Number of folds: {args.n_folds}")
    
    # Setup logging
    print("\n[2/6] Initializing Logfire...")
    setup_logfire(send_to_logfire=False)
    logfire.info('Cross-validation started', n_folds=args.n_folds, seed=seed)
    
    # Setup MLflow
    print("\n[3/6] Setting up MLflow...")
    experiment_name = args.experiment_name or f"{mlflow_config['experiment_name']}-kfold"
    setup_mlflow(
        tracking_uri=mlflow_config['tracking_uri'],
        experiment_name=experiment_name
    )
    
    # Start MLflow run for entire CV
    cv_run = mlflow.start_run(run_name=f"cv-{args.n_folds}fold")
    mlflow_run_id = cv_run.info.run_id
    
    # Log configs
    log_params_from_config(data_config_dict, prefix="data.")
    log_params_from_config(training_config, prefix="training.")
    mlflow.log_param("n_folds", args.n_folds)
    mlflow.log_param("cv_strategy", "stratified_kfold")
    
    print(f"✓ MLflow run: cv-{args.n_folds}fold")
    
    # Load all data
    print("\n[4/6] Loading data...")
    import pandas as pd
    from data.models import ECGMetadata
    
    df = pd.read_csv(data_config_dict['metadata_path'])
    all_metadata = [ECGMetadata(**row) for _, row in df.iterrows()]
    
    # Create labels for stratification (brugada >= 1)
    labels = np.array([1 if m.brugada >= 1 else 0 for m in all_metadata])
    
    print(f"✓ Total samples: {len(all_metadata)}")
    print(f"  Normal: {np.sum(labels == 0)}, Brugada: {np.sum(labels == 1)}")
    
    # Create data config
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
    
    # Setup k-fold
    print("\n[5/6] Setting up k-fold cross-validation...")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
    
    checkpoint_dir = Path('checkpoints_cv')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Train each fold
    print("\n[6/6] Training folds...")
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_metadata, labels)):
        # Create datamodule for this fold
        datamodule = create_fold_datamodule(
            all_metadata,
            train_idx,
            val_idx,
            data_config,
            augmentation_config
        )
        
        # Train fold
        metrics = train_fold(
            fold,
            datamodule,
            training_config,
            checkpoint_dir,
            mlflow_run_id
        )
        fold_metrics.append(metrics)
    
    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    metric_names = ['val_auroc', 'val_acc', 'val_sensitivity', 'val_recall', 'val_precision', 'val_f1', 'val_auprc']
    
    print(f"\n{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    aggregate_metrics = {}
    for metric in metric_names:
        values = [m.get(metric, 0) for m in fold_metrics if metric in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            aggregate_metrics[f"{metric}_mean"] = mean_val
            aggregate_metrics[f"{metric}_std"] = std_val
            aggregate_metrics[f"{metric}_min"] = min_val
            aggregate_metrics[f"{metric}_max"] = max_val
            
            print(f"{metric:<20} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
    
    # Log aggregate metrics to MLflow (already in the parent run context)
    for metric_name, metric_value in aggregate_metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    mlflow.end_run()
    
    print(f"\n✓ Cross-validation complete!")
    print(f"  Mean AUROC: {aggregate_metrics.get('val_auroc_mean', 0):.4f} ± {aggregate_metrics.get('val_auroc_std', 0):.4f}")
    print(f"  Mean Sensitivity: {aggregate_metrics.get('val_sensitivity_mean', 0):.4f} ± {aggregate_metrics.get('val_sensitivity_std', 0):.4f}")
    print(f"\nView results:")
    print(f"  MLflow UI: mlflow ui")
    print(f"  Checkpoints: {checkpoint_dir}")
    
    logfire.info('Cross-validation complete', **aggregate_metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logfire.error('Cross-validation failed', error=str(e), exc_info=True)
        raise
