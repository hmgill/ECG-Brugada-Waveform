#!/usr/bin/env python3
"""
Training script for multi-dataset ECG disease detection.

Usage:
    python scripts/train.py --config config/train.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime
from collections import Counter

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import mlflow
import logfire

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.models import DataConfig, AugmentationConfig
from data.datamodule import UnifiedDataModule
from models.ecg_transformer import create_ecg_transformer_rope
from models.losses import get_loss_function
from models.lightning_module import MultiTaskClassifier
from helpers.mlflow_utils import setup_mlflow, log_params_from_config  # ← added


# ── Callbacks ─────────────────────────────────────────────────────────────────

class BestModelMetricsCallback(Callback):
    """Logs the full metric snapshot at the best epoch to MLflow and Logfire."""

    def __init__(self, monitor: str = "val_sup_auroc_macro"):
        self.monitor = monitor
        self.best_score = None
        self.best_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        if self.best_score is None or current > self.best_score:
            self.best_score = current.item()
            self.best_metrics = {
                k: v.item() if hasattr(v, 'item') else v
                for k, v in trainer.callback_metrics.items()
            }
            self.best_metrics['best_epoch'] = trainer.current_epoch

    def on_train_end(self, trainer, pl_module):
        if not self.best_metrics:
            return

        print("\n" + "=" * 60)
        print("BEST MODEL METRICS")
        print("=" * 60)
        for k, v in sorted(self.best_metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        mlflow.log_metrics(
            {f"best/{k}": v for k, v in self.best_metrics.items()
             if isinstance(v, (int, float))}
        )

        logfire.info(
            'Best model metrics',
            monitor=self.monitor,
            **{k: v for k, v in self.best_metrics.items()
               if isinstance(v, (int, float))}
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-dataset ECG classifier')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/train.yaml'),
        help='Path to training config file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='MLflow run name'
    )
    return parser.parse_args()


def create_callbacks(config: dict, checkpoint_dir: Path) -> list:
    """Create training callbacks."""
    callbacks = []

    checkpoint_params = config['training']['checkpoint']
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_params['filename'],
        monitor=checkpoint_params['monitor'],
        mode=checkpoint_params['mode'],
        save_top_k=checkpoint_params['save_top_k'],
        verbose=True,
        save_last=True
    ))

    callbacks.append(BestModelMetricsCallback(
        monitor=config['training']['checkpoint']['monitor']
    ))

    early_stop_params = config['training']['early_stopping']
    callbacks.append(EarlyStopping(
        monitor=early_stop_params['monitor'],
        patience=early_stop_params['patience'],
        mode=early_stop_params['mode'],
        verbose=True
    ))

    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    return callbacks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Main training function."""
    args = parse_args()

    print("=" * 80)
    print("MULTI-DATASET ECG DISEASE DETECTION - TRAINING")
    print("=" * 80)

    # ── [1/7] Config ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading configuration...")
    config = load_config(args.config)

    seed = config['data']['random_seed']
    pl.seed_everything(seed, workers=True)
    print(f"✓ Random seed: {seed}")

    # ── [2/7] Data config ─────────────────────────────────────────────────────
    print("\n[2/7] Preparing data configuration...")
    data_config_dict = config['data']

    data_config = DataConfig(
        use_brugada=data_config_dict['use_brugada'],
        brugada_metadata_path=Path(data_config_dict['brugada_metadata_path']),
        brugada_scp_statements_path=Path(data_config_dict['brugada_scp_statements_path']),
        brugada_data_root=Path(data_config_dict['brugada_data_root']),
        use_ptbxl=data_config_dict['use_ptbxl'],
        ptbxl_metadata_path=Path(data_config_dict['ptbxl_metadata_path']),
        ptbxl_data_root=Path(data_config_dict['ptbxl_data_root']),
        ptbxl_scp_statements_path=Path(data_config_dict['ptbxl_scp_statements_path']),
        ptbxl_sampling_ratio=data_config_dict['ptbxl_sampling_ratio'],
        target_sampling_rate=data_config_dict['target_sampling_rate'],
        target_length_seconds=data_config_dict['target_length_seconds'],
        train_split=data_config_dict['train_split'],
        val_split=data_config_dict['val_split'],
        test_split=data_config_dict['test_split'],
        stratified=data_config_dict['stratified'],
        batch_size=data_config_dict['batch_size'],
        num_workers=data_config_dict['num_workers'],
        pin_memory=data_config_dict['pin_memory'],
        normalize=data_config_dict['normalize'],
        normalization_method=data_config_dict['normalization_method'],
        augment_train=data_config_dict.get('augment_train', True),
        augment_val=data_config_dict.get('augment_val', False),
        random_seed=data_config_dict['random_seed'],
    )

    aug_params = data_config_dict.get('augmentation', {})
    augmentation_config = AugmentationConfig(
        amplitude_scale_range=tuple(aug_params.get('amplitude_scale_range', [0.8, 1.2])),
        amplitude_scale_prob=aug_params.get('amplitude_scale_prob', 0.3),
        noise_std=aug_params.get('noise_std', 0.01),
        noise_prob=aug_params.get('noise_prob', 0.3),
        baseline_wander_amplitude=aug_params.get('baseline_wander_amplitude', 0.05),
        baseline_wander_frequency=tuple(aug_params.get('baseline_wander_frequency', [0.05, 0.5])),
        baseline_wander_prob=aug_params.get('baseline_wander_prob', 0.3),
        time_warp_sigma=aug_params.get('time_warp_sigma', 0.2),
        time_warp_knots=aug_params.get('time_warp_knots', 4),
        time_warp_prob=aug_params.get('time_warp_prob', 0.2),
        lead_scale_range=tuple(aug_params.get('lead_scale_range', [0.8, 1.2])),
        lead_scale_prob=aug_params.get('lead_scale_prob', 0.2),
        lead_masking_prob=aug_params.get('lead_masking_prob', 0.1),
        lead_masking_max_leads=aug_params.get('lead_masking_max_leads', 6),
    )

    print(f"✓ Target sampling rate: {data_config.target_sampling_rate} Hz")
    print(f"✓ Target length: {data_config.target_length_seconds} seconds")

    # ── [3/7] Load data ───────────────────────────────────────────────────────
    print("\n[3/7] Loading data...")
    datamodule = UnifiedDataModule(
        config=data_config,
        augmentation_config=augmentation_config
    )
    datamodule.setup(stage='fit')

    subclass_names = datamodule.train_dataset.subclass_order
    class_weights = datamodule.get_class_weights()
    print(f"\n✓ Class weights computed:")
    for superclass, weight in class_weights['superclass_weights'].items():
        print(f"  {superclass}: {weight:.3f}")

    # ── [4/7] Build model ─────────────────────────────────────────────────────
    print("\n[4/7] Building model...")
    model_config = config['model']

    base_model = create_ecg_transformer_rope(
        model_size=model_config['size'],
        in_channels=model_config['in_channels'],
        num_superclasses=model_config['num_superclasses'],
        num_subclasses=len(subclass_names),
        dropout=model_config['dropout']
    )

    print(f"✓ Model: {model_config['architecture']} ({model_config['size']})")
    print(f"  Parameters: {base_model.get_num_params():,}")
    print(f"  Superclasses: {model_config['num_superclasses']}")
    print(f"  Subclasses: {len(subclass_names)} (incl. BRUG)")

    # ── [5/7] Loss function ───────────────────────────────────────────────────
    print("\n[5/7] Configuring loss function...")
    loss_config = config['loss']
    loss_type   = loss_config['type']

    if loss_type == 'weighted_bce':
        loss_kwargs = {
            'task_weights':       loss_config.get('task_weights', {}),
            'superclass_weights': torch.tensor([
                class_weights['superclass_weights'][sc]
                for sc in ['NORM', 'MI', 'STTC', 'CD', 'HYP']
            ]),
        }
    elif loss_type == 'focal':
        focal_cfg = loss_config.get('focal', {})
        loss_kwargs = {
            'task_weights':             loss_config.get('task_weights', {}),
            'alpha_superclass':         focal_cfg.get('alpha_superclass', 0.25),
            'alpha_subclass':           focal_cfg.get('alpha_subclass', 0.25),
            'gamma':                    focal_cfg.get('gamma', 2.0),
            'subclass_alpha_overrides': focal_cfg.get('subclass_alpha_overrides', None),
            'subclass_names':           subclass_names,
        }
    else:
        loss_kwargs = {'task_weights': loss_config.get('task_weights', {})}

    loss_fn = get_loss_function(loss_type, **loss_kwargs)
    print(f"✓ Loss function: {loss_type}")
    print(f"  Task weights: {loss_config.get('task_weights', {})}")
    if loss_type == 'focal':
        overrides = loss_config.get('focal', {}).get('subclass_alpha_overrides') or {}
        if overrides:
            print(f"  Per-class alpha overrides: {overrides}")

    # ── [6/7] Lightning module ────────────────────────────────────────────────
    print("\n[6/7] Creating Lightning module...")
    scheduler_params = config['training'].get('scheduler', {})
    if scheduler_params.get('type') == 'none':
        scheduler_params = None

    combined_auroc_weights = config['training'].get('combined_auroc_weights', None)

    lightning_model = MultiTaskClassifier(
        model=base_model,
        loss_fn=loss_fn,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_params=scheduler_params,
        subclass_names=subclass_names,
        combined_auroc_weights=combined_auroc_weights,
    )

    print(f"✓ Lightning module created")
    print(f"  Learning rate:          {config['training']['learning_rate']}")
    print(f"  Weight decay:           {config['training']['weight_decay']}")
    print(f"  Checkpoint monitor:     {config['training']['checkpoint']['monitor']}")
    cw = combined_auroc_weights or {'superclass': 0.5, 'subclass': 0.5}
    print(f"  Combined AUROC weights: sup={cw.get('superclass', 0.5)}, sub={cw.get('subclass', 0.5)}")

    # ── [7/7] Trainer ─────────────────────────────────────────────────────────
    print("\n[7/7] Setting up training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=config['logging']['project_name'],
        version=run_name
    )

    checkpoint_dir = Path('checkpoints') / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = create_callbacks(config, checkpoint_dir)

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        num_nodes=config['hardware'].get('num_nodes', 1),
        precision=config['hardware']['precision'],
        strategy=config['hardware'].get('strategy', 'auto'),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=False,
        benchmark=True,
        enable_progress_bar=True
    )

    print(f"✓ Trainer configured")
    print(f"  Max epochs:   {config['training']['max_epochs']}")
    print(f"  Accelerator:  {config['hardware']['accelerator']}")
    print(f"  Checkpoints:  {checkpoint_dir}")

    # ── MLflow setup ──────────────────────────────────────────────────────────
    # Resolve experiment name: CLI flag → config → default
    experiment_name = (
        args.experiment_name
        or config.get('logging', {}).get('experiment_name', 'brugada-ecg-classification')
    )
    tracking_uri = config.get('logging', {}).get('mlflow_tracking_uri', './mlruns')
    setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    print(f"\n✓ MLflow experiment: '{experiment_name}' @ {tracking_uri}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    with mlflow.start_run(run_name=run_name):
        # Log config and dataset info at the start of the run
        log_params_from_config(config)
        if datamodule.statistics is not None:
            s = datamodule.statistics
            mlflow.log_params({
                'data/total_samples':   s.total_samples,
                'data/brugada_samples': s.brugada_samples,
                'data/ptbxl_samples':   s.ptbxl_samples,
                'data/normal_samples':  s.normal_samples,
                'data/mi_samples':      s.mi_samples,
                'data/sttc_samples':    s.sttc_samples,
                'data/cd_samples':      s.cd_samples,
                'data/hyp_samples':     s.hyp_samples,
            })

        trainer.fit(model=lightning_model, datamodule=datamodule)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score

    print(f"\n✓ Best model: {best_model_path}")
    print(f"✓ Best {trainer.checkpoint_callback.monitor}: {best_score:.4f}")
    print(f"\nView results:")
    print(f"  TensorBoard: tensorboard --logdir lightning_logs")
    print(f"  MLflow:      mlflow ui --backend-store-uri {tracking_uri}")
    print(f"  Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
