"""PyTorch Lightning module for multi-task ECG disease detection."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision,
    MetricCollection,
)


class MultiTaskClassifier(pl.LightningModule):
    """
    Lightning module for multi-task ECG classification.

    Tasks:
    - Superclass classification (5 classes, multi-label)
    - Subclass classification (N classes, multi-label, incl. BRUG)

    Checkpoint monitor
    ------------------
    The module logs `val_combined_auroc` at the end of every validation epoch:

        val_combined_auroc = sup_weight * val_sup_auroc_macro
                           + sub_weight * val_sub_auroc_macro

    where the weights come from `combined_auroc_weights` (default 0.5 / 0.5).
    Set `checkpoint.monitor: val_combined_auroc` in train.yaml to select
    checkpoints that balance both tasks rather than optimising superclass alone.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        scheduler_params: Optional[Dict[str, Any]] = None,
        subclass_names: Optional[List[str]] = None,
        # Weights for the combined checkpoint monitor (must sum to 1.0)
        combined_auroc_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.model        = loss_fn and model   # model kept first for clarity
        self.model        = model
        self.loss_fn      = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.scheduler_params = scheduler_params or {}

        # ── Class names ───────────────────────────────────────────────────────
        self.superclass_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

        self.subclass_names = subclass_names or [
            'AMI', 'BRUG', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB',
            'ISCA', 'ISCI', 'ISC_', 'IVCD', 'LAFB/LPFB', 'LAO/LAE', 'LMI',
            'LVH', 'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW',
            '_AVB',
        ]

        num_subclasses = len(self.subclass_names)

        self.brug_subclass_idx = (
            self.subclass_names.index('BRUG')
            if 'BRUG' in self.subclass_names else None
        )

        # ── Combined AUROC monitor weights ────────────────────────────────────
        cw = combined_auroc_weights or {'superclass': 0.5, 'subclass': 0.5}
        self.combined_sup_weight = cw.get('superclass', 0.5)
        self.combined_sub_weight = cw.get('subclass', 0.5)

        # Save hyperparameters (exclude non-serialisable objects)
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

        # ── Metrics ───────────────────────────────────────────────────────────
        self.train_metrics_superclass = self._create_multilabel_metrics(
            num_labels=5, prefix='train_sup_'
        )
        self.val_metrics_superclass = self._create_multilabel_metrics(
            num_labels=5, prefix='val_sup_'
        )
        self.train_metrics_subclass = self._create_multilabel_metrics(
            num_labels=num_subclasses, prefix='train_sub_'
        )
        self.val_metrics_subclass = self._create_multilabel_metrics(
            num_labels=num_subclasses, prefix='val_sub_'
        )

    # ── Metric factory ────────────────────────────────────────────────────────

    def _create_multilabel_metrics(self, num_labels: int, prefix: str) -> MetricCollection:
        return MetricCollection({
            'auroc_macro':    AUROC(task='multilabel', num_labels=num_labels, average='macro'),
            'auroc_per_class': AUROC(task='multilabel', num_labels=num_labels, average=None),
            'auprc_macro':    AveragePrecision(task='multilabel', num_labels=num_labels, average='macro'),
            'acc_macro':      Accuracy(task='multilabel', num_labels=num_labels, average='macro'),
            'acc_micro':      Accuracy(task='multilabel', num_labels=num_labels, average='micro'),
            'f1_macro':       F1Score(task='multilabel', num_labels=num_labels, average='macro'),
            'f1_per_class':   F1Score(task='multilabel', num_labels=num_labels, average=None),
            'precision_macro': Precision(task='multilabel', num_labels=num_labels, average='macro'),
            'precision_per_class': Precision(task='multilabel', num_labels=num_labels, average=None),
            'recall_macro':   Recall(task='multilabel', num_labels=num_labels, average='macro'),
            'recall_per_class': Recall(task='multilabel', num_labels=num_labels, average=None),
        }, prefix=prefix)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        signals = batch['signal']
        labels  = batch['labels']

        predictions = self(signals)
        losses      = self.loss_fn(predictions, labels)

        for name, val in losses.items():
            self.log(f'train_{name}_loss', val, on_step=False, on_epoch=True,
                     prog_bar=(name == 'total'))

        sup_probs = torch.sigmoid(predictions['superclass'])
        sub_probs = torch.sigmoid(predictions['subclass'])
        self.train_metrics_superclass.update(sup_probs, labels['superclass'].int())
        self.train_metrics_subclass.update(sub_probs,   labels['subclass'].int())

        return losses['total']

    def on_train_epoch_end(self):
        sup_metrics = self.train_metrics_superclass.compute()
        for metric_name, metric_value in sup_metrics.items():
            if 'per_class' in metric_name:
                base = metric_name.replace('train_sup_', '').replace('_per_class', '')
                for i, cls in enumerate(self.superclass_names):
                    self.log(f'train_sup_{base}_{cls}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=('auroc_macro' in metric_name))
        self.train_metrics_superclass.reset()

        sub_metrics = self.train_metrics_subclass.compute()
        for metric_name, metric_value in sub_metrics.items():
            if 'per_class' in metric_name:
                base = metric_name.replace('train_sub_', '').replace('_per_class', '')
                for i, cls in enumerate(self.subclass_names):
                    self.log(f'train_sub_{base}_{cls}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=False)
        self.train_metrics_subclass.reset()

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        signals = batch['signal']
        labels  = batch['labels']

        predictions = self(signals)
        losses      = self.loss_fn(predictions, labels)

        for name, val in losses.items():
            self.log(f'val_{name}_loss', val, on_step=False, on_epoch=True,
                     prog_bar=(name == 'total'))

        sup_probs = torch.sigmoid(predictions['superclass'])
        sub_probs = torch.sigmoid(predictions['subclass'])
        self.val_metrics_superclass.update(sup_probs, labels['superclass'].int())
        self.val_metrics_subclass.update(sub_probs,   labels['subclass'].int())

        return losses['total']

    def on_validation_epoch_end(self):
        # ── Superclass ────────────────────────────────────────────────────────
        sup_metrics = self.val_metrics_superclass.compute()
        sup_auroc_macro = None
        for metric_name, metric_value in sup_metrics.items():
            if 'per_class' in metric_name:
                base = metric_name.replace('val_sup_', '').replace('_per_class', '')
                for i, cls in enumerate(self.superclass_names):
                    self.log(f'val_sup_{base}_{cls}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=True)
                if metric_name == 'val_sup_auroc_macro':
                    sup_auroc_macro = metric_value
        self.val_metrics_superclass.reset()

        # ── Subclass ──────────────────────────────────────────────────────────
        sub_metrics = self.val_metrics_subclass.compute()
        sub_auroc_macro = None
        for metric_name, metric_value in sub_metrics.items():
            if 'per_class' in metric_name:
                base = metric_name.replace('val_sub_', '').replace('_per_class', '')
                for i, cls in enumerate(self.subclass_names):
                    self.log(f'val_sub_{base}_{cls}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=False)
                if metric_name == 'val_sub_auroc_macro':
                    sub_auroc_macro = metric_value
        self.val_metrics_subclass.reset()

        # ── Combined AUROC monitor ─────────────────────────────────────────────
        # Logged every epoch so it's available as a checkpoint / early-stopping
        # monitor.  Uses whatever weights were set at init (default 0.5 / 0.5).
        if sup_auroc_macro is not None and sub_auroc_macro is not None:
            combined = (
                self.combined_sup_weight * sup_auroc_macro
                + self.combined_sub_weight * sub_auroc_macro
            )
            self.log('val_combined_auroc', combined, prog_bar=True,
                     sync_dist=True)

    # ── Test (delegates to validation) ────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.scheduler_params:
            return optimizer

        scheduler_type = self.scheduler_params.get('type', 'cosine')

        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_epochs = self.scheduler_params.get('warmup_epochs', 10)
            total_epochs  = self.scheduler_params.get('t_max', 200)

            warmup  = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine  = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            return [optimizer], [scheduler]

        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_params.get('factor', 0.5),
                patience=self.scheduler_params.get('patience', 5),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor':   'val_total_loss',
                    'interval':  'epoch',
                    'frequency': 1,
                },
            }

        return optimizer
