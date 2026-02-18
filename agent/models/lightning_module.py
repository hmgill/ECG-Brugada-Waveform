"""PyTorch Lightning module for multi-task ECG disease detection."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, MetricCollection


class MultiTaskClassifier(pl.LightningModule):
    """
    Lightning module for multi-task ECG classification.
    
    Tasks:
    - Superclass classification (5 classes, multi-label)
    - Subclass classification (24 classes, multi-label, incl. BRUG)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        scheduler_params: Optional[Dict[str, Any]] = None,
        subclass_names: Optional[List[str]] = None,
    ):
        super().__init__()
        
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params or {}
        
        # Class names for logging per-class metrics
        self.superclass_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

        # Accept subclass names at init so the caller can pass the runtime-built
        # list (23 PTB-XL subclasses + BRUG = 24).  Fall back to the 23 PTB-XL
        # names for backwards compatibility.
        self.subclass_names = subclass_names or [
            'NDT', 'NST_', 'DIG', 'LNGQT', 'NORM', 'IMI', 'ASMI', 'LVH',
            'LAFB/LPFB', 'ISC_', 'IRBBB', '1AVB', 'IVCD', 'ISCAL', 'CRBBB',
            'CLBBB', 'ILMI', 'LAO/LAE', 'AMI', 'ALMI', 'IPLMI', 'IPMI', 'SEHYP',
            'BRUG',
        ]  # 24 classes

        num_subclasses = len(self.subclass_names)

        # Index of the BRUG subclass for targeted metric logging
        self.brug_subclass_idx = (
            self.subclass_names.index('BRUG')
            if 'BRUG' in self.subclass_names
            else None
        )
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        # ── Metrics ───────────────────────────────────────────────────────────

        # Superclass (5 classes, multi-label)
        self.train_metrics_superclass = self._create_multilabel_metrics(num_labels=5, prefix='train_sup_')
        self.val_metrics_superclass   = self._create_multilabel_metrics(num_labels=5, prefix='val_sup_')
        
        # Subclass (24 classes, multi-label)
        self.train_metrics_subclass = self._create_multilabel_metrics(num_labels=num_subclasses, prefix='train_sub_')
        self.val_metrics_subclass   = self._create_multilabel_metrics(num_labels=num_subclasses, prefix='val_sub_')

    
    # ── Metric factory helpers ────────────────────────────────────────────────

    def _create_multilabel_metrics(self, num_labels: int, prefix: str) -> MetricCollection:
        """Create metric collection for multi-label classification."""
        threshold = 0.3
        return MetricCollection({
            'auroc_macro':     AUROC(task='multilabel', num_labels=num_labels, average='macro'),
            'auprc_macro':     AveragePrecision(task='multilabel', num_labels=num_labels, average='macro'),
            'f1_macro':        F1Score(task='multilabel', num_labels=num_labels, average='macro', threshold=threshold),
            'precision_macro': Precision(task='multilabel', num_labels=num_labels, average='macro', threshold=threshold),
            'recall_macro':    Recall(task='multilabel', num_labels=num_labels, average='macro', threshold=threshold),
            'acc_micro':  Accuracy(task='multilabel', num_labels=num_labels, average='micro', threshold=threshold),
            'acc_macro':  Accuracy(task='multilabel', num_labels=num_labels, average='macro', threshold=threshold),            
            # Per-class (for named logging)
            'auroc_per_class':     AUROC(task='multilabel', num_labels=num_labels, average=None),
            'f1_per_class':        F1Score(task='multilabel', num_labels=num_labels, average=None, threshold=threshold),
            'precision_per_class': Precision(task='multilabel', num_labels=num_labels, average=None, threshold=threshold),
            'recall_per_class':    Recall(task='multilabel', num_labels=num_labels, average=None, threshold=threshold),
        }, prefix=prefix)
    
    def _create_binary_metrics(self, prefix: str) -> MetricCollection:
        """Create metric collection for binary classification."""
        return MetricCollection({
            'auroc':     AUROC(task='binary'),
            'auprc':     AveragePrecision(task='binary'),
            'acc':       Accuracy(task='binary'),
            'precision': Precision(task='binary'),
            'recall':    Recall(task='binary'),
            'f1':        F1Score(task='binary'),
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
        
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}_loss', loss_value, on_step=False, on_epoch=True,
                     prog_bar=(loss_name == 'total'))
        
        # Superclass metrics
        sup_probs = torch.sigmoid(predictions['superclass'])
        self.train_metrics_superclass.update(sup_probs, labels['superclass'].int())
        
        # Subclass metrics
        sub_probs = torch.sigmoid(predictions['subclass'])
        self.train_metrics_subclass.update(sub_probs, labels['subclass'].int())

        
        return losses['total']
    
    def on_train_epoch_end(self):
        # Superclass
        sup_metrics = self.train_metrics_superclass.compute()
        for metric_name, metric_value in sup_metrics.items():
            if 'per_class' in metric_name:
                metric_base = metric_name.replace('train_sup_', '').replace('_per_class', '')
                for i, class_name in enumerate(self.superclass_names):
                    self.log(f'train_sup_{metric_base}_{class_name}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=('auroc' in metric_name))
        self.train_metrics_superclass.reset()
        
        # Subclass
        sub_metrics = self.train_metrics_subclass.compute()
        for metric_name, metric_value in sub_metrics.items():
            if 'per_class' in metric_name:
                metric_base = metric_name.replace('train_sub_', '').replace('_per_class', '')
                for i, class_name in enumerate(self.subclass_names):
                    self.log(f'train_sub_{metric_base}_{class_name}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=False)
        self.train_metrics_subclass.reset()

    
    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        signals = batch['signal']
        labels  = batch['labels']
        
        predictions = self(signals)
        losses      = self.loss_fn(predictions, labels)
        
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}_loss', loss_value, on_step=False, on_epoch=True,
                     prog_bar=(loss_name == 'total'))
        
        # Superclass metrics
        sup_probs = torch.sigmoid(predictions['superclass'])
        self.val_metrics_superclass.update(sup_probs, labels['superclass'].int())
        
        # Subclass metrics
        sub_probs = torch.sigmoid(predictions['subclass'])
        self.val_metrics_subclass.update(sub_probs, labels['subclass'].int())

        
        return losses['total']
    
    def on_validation_epoch_end(self):
        # Superclass
        sup_metrics = self.val_metrics_superclass.compute()
        for metric_name, metric_value in sup_metrics.items():
            if 'per_class' in metric_name:
                metric_base = metric_name.replace('val_sup_', '').replace('_per_class', '')
                for i, class_name in enumerate(self.superclass_names):
                    self.log(f'val_sup_{metric_base}_{class_name}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=True)
        self.val_metrics_superclass.reset()
        
        # Subclass
        sub_metrics = self.val_metrics_subclass.compute()
        for metric_name, metric_value in sub_metrics.items():
            if 'per_class' in metric_name:
                metric_base = metric_name.replace('val_sub_', '').replace('_per_class', '')
                for i, class_name in enumerate(self.subclass_names):
                    self.log(f'val_sub_{metric_base}_{class_name}', metric_value[i], prog_bar=False)
            else:
                self.log(metric_name, metric_value, prog_bar=False)
        self.val_metrics_subclass.reset()

    
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
            weight_decay=self.weight_decay
        )
        
        if not self.scheduler_params:
            return optimizer
        
        scheduler_type = self.scheduler_params.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_epochs = self.scheduler_params.get('warmup_epochs', 10)
            total_epochs  = self.scheduler_params.get('t_max', 150)
            
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
            
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            return [optimizer], [scheduler]
        
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_params.get('factor', 0.5),
                patience=self.scheduler_params.get('patience', 5)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_total_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        
        return optimizer
