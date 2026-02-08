"""PyTorch Lightning module for Brugada ECG classification."""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision

from .inception1d import Inception1D


class BrugadaClassifier(pl.LightningModule):
    """
    Lightning module for Brugada syndrome classification.
    
    Args:
        model: The base model (e.g., Inception1D)
        pos_weight: Weight for positive class in BCE loss
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        scheduler_params: Optional parameters for learning rate scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_type: str = 'weighted_bce',
        loss_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params or {}
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        
        # Create loss function based on type
        from training import get_loss_function
        self.loss_fn = get_loss_function(loss_type, **self.loss_params)
        
        # For backward compatibility with pos_weight logging
        if loss_type == 'weighted_bce':
            self.pos_weight = self.loss_params.get('pos_weight', 1.0)
        elif loss_type == 'focal':
            self.alpha = self.loss_params.get('alpha', 0.25)
            self.gamma = self.loss_params.get('gamma', 2.0)
        
        # Metrics (for binary classification)
        metrics_kwargs = {'task': 'binary'}
        self.train_acc = Accuracy(**metrics_kwargs)
        self.val_acc = Accuracy(**metrics_kwargs)
        self.val_precision = Precision(**metrics_kwargs)
        self.val_recall = Recall(**metrics_kwargs)  # Sensitivity
        self.val_f1 = F1Score(**metrics_kwargs)
        self.val_auroc = AUROC(**metrics_kwargs)
        self.val_auprc = AveragePrecision(**metrics_kwargs)
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss using the configured loss function."""
        return self.loss_fn(logits.squeeze(), labels)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Extract from batch (list of ECGSample objects)
        signals = torch.stack([s.signal for s in batch]).to(self.device)
        labels = torch.stack([s.label for s in batch]).to(self.device)
        
        # Forward pass
        logits = self(signals)
        loss = self._compute_loss(logits, labels)
        
        # Compute accuracy
        preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Extract from batch
        signals = torch.stack([s.signal for s in batch]).to(self.device)
        labels = torch.stack([s.label for s in batch]).to(self.device)
        
        # Forward pass
        logits = self(signals)
        loss = self._compute_loss(logits, labels)
        
        # Get predictions and probabilities
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs > 0.5).long()
        
        # Update metrics
        self.val_acc.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_auroc.update(probs, labels)
        self.val_auprc.update(probs, labels)
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log metrics at end of validation epoch."""
        # Compute metrics
        acc = self.val_acc.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()  # Sensitivity
        f1 = self.val_f1.compute()
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        
        # Log metrics
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall, prog_bar=True)  # Show sensitivity in progress bar
        self.log('val_sensitivity', recall)  # Also log as sensitivity
        self.log('val_f1', f1)
        self.log('val_auroc', auroc, prog_bar=True)
        self.log('val_auprc', auprc)
        
        # Compute specificity (TN / (TN + FP))
        # For binary classification: specificity = precision of negative class
        # We can approximate or compute it from confusion matrix if needed
        
        # Reset metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        if not self.scheduler_params:
            return optimizer
        
        # Create scheduler
        scheduler_type = self.scheduler_params.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
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
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_params.get('t_max', 50)
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
