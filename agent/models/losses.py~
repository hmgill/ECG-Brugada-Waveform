"""Multi-task loss functions for ECG disease detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining superclass, subclass, and brugada tasks.
    
    Loss = w_sup * L_superclass + w_sub * L_subclass + w_brug * L_brugada
    """
    
    def __init__(
        self,
        superclass_weights: Optional[torch.Tensor] = None,
        subclass_weights: Optional[torch.Tensor] = None,
        brugada_pos_weight: float = 1.0,
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        # Task weights (how much each task contributes to total loss)
        default_task_weights = {
            'superclass': 1.0,
            'subclass': 0.5,
            'brugada': 1.0
        }
        self.task_weights = task_weights or default_task_weights
        
        # Loss functions
        # Multi-label classification for superclass (multi-hot)
        self.superclass_loss = nn.BCEWithLogitsLoss(
            pos_weight=superclass_weights
        )
        
        # Multi-label classification for subclass (multi-hot)
        self.subclass_loss = nn.BCEWithLogitsLoss(
            pos_weight=subclass_weights
        )
        
        # Binary classification for brugada
        self.brugada_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([brugada_pos_weight])
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dict with 'superclass', 'subclass', 'brugada' logits
            targets: Dict with corresponding ground truth labels
            
        Returns:
            Dict with 'total', 'superclass', 'subclass', 'brugada' losses
        """
        losses = {}
        
        # Superclass loss
        loss_sup = self.superclass_loss(
            predictions['superclass'],
            targets['superclass']
        )
        losses['superclass'] = loss_sup
        
        # Subclass loss
        loss_sub = self.subclass_loss(
            predictions['subclass'],
            targets['subclass']
        )
        losses['subclass'] = loss_sub
        
        # Brugada loss
        loss_brug = self.brugada_loss(
            predictions['brugada'].squeeze(-1),
            targets['brugada']
        )
        losses['brugada'] = loss_brug
        
        # Total weighted loss
        losses['total'] = (
            self.task_weights['superclass'] * loss_sup +
            self.task_weights['subclass'] * loss_sub +
            self.task_weights['brugada'] * loss_brug
        )
        
        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs
            targets: Ground truth (0 or 1)
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine
        focal_loss = alpha_t * focal_term * bce
        
        return focal_loss.mean()


class MultiTaskFocalLoss(nn.Module):
    """Multi-task loss using Focal Loss for all tasks."""
    
    def __init__(
        self,
        alpha_superclass: float = 0.25,
        alpha_subclass: float = 0.25,
        alpha_brugada: float = 0.25,
        gamma: float = 2.0,
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        default_task_weights = {
            'superclass': 1.0,
            'subclass': 0.5,
            'brugada': 1.0
        }
        self.task_weights = task_weights or default_task_weights
        
        self.superclass_loss = FocalLoss(alpha=alpha_superclass, gamma=gamma)
        self.subclass_loss = FocalLoss(alpha=alpha_subclass, gamma=gamma)
        self.brugada_loss = FocalLoss(alpha=alpha_brugada, gamma=gamma)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task focal loss."""
        losses = {}
        
        # Compute per-class focal loss and average
        # Superclass (5 classes, multi-label)
        loss_sup = torch.stack([
            self.superclass_loss(
                predictions['superclass'][:, i],
                targets['superclass'][:, i]
            )
            for i in range(predictions['superclass'].shape[1])
        ]).mean()
        losses['superclass'] = loss_sup
        
        # Subclass (23 classes, multi-label)
        loss_sub = torch.stack([
            self.subclass_loss(
                predictions['subclass'][:, i],
                targets['subclass'][:, i]
            )
            for i in range(predictions['subclass'].shape[1])
        ]).mean()
        losses['subclass'] = loss_sub
        
        # Brugada (binary)
        loss_brug = self.brugada_loss(
            predictions['brugada'].squeeze(-1),
            targets['brugada']
        )
        losses['brugada'] = loss_brug
        
        # Total
        losses['total'] = (
            self.task_weights['superclass'] * loss_sup +
            self.task_weights['subclass'] * loss_sub +
            self.task_weights['brugada'] * loss_brug
        )
        
        return losses


def get_loss_function(
    loss_type: str = 'bce',
    **kwargs
) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_type: 'bce' or 'focal'
        **kwargs: Parameters for the loss function
        
    Returns:
        Loss function module
    """
    if loss_type == 'bce':
        return MultiTaskLoss(**kwargs)
    elif loss_type == 'focal':
        return MultiTaskFocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'bce' or 'focal'.")
