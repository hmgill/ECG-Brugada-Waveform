"""Loss functions for Brugada ECG classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    
    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    where:
        p_t = p if y=1, else (1-p)
        alpha_t = alpha if y=1, else (1-alpha)
        
    Args:
        alpha: Weight for positive class. For imbalanced datasets, 
               set to (n_positive / n_total). Default: 0.25
        gamma: Focusing parameter. Higher values focus more on hard examples.
               Default: 2.0 (from paper)
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    
    Example:
        >>> loss_fn = FocalLoss(alpha=0.21, gamma=2.0)
        >>> logits = model(inputs)
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(
        self, 
        alpha: float = 0.65, 
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Raw model outputs (batch_size,) - logits before sigmoid
            targets: Ground truth binary labels (batch_size,) - 0 or 1
            
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Ensure targets are float
        targets = targets.float()
        
        # Compute p_t: probability of the true class
        # If target=1: p_t = p
        # If target=0: p_t = 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute focal term: (1 - p_t)^gamma
        # This downweights easy examples (where p_t is close to 1)
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute standard BCE: -log(p_t)
        # We use binary_cross_entropy_with_logits for numerical stability
        bce = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            reduction='none'
        )
        
        # Apply alpha weighting
        # If target=1: alpha_t = alpha
        # If target=0: alpha_t = 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine all terms: FL = -alpha_t * (1-p_t)^gamma * log(p_t)
        focal_loss = alpha_t * focal_term * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Simple wrapper around F.binary_cross_entropy_with_logits
    for consistency with FocalLoss API.
    
    Args:
        pos_weight: Weight for positive class. Recommended: n_negative / n_positive
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    
    Example:
        >>> loss_fn = WeightedBCELoss(pos_weight=3.78)
        >>> logits = model(inputs)
        >>> loss = loss_fn(logits, labels)
    """
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Raw model outputs (batch_size,)
            targets: Ground truth binary labels (batch_size,)
            
        Returns:
            Weighted BCE loss value
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight.to(logits.device),
            reduction=self.reduction
        )


def get_loss_function(loss_type: str, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: One of 'bce', 'weighted_bce', 'focal'
        **kwargs: Arguments for the loss function
        
    Returns:
        Loss function instance
        
    Example:
        >>> # Standard BCE
        >>> loss_fn = get_loss_function('bce')
        
        >>> # Weighted BCE
        >>> loss_fn = get_loss_function('weighted_bce', pos_weight=3.78)
        
        >>> # Focal Loss
        >>> loss_fn = get_loss_function('focal', alpha=0.25, gamma=2.0)
    """
    if loss_type == 'bce':
        # Standard unweighted BCE
        return lambda logits, targets: F.binary_cross_entropy_with_logits(
            logits, targets.float()
        )
    elif loss_type == 'weighted_bce':
        pos_weight = kwargs.get('pos_weight', 1.0)
        return WeightedBCELoss(pos_weight=pos_weight)
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. "
            f"Choose from: 'bce', 'weighted_bce', 'focal'"
        )
