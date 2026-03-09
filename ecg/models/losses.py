"""Multi-task loss functions for ECG disease detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining superclass and subclass tasks using weighted BCE.

    Loss = w_sup * L_superclass + w_sub * L_subclass
    """

    def __init__(
        self,
        superclass_weights: Optional[torch.Tensor] = None,
        subclass_weights:   Optional[torch.Tensor] = None,
        task_weights:       Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.task_weights = task_weights or {'superclass': 1.0, 'subclass': 0.5}

        self.superclass_loss = nn.BCEWithLogitsLoss(pos_weight=superclass_weights)
        self.subclass_loss   = nn.BCEWithLogitsLoss(pos_weight=subclass_weights)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        loss_sup = self.superclass_loss(predictions['superclass'], targets['superclass'])
        loss_sub = self.subclass_loss(predictions['subclass'],     targets['subclass'])

        losses['superclass'] = loss_sup
        losses['subclass']   = loss_sub
        losses['total'] = (
            self.task_weights['superclass'] * loss_sup
            + self.task_weights['subclass']   * loss_sub
        )
        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss with optional per-class alpha weighting.

    Supports two calling modes:

    1. Scalar alpha (original behaviour, applied uniformly to all classes):
        FocalLoss(alpha=0.25, gamma=2.0)

    2. Per-class alpha tensor (one value per output class):
        FocalLoss(alpha=torch.tensor([0.25, 0.25, 0.5, ...]), gamma=2.0)

       When a per-class tensor is supplied, the loss for class i uses
       alpha[i] rather than a shared value.  This lets you upweight the
       minority or morphologically hard classes without changing gamma.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma

        if isinstance(alpha, torch.Tensor):
            # Register as buffer so it moves with .to(device) calls
            self.register_buffer('alpha', alpha.float())
            self.per_class_alpha = True
        else:
            self.alpha = float(alpha)
            self.per_class_alpha = False

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) raw model outputs for a single class, or
                     (B, C) for all classes simultaneously
            targets: same shape as logits, ground truth in {0, 1}

        Returns:
            Scalar mean focal loss.
        """
        targets = targets.float()
        probs   = torch.sigmoid(logits)

        # p_t: probability assigned to the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulation
        focal_term = (1 - p_t) ** self.gamma

        # Standard BCE (element-wise)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        if self.per_class_alpha:
            # alpha shape: (C,) — broadcast over batch dimension
            # logits shape: (B, C) when called on a full tensor, or
            # (B,) when called per-class in a loop (alpha is a scalar then)
            if logits.dim() == 1:
                # Called per-class from a loop — alpha is indexed externally
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Called on the full (B, C) tensor at once
                alpha = self.alpha.view(1, -1)          # (1, C)
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        return (alpha_t * focal_term * bce).mean()


class MultiTaskFocalLoss(nn.Module):
    """
    Multi-task focal loss with per-class alpha support for subclasses.

    Superclass alpha is a single scalar (5 well-balanced classes).

    Subclass alpha can be either:
      - A single scalar applied uniformly to all subclasses (original), or
      - A dict mapping subclass name → alpha, e.g.:
            {'IVCD': 0.5, 'LAO/LAE': 0.5, 'NST_': 0.5}
        Unspecified classes receive the default alpha_subclass value.

    The subclass_names list must be supplied when using per-class alpha so
    that names can be resolved to tensor indices.  It is passed automatically
    by train.py from the runtime subclass order.
    """

    def __init__(
        self,
        alpha_superclass:    float = 0.25,
        alpha_subclass:      float = 0.25,
        gamma:               float = 2.0,
        task_weights:        Optional[Dict[str, float]] = None,
        # Per-class overrides — only used when subclass_names is also provided
        subclass_alpha_overrides: Optional[Dict[str, float]] = None,
        subclass_names:      Optional[List[str]] = None,
    ):
        super().__init__()

        self.task_weights = task_weights or {'superclass': 1.0, 'subclass': 0.5}
        self.gamma = gamma

        # ── Superclass loss (single scalar alpha) ─────────────────────────────
        self.superclass_loss = FocalLoss(alpha=alpha_superclass, gamma=gamma)

        # ── Subclass loss (scalar or per-class tensor alpha) ──────────────────
        if subclass_alpha_overrides and subclass_names:
            # Build a per-class alpha tensor, starting from the default
            alphas = torch.full((len(subclass_names),), alpha_subclass)
            for name, val in subclass_alpha_overrides.items():
                if name in subclass_names:
                    alphas[subclass_names.index(name)] = val
                else:
                    import warnings
                    warnings.warn(
                        f"subclass_alpha_overrides: '{name}' not found in "
                        f"subclass_names — skipping.",
                        stacklevel=2,
                    )
            self.subclass_loss = FocalLoss(alpha=alphas, gamma=gamma)
        else:
            self.subclass_loss = FocalLoss(alpha=alpha_subclass, gamma=gamma)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: {'superclass': (B, 5), 'subclass': (B, N)} logits
            targets:     same structure, ground truth in {0, 1}

        Returns:
            {'total', 'superclass', 'subclass'} scalar losses
        """
        losses = {}

        # Superclass: loop per-class so each uses its own alpha/focal terms
        loss_sup = torch.stack([
            self.superclass_loss(
                predictions['superclass'][:, i],
                targets['superclass'][:, i],
            )
            for i in range(predictions['superclass'].shape[1])
        ]).mean()

        # Subclass: pass the full (B, N) tensor so per-class alpha broadcasts
        loss_sub = self.subclass_loss(
            predictions['subclass'],
            targets['subclass'],
        )

        losses['superclass'] = loss_sup
        losses['subclass']   = loss_sub
        losses['total'] = (
            self.task_weights['superclass'] * loss_sup
            + self.task_weights['subclass']   * loss_sub
        )
        return losses


def get_loss_function(loss_type: str = 'focal', **kwargs) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_type: 'bce' or 'focal'
        **kwargs:  Passed directly to the loss constructor.
                   For 'focal', relevant keys include:
                     alpha_superclass, alpha_subclass, gamma, task_weights,
                     subclass_alpha_overrides, subclass_names

    Returns:
        Configured loss module.
    """
    if loss_type == 'bce':
        return MultiTaskLoss(**kwargs)
    elif loss_type == 'focal':
        return MultiTaskFocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Choose 'bce' or 'focal'.")
