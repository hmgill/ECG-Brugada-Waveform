"""
HuBERT-ECG Multi-Task Model
============================
Wraps the pretrained HuBERT-ECG encoder with multi-task classification heads.

Architecture
------------
1.  HuBERT-ECG encoder  (frozen during warm-up, then unfrozen)
        - Input: (B, 12, 500)  — 5-second window @ 100 Hz
        - Output: (B, T, d_model)  — sequence of contextual representations
2.  Temporal pooling  → (B, d_model)
        - CLS token if the encoder produces one, otherwise mean-pool
3.  Dual-window fusion  (optional, for recordings longer than 5 s)
        - Split signal into N non-overlapping 5-second windows
        - Run each window through the shared encoder
        - Mean-pool window embeddings  → (B, d_model)
4.  Per-class superclass heads  → (B, 5) logits
        - One independent MLP per superclass (NORM, MI, STTC, CD, HYP)
        - Gradients from rare classes (e.g. HYP) no longer compete with
          gradients from majority classes (e.g. NORM) through shared weights
5.  Subclass head    → (B, N)   logits  (N = len(subclass_names))

Preprocessing contract
-----------------------
Signals must arrive already resampled to 100 Hz and z-score normalised
per-lead, exactly as your existing dataset.py pipeline produces them.
The encoder was pretrained under that same normalisation scheme.

Usage in train.py
-----------------
Replace:
    base_model = create_ecg_transformer_rope(...)

With:
    from models.hubert_ecg_multitask import create_hubert_multitask
    base_model = create_hubert_multitask(
        num_superclasses=5,
        num_subclasses=len(subclass_names),
        hubert_size="base",          # "small" | "base" | "large"
        freeze_encoder_epochs=10,    # passed to MultiTaskClassifier
        window_seconds=5.0,
        dropout=0.1,
    )

The model exposes:
    model.freeze_encoder()
    model.unfreeze_encoder()
    model.get_num_params()
    model.get_num_trainable_params()
and the standard forward() returning {'superclass': ..., 'subclass': ...}.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HUBERT_HIDDEN_SIZES = {
    "small": 256,
    "base":  768,
    "large": 1024,
}

HUBERT_HF_IDS = {
    "small": "Edoardo-BS/hubert-ecg-small",
    "base":  "Edoardo-BS/hubert-ecg-base",
    "large": "Edoardo-BS/hubert-ecg-large",
}

# HuBERT-ECG was pretrained on 5-second windows at 100 Hz
HUBERT_WINDOW_SAMPLES = 500   # 5 s × 100 Hz
HUBERT_SAMPLING_RATE  = 100   # Hz


# ─────────────────────────────────────────────────────────────────────────────
# Encoder loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_hubert_encoder(size: str) -> nn.Module:
    """
    Load HuBERT-ECG pretrained encoder from HuggingFace.

    Returns the encoder module only (no classification head).
    Raises ImportError with a clear install message if transformers is absent.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "The `transformers` package is required for HuBERT-ECG.\n"
            "Install it with:  pip install transformers"
        ) from exc

    hf_id = HUBERT_HF_IDS[size]
    print(f"  Loading HuBERT-ECG {size} from {hf_id} ...")

    # trust_remote_code is needed because HuBERT-ECG ships custom model code
    full_model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)

    # HuBERT-ECG models expose the transformer body as `.hubert` or `.encoder`;
    # the exact attribute depends on the version.  We probe in order of
    # likelihood and fall back to returning the whole model if neither exists,
    # which is safe because we never call its classification head.
    encoder = (
        getattr(full_model, "hubert",  None)
        or getattr(full_model, "encoder", None)
        or full_model
    )

    print(f"  ✓ Encoder loaded ({sum(p.numel() for p in encoder.parameters()):,} params)")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Classification heads
# ─────────────────────────────────────────────────────────────────────────────

def _make_single_superclass_head(d_model: int, dropout: float) -> nn.Sequential:
    """
    One binary MLP head for a single superclass.

    Using separate heads per superclass means each class has its own gradient
    path from the shared embedding.  Rare classes like HYP no longer compete
    with NORM for weight updates in the final projection layer, which is the
    main practical benefit over a single shared Linear(d_model → 5).

    Hidden dim 64 keeps parameter count modest while allowing non-linear
    feature selection from the 768-dim HuBERT embedding.
    """
    return nn.Sequential(
        nn.Linear(d_model, 64),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(64, 1),       # single logit per class
    )


def _make_subclass_head(d_model: int, num_subclasses: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, num_subclasses),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class HuBERTECGMultiTask(nn.Module):
    """
    HuBERT-ECG encoder + multi-task classification heads.

    Dual-window strategy
    --------------------
    A 10-second PTB-XL recording (1000 samples @ 100 Hz) is split into two
    non-overlapping 5-second windows.  Each window is encoded independently,
    and the resulting embeddings are averaged before the classification heads.

    For recordings shorter than 5 seconds the signal is zero-padded to 500
    samples.  For recordings longer than 10 seconds the signal is split into
    floor(T / 500) windows; any remainder is dropped (centre-crop alignment
    is applied in _split_windows to minimise boundary artefacts).

    The dual-window fusion is applied in both training and inference, so the
    heads always see the same embedding distribution.
    """

    def __init__(
        self,
        encoder:          nn.Module,
        d_model:          int,
        num_superclasses: int,
        num_subclasses:   int,
        dropout:          float = 0.1,
        window_samples:   int   = HUBERT_WINDOW_SAMPLES,
        pool_mode:        str   = "mean",   # "mean" | "cls"
    ):
        super().__init__()

        self.encoder        = encoder
        self.d_model        = d_model
        self.num_superclasses = num_superclasses
        self.window_samples = window_samples
        self.pool_mode      = pool_mode

        # Layer norm applied after pooling for training stability
        self.post_pool_norm = nn.LayerNorm(d_model)

        # Per-superclass binary heads — one independent MLP per class.
        # Output is concatenated to (B, num_superclasses) in forward().
        self.superclass_heads = nn.ModuleList([
            _make_single_superclass_head(d_model, dropout)
            for _ in range(num_superclasses)
        ])

        self.head_subclass = _make_subclass_head(d_model, num_subclasses, dropout)

        self._init_heads()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_heads(self) -> None:
        """Initialise head weights; encoder weights come from pretraining."""
        modules_to_init = [*self.superclass_heads, self.head_subclass, self.post_pool_norm]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ── Encoder freeze / unfreeze ─────────────────────────────────────────────

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (linear-probe phase)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("  Encoder frozen  — training heads only")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for end-to-end fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("  Encoder unfrozen — end-to-end fine-tuning")

    # ── Window splitting ──────────────────────────────────────────────────────

    def _split_windows(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split (B, 12, T) into (B * n_windows, 12, window_samples).

        If T < window_samples the signal is zero-padded.
        If T is not a multiple of window_samples, the signal is centre-cropped
        to the largest multiple before splitting, avoiding ragged final windows.

        Returns:
            windows:    (B * n_windows, 12, window_samples)
            n_windows:  int  (same for every sample in the batch)
        """
        B, C, T = x.shape
        W = self.window_samples

        if T < W:
            # Pad short recordings to exactly one window
            pad = torch.zeros(B, C, W - T, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)
            T = W

        n_windows = T // W
        # Centre-crop to discard any ragged remainder
        crop_len = n_windows * W
        start    = (T - crop_len) // 2
        x        = x[:, :, start : start + crop_len]

        # (B, C, n_windows * W) → (B, n_windows, C, W) → (B*n_windows, C, W)
        x = x.reshape(B, C, n_windows, W)
        x = x.permute(0, 2, 1, 3)                     # (B, n_windows, C, W)
        x = x.reshape(B * n_windows, C, W)
        return x, n_windows

    # ── Temporal pooling ──────────────────────────────────────────────────────

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool (B, T_seq, d_model) → (B, d_model).

        pool_mode="cls"  : use the first token (CLS position)
        pool_mode="mean" : mean across all time steps (default, more robust)
        """
        if self.pool_mode == "cls":
            return hidden_states[:, 0, :]
        return hidden_states.mean(dim=1)

    # ── Encoder forward ───────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run encoder on (B, 12, W) and return pooled (B, d_model).

        HuBERT-ECG expects (batch, time) — no lead dimension.
        We treat each lead as an independent batch element:
          (B, 12, W) → (B*12, W) → encode → pool time → (B*12, d_model)
                     → reshape (B, 12, d_model) → mean leads → (B, d_model)
        """
        B, n_leads, W = x.shape

        # Flatten leads into batch dimension
        x = x.reshape(B * n_leads, W)              # (B*12, W)

        out = self.encoder(x)

        # Unpack if it's a dataclass / named tuple output
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state          # (B*12, T_seq, d_model)
        elif isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out                            # plain tensor

        # Pool time dimension
        if hidden.dim() == 3:
            hidden = self._pool(hidden)             # (B*12, d_model)

        # Mean-pool across leads
        hidden = hidden.reshape(B, n_leads, -1)     # (B, 12, d_model)
        return hidden.mean(dim=1)                   # (B, d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x:                 (B, 12, T)  — signal at 100 Hz, z-score normalised
            return_embeddings: if True return the pooled embedding instead of logits

        Returns:
            {'superclass': (B, 5), 'subclass': (B, N)}  logits
            or (B, d_model) embedding tensor if return_embeddings=True
        """
        B = x.shape[0]

        # 1. Split into fixed-size windows
        windows, n_windows = self._split_windows(x)   # (B*n_windows, 12, W)

        # 2. Encode each window
        embeddings = self._encode(windows)             # (B*n_windows, d_model)

        # 3. Average over windows → one embedding per recording
        embeddings = embeddings.reshape(B, n_windows, self.d_model)
        embedding  = embeddings.mean(dim=1)            # (B, d_model)

        # 4. Post-pool normalisation
        embedding = self.post_pool_norm(embedding)

        if return_embeddings:
            return embedding

        # 5. Multi-task heads
        # Per-class superclass heads: each returns (B, 1), cat → (B, 5)
        sup_logits = torch.cat(
            [head(embedding) for head in self.superclass_heads], dim=1
        )
        return {
            "superclass": sup_logits,                    # (B, num_superclasses)
            "subclass":   self.head_subclass(embedding), # (B, N)
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function  (mirrors create_ecg_transformer_rope API)
# ─────────────────────────────────────────────────────────────────────────────

def create_hubert_multitask(
    num_superclasses:     int   = 5,
    num_subclasses:       int   = 23,
    hubert_size:          str   = "base",
    dropout:              float = 0.1,
    window_seconds:       float = 5.0,
    pool_mode:            str   = "mean",
    freeze_on_init:       bool  = True,
) -> HuBERTECGMultiTask:
    """
    Factory function — mirrors create_ecg_transformer_rope() signature style.

    Args:
        num_superclasses:  number of superclass output neurons (default 5)
        num_subclasses:    number of subclass output neurons (derived from data)
        hubert_size:       "small" | "base" | "large"
        dropout:           dropout rate applied inside classification heads
        window_seconds:    encoder input window length (must be 5.0 for HuBERT-ECG)
        pool_mode:         "mean" (default) or "cls"
        freeze_on_init:    freeze encoder immediately after loading

    Returns:
        HuBERTECGMultiTask ready for training
    """
    if hubert_size not in HUBERT_HF_IDS:
        raise ValueError(
            f"hubert_size must be one of {list(HUBERT_HF_IDS.keys())}, "
            f"got '{hubert_size}'"
        )

    if abs(window_seconds - 5.0) > 1e-6:
        raise ValueError(
            f"HuBERT-ECG was pretrained on 5-second windows; "
            f"window_seconds must be 5.0, got {window_seconds}"
        )

    d_model      = HUBERT_HIDDEN_SIZES[hubert_size]
    window_samps = int(window_seconds * HUBERT_SAMPLING_RATE)

    encoder = _load_hubert_encoder(hubert_size)

    model = HuBERTECGMultiTask(
        encoder          = encoder,
        d_model          = d_model,
        num_superclasses = num_superclasses,
        num_subclasses   = num_subclasses,
        dropout          = dropout,
        window_samples   = window_samps,
        pool_mode        = pool_mode,
    )

    if freeze_on_init:
        model.freeze_encoder()

    total     = model.get_num_params()
    trainable = model.get_num_trainable_params()
    print(
        f"  HuBERTECGMultiTask ({hubert_size})  "
        f"total={total:,}  trainable={trainable:,}"
    )
    return model
