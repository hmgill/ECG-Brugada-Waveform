"""
Unified PyTorch Dataset with variable-length ECG support.

Key changes from fixed-length version:
  - _standardize_length() now only pads signals shorter than min_length_seconds;
    signals longer than that are kept at their native length.
  - A max_length_seconds cap prevents GPU OOM from unusually long recordings.
  - collate_variable_length() pads each batch to the longest signal in that
    batch, replacing the fixed-size default collate.
  - All other behaviour (resampling, filtering, normalisation, augmentation,
    label generation) is unchanged.
"""

from typing import List, Optional, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from scipy import signal as scipy_signal

from .models import (
    ECGMetadata, ECGSample, AugmentationConfig,
    DatasetSource, DiagnosticSuperclass,
)


# ─────────────────────────────────────────────────────────────────────────────
# Collate function (replaces default — must be passed to every DataLoader)
# ─────────────────────────────────────────────────────────────────────────────

def collate_variable_length(batch: List[ECGSample]) -> Dict:
    """
    Collate ECGSamples with different signal lengths into a padded batch.

    Signals are right-padded with zeros to the length of the longest signal
    in the batch.  A boolean padding mask is returned so the model can ignore
    padded positions if needed (e.g. for loss masking or attention masking).

    Args:
        batch: List of ECGSample objects from __getitem__

    Returns:
        dict with keys:
            signal          (B, 12, T_max)   — zero-padded
            padding_mask    (B, T_max)        — True where real signal, False where pad
            labels
                superclass  (B, 5)
                subclass    (B, N_sub)
            lengths         (B,)              — actual signal length per sample
            metadata        List[ECGMetadata]
    """
    # Find longest signal in this batch
    lengths = torch.tensor([s.signal.shape[1] for s in batch], dtype=torch.long)
    t_max   = int(lengths.max().item())
    n_leads = batch[0].signal.shape[0]

    signals = torch.zeros(len(batch), n_leads, t_max)
    mask    = torch.zeros(len(batch), t_max, dtype=torch.bool)

    for i, sample in enumerate(batch):
        t = sample.signal.shape[1]
        signals[i, :, :t] = sample.signal
        mask[i, :t]        = True          # True = real signal

    return {
        'signal':       signals,
        'padding_mask': mask,
        'lengths':      lengths,
        'labels': {
            'superclass': torch.stack([s.label_superclass for s in batch]),
            'subclass':   torch.stack([s.label_subclass   for s in batch]),
        },
        'metadata': [s.original_metadata for s in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class ECGAugmentation:
    """ECG-specific data augmentation. Operates on (n_samples, n_leads) arrays."""

    def __init__(self, config: Optional[AugmentationConfig]):
        self.config = config

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if self.config is None:
            return signal

        cfg = self.config

        if np.random.random() < cfg.amplitude_scale_prob:
            signal = signal * np.random.uniform(*cfg.amplitude_scale_range)

        if np.random.random() < cfg.noise_prob:
            noise  = np.random.normal(0, cfg.noise_std, signal.shape)
            signal = signal + noise * signal.std()

        if np.random.random() < cfg.baseline_wander_prob:
            signal = self._add_baseline_wander(signal)

        if np.random.random() < cfg.time_warp_prob:
            signal = self._time_warp(signal)

        if np.random.random() < cfg.lead_scale_prob:
            for lead_idx in range(signal.shape[1]):
                signal[:, lead_idx] *= np.random.uniform(*cfg.lead_scale_range)

        if np.random.random() < cfg.lead_masking_prob:
            signal = self._mask_leads(signal)

        return signal

    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        n      = signal.shape[0]
        freq   = np.random.uniform(*self.config.baseline_wander_frequency)
        amp    = self.config.baseline_wander_amplitude * (signal.max() - signal.min())
        wander = amp * np.sin(2 * np.pi * freq * np.arange(n) / 100)
        return signal + wander[:, np.newaxis]

    def _time_warp(self, signal: np.ndarray) -> np.ndarray:
        from scipy.interpolate import CubicSpline

        cfg      = self.config
        n, n_leads = signal.shape

        orig_steps  = np.linspace(0, n - 1, cfg.time_warp_knots)
        warps       = np.random.normal(0, cfg.time_warp_sigma, cfg.time_warp_knots)
        warp_steps  = np.sort(orig_steps + warps * n)
        warp_steps[0], warp_steps[-1] = 0, n - 1

        warper         = CubicSpline(orig_steps, warp_steps)
        warped_indices = np.clip(warper(np.arange(n)), 0, n - 1)

        warped = np.zeros_like(signal)
        for i in range(n_leads):
            warped[:, i] = np.interp(warped_indices, np.arange(n), signal[:, i])
        return warped

    def _mask_leads(self, signal: np.ndarray) -> np.ndarray:
        cfg    = self.config
        n_leads = signal.shape[1]
        n_mask  = np.random.randint(1, min(cfg.lead_masking_max_leads, n_leads - 1) + 1)
        indices = np.random.choice(n_leads, size=n_mask, replace=False)
        out     = signal.copy()
        out[:, indices] = 0
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedECGDataset(Dataset):
    """
    Unified Dataset supporting PTB-XL and Brugada-HUCA ECG recordings.

    Variable-length mode (variable_length=True, default):
        Signals are kept at their native length after resampling, subject to:
          - min_length_seconds: signals shorter than this are zero-padded up to it
          - max_length_seconds: signals longer than this are truncated to it
        Batching requires collate_variable_length() in the DataLoader.

    Fixed-length mode (variable_length=False):
        All signals are padded/truncated to exactly target_length_seconds.
        Compatible with the default PyTorch collate function.
        Use this when fine-tuning from a checkpoint trained in fixed mode.
    """

    def __init__(
        self,
        metadata_list:        List[ECGMetadata],
        data_roots:           Dict[DatasetSource, Path],
        scp_statements_df:    Optional[pd.DataFrame] = None,
        augmentation_config:  Optional[AugmentationConfig] = None,
        normalize:            bool  = True,
        target_sampling_rate: int   = 100,
        # ── Length control ────────────────────────────────────────────────────
        variable_length:      bool  = True,
        target_length_seconds: float = 10.0,   # fixed target OR minimum length
        max_length_seconds:   float  = 30.0,   # cap for variable mode
        is_training:          bool  = False,   # unused directly; augmentation_config=None disables aug
    ):
        self.metadata_list        = metadata_list
        self.data_roots           = data_roots
        self.normalize            = normalize
        self.target_sampling_rate = target_sampling_rate
        self.variable_length      = variable_length
        self.min_length           = int(target_sampling_rate * target_length_seconds)
        self.max_length           = int(target_sampling_rate * max_length_seconds)
        self.target_length        = self.min_length   # kept for back-compat

        self.augmentation = ECGAugmentation(augmentation_config)

        # Fixed superclass order (alphabetical within PTB-XL convention)
        self.superclass_order = [
            DiagnosticSuperclass.NORM,
            DiagnosticSuperclass.MI,
            DiagnosticSuperclass.STTC,
            DiagnosticSuperclass.CD,
            DiagnosticSuperclass.HYP,
        ]

        # Subclass order derived from scp_statements (sorted for reproducibility)
        self.subclass_order: List[str] = self._build_subclass_order(scp_statements_df)

    # ── Subclass order ────────────────────────────────────────────────────────

    def _build_subclass_order(
        self, scp_statements_df: Optional[pd.DataFrame]
    ) -> List[str]:
        subclasses: set = set()
        if scp_statements_df is not None:
            diag_df = scp_statements_df[
                (scp_statements_df.get('diagnostic', pd.Series(dtype=float)) == 1)
                & scp_statements_df['diagnostic_subclass'].notna()
            ]
            for sub in diag_df['diagnostic_subclass'].unique():
                s = str(sub).strip()
                if s:
                    subclasses.add(s)
        return sorted(subclasses)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, idx: int) -> ECGSample:
        metadata = self.metadata_list[idx]

        # 1. Load & resample to target_sampling_rate
        signal = self._load_and_resample_signal(metadata)

        # 2. Bandpass filter (0.5–40 Hz clinical standard)
        signal = self._bandpass_filter(signal)

        # 3. Length handling (variable or fixed)
        signal = self._handle_length(signal)

        # 4. Normalise (per-lead z-score)
        if self.normalize:
            signal = self._normalize_signal(signal)

        # 5. Augment
        signal = self.augmentation(signal)

        # 6. (T, 12) → (12, T) for PyTorch conv layers
        signal_tensor = torch.from_numpy(signal.T).float()

        # 7. Labels
        labels = self._generate_labels(metadata)

        return ECGSample(
            signal=signal_tensor,
            label_superclass=labels['superclass'],
            label_subclass=labels['subclass'],
            patient_id=str(metadata.patient_id),
            source=metadata.dataset_source,
            original_metadata=metadata,
            readable_label=metadata.diagnosis_readable,
        )

    # ── Signal processing ─────────────────────────────────────────────────────

    def _load_and_resample_signal(self, metadata: ECGMetadata) -> np.ndarray:
        """Load via WFDB and resample to target_sampling_rate."""
        root_path = self.data_roots.get(metadata.dataset_source)
        if root_path is None:
            raise ValueError(f"No data root for {metadata.dataset_source}")

        path_str = str(root_path / metadata.final_path)
        path_str = path_str.replace('.dat', '').replace('.hea', '')

        try:
            record      = wfdb.rdrecord(path_str)
            signal      = record.p_signal.astype(np.float32)   # (T, leads)
            original_fs = record.fs

            if original_fs != self.target_sampling_rate:
                signal = self._resample_signal(signal, original_fs, self.target_sampling_rate)

            # Ensure exactly 12 leads
            if signal.shape[1] < 12:
                pad    = np.zeros((signal.shape[0], 12 - signal.shape[1]), dtype=np.float32)
                signal = np.hstack([signal, pad])
            elif signal.shape[1] > 12:
                signal = signal[:, :12]

            return signal

        except Exception as e:
            print(f"Error loading {path_str}: {e}")
            return np.zeros((self.min_length, 12), dtype=np.float32)

    def _resample_signal(
        self,
        signal:      np.ndarray,
        original_fs: float,
        target_fs:   float,
    ) -> np.ndarray:
        n_target  = int(signal.shape[0] * target_fs / original_fs)
        resampled = np.zeros((n_target, signal.shape[1]), dtype=np.float32)
        for i in range(signal.shape[1]):
            resampled[:, i] = scipy_signal.resample(signal[:, i], n_target)
        return resampled

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Zero-phase 4th-order Butterworth bandpass 0.5–40 Hz.
        Skipped if signal is too short for filtfilt or fs is too low.
        """
        nyq  = self.target_sampling_rate / 2.0
        low  = 0.5  / nyq
        high = 40.0 / nyq

        if high >= 1.0 or signal.shape[0] < 27:
            return signal

        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=0).astype(np.float32)

    def _handle_length(self, signal: np.ndarray) -> np.ndarray:
        """
        Variable-length mode:
            - Signals shorter than min_length → right-pad with zeros
            - Signals longer than max_length  → truncate (centre crop optional)
            - Signals in [min_length, max_length] → pass through unchanged

        Fixed-length mode (variable_length=False):
            - All signals padded/truncated to exactly min_length (= target_length)
            - Identical to old _standardize_length() behaviour
        """
        n = signal.shape[0]

        if self.variable_length:
            # Cap at max_length
            if n > self.max_length:
                # Centre-crop rather than head-truncate to preserve mid-signal info
                start  = (n - self.max_length) // 2
                signal = signal[start : start + self.max_length]
                n      = self.max_length

            # Pad up to min_length if too short
            if n < self.min_length:
                pad    = np.zeros((self.min_length - n, signal.shape[1]), dtype=np.float32)
                signal = np.vstack([signal, pad])

        else:
            # Fixed-length: pad or truncate to exactly min_length
            if n < self.min_length:
                pad    = np.zeros((self.min_length - n, signal.shape[1]), dtype=np.float32)
                signal = np.vstack([signal, pad])
            elif n > self.min_length:
                signal = signal[:self.min_length]

        return signal

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Per-lead z-score normalisation."""
        mean = signal.mean(axis=0, keepdims=True)
        std  = signal.std(axis=0,  keepdims=True) + 1e-8
        return (signal - mean) / std

    # ── Label generation ──────────────────────────────────────────────────────

    def _generate_labels(self, metadata: ECGMetadata) -> Dict[str, torch.Tensor]:
        """Multi-hot tensors for superclass (5,) and subclass (N,)."""
        superclass_tensor = torch.zeros(len(self.superclass_order), dtype=torch.float)
        for sc in metadata.diagnostic_superclass:
            try:
                superclass_tensor[self.superclass_order.index(sc)] = 1.0
            except ValueError:
                pass

        subclass_tensor = torch.zeros(len(self.subclass_order), dtype=torch.float)
        for sub in metadata.diagnostic_subclass:
            if sub in self.subclass_order:
                subclass_tensor[self.subclass_order.index(sub)] = 1.0

        return {
            'superclass': superclass_tensor,
            'subclass':   subclass_tensor,
        }
