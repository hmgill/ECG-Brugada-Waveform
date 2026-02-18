"""Unified PyTorch Dataset with signal resampling for variable-length ECGs."""

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
    DatasetSource, DiagnosticSuperclass
)


# ── Augmentation ──────────────────────────────────────────────────────────────

class ECGAugmentation:
    """ECG-specific data augmentation."""

    def __init__(self, config: Optional[AugmentationConfig]):
        self.config = config

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to ECG signal."""
        if self.config is None:
            return signal

        cfg = self.config

        # Amplitude scaling
        if np.random.random() < cfg.amplitude_scale_prob:
            signal = signal * np.random.uniform(*cfg.amplitude_scale_range)

        # Gaussian noise
        if np.random.random() < cfg.noise_prob:
            noise = np.random.normal(0, cfg.noise_std, signal.shape)
            signal = signal + noise * signal.std()

        # Baseline wander
        if np.random.random() < cfg.baseline_wander_prob:
            signal = self._add_baseline_wander(signal)

        # Time warping
        if np.random.random() < cfg.time_warp_prob:
            signal = self._time_warp(signal)

        # Per-lead scaling
        if np.random.random() < cfg.lead_scale_prob:
            for lead_idx in range(signal.shape[1]):
                signal[:, lead_idx] *= np.random.uniform(*cfg.lead_scale_range)

        # Lead masking
        if np.random.random() < cfg.lead_masking_prob:
            signal = self._mask_leads(signal)

        return signal

    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Add low-frequency baseline wander."""
        n_samples = signal.shape[0]
        freq = np.random.uniform(*self.config.baseline_wander_frequency)
        amplitude = self.config.baseline_wander_amplitude * (signal.max() - signal.min())
        t = np.arange(n_samples)
        wander = amplitude * np.sin(2 * np.pi * freq * t / 100)
        return signal + wander[:, np.newaxis]

    def _time_warp(self, signal: np.ndarray) -> np.ndarray:
        """Apply subtle time warping using cubic spline."""
        from scipy.interpolate import CubicSpline

        cfg = self.config
        n_samples, n_leads = signal.shape

        orig_steps = np.linspace(0, n_samples - 1, cfg.time_warp_knots)
        warps = np.random.normal(0, cfg.time_warp_sigma, cfg.time_warp_knots)
        warp_steps = np.sort(orig_steps + warps * n_samples)
        warp_steps[0], warp_steps[-1] = 0, n_samples - 1

        warper = CubicSpline(orig_steps, warp_steps)
        warped_indices = np.clip(warper(np.arange(n_samples)), 0, n_samples - 1)

        warped = np.zeros_like(signal)
        for lead_idx in range(n_leads):
            warped[:, lead_idx] = np.interp(
                warped_indices, np.arange(n_samples), signal[:, lead_idx]
            )
        return warped

    def _mask_leads(self, signal: np.ndarray) -> np.ndarray:
        """
        Randomly zero out ECG leads to make model robust to missing leads.

        Simulates scenarios where only 1-, 3-, or 6-lead ECGs are available,
        or where individual leads fail / have poor signal quality.
        """
        cfg = self.config
        n_samples, n_leads = signal.shape

        num_to_mask = np.random.randint(
            1, min(cfg.lead_masking_max_leads, n_leads - 1) + 1
        )
        mask_indices = np.random.choice(n_leads, size=num_to_mask, replace=False)

        signal_masked = signal.copy()
        signal_masked[:, mask_indices] = 0
        return signal_masked


# ── Dataset ───────────────────────────────────────────────────────────────────

class UnifiedECGDataset(Dataset):
    """
    Unified Dataset supporting PTB-XL and Brugada-HUCA ECG recordings.

    Features:
    - Automatic resampling to target sampling rate
    - Clinical bandpass filter (0.5–40 Hz)
    - Length standardisation via padding / truncation
    - Multi-hot label generation for superclass (5) and subclass (24 incl. BRUG)
    - Support for Brugada-HUCA and PTB-XL datasets
    """

    def __init__(
        self,
        metadata_list: List[ECGMetadata],
        data_roots: Dict[DatasetSource, Path],
        scp_statements_df: Optional[pd.DataFrame] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        normalize: bool = True,
        target_sampling_rate: int = 100,
        target_length_seconds: float = 10.0,
    ):
        self.metadata_list = metadata_list
        self.data_roots = data_roots
        self.normalize = normalize
        self.target_sampling_rate = target_sampling_rate
        self.target_length = int(target_sampling_rate * target_length_seconds)

        self.augmentation = ECGAugmentation(augmentation_config)

        # Fixed superclass order
        self.superclass_order = [
            DiagnosticSuperclass.NORM,
            DiagnosticSuperclass.MI,
            DiagnosticSuperclass.STTC,
            DiagnosticSuperclass.CD,
            DiagnosticSuperclass.HYP,
        ]

        # Build subclass order from scp_statements (sorted for reproducibility)
        # Includes all PTB-XL subclasses + BRUG = 24 total
        self.subclass_order: List[str] = self._build_subclass_order(scp_statements_df)

    def _build_subclass_order(
        self, scp_statements_df: Optional[pd.DataFrame]
    ) -> List[str]:
        """Build a sorted, deduplicated list of all diagnostic subclasses."""
        subclasses: set = set()
        if scp_statements_df is not None:
            # Filter to diagnostic rows that have a subclass label
            diag_df = scp_statements_df[
                (scp_statements_df.get('diagnostic', pd.Series(dtype=float)) == 1)
                & scp_statements_df['diagnostic_subclass'].notna()
            ]
            for sub in diag_df['diagnostic_subclass'].unique():
                s = str(sub).strip()
                if s:
                    subclasses.add(s)
        return sorted(subclasses)

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, idx: int) -> ECGSample:
        metadata = self.metadata_list[idx]

        # 1. Load & resample
        signal = self._load_and_resample_signal(metadata)

        # 2. Bandpass filter (applied at target sampling rate)
        signal = self._bandpass_filter(signal)

        # 3. Standardise length
        signal = self._standardize_length(signal)

        # 4. Normalise
        if self.normalize:
            signal = self._normalize_signal(signal)

        # 5. Augment
        signal = self.augmentation(signal)

        # 6. (Time, Leads) → (Leads, Time) for PyTorch
        signal_tensor = torch.from_numpy(signal.T).float()

        # 7. Generate labels
        labels = self._generate_labels(metadata)

        return ECGSample(
            signal=signal_tensor,
            label_superclass=labels['superclass'],
            label_subclass=labels['subclass'],
            # REMOVED: label_brugada
            patient_id=str(metadata.patient_id),
            source=metadata.dataset_source,
            original_metadata=metadata,
            readable_label=metadata.diagnosis_readable,
        )

    # ── Signal processing ─────────────────────────────────────────────────────

    def _load_and_resample_signal(self, metadata: ECGMetadata) -> np.ndarray:
        """Load ECG signal via WFDB and resample to target sampling rate."""
        root_path = self.data_roots.get(metadata.dataset_source)
        if root_path is None:
            raise ValueError(f"No data root specified for {metadata.dataset_source}")

        full_path = root_path / metadata.final_path
        path_str = str(full_path).replace('.dat', '').replace('.hea', '')

        try:
            record = wfdb.rdrecord(path_str)
            signal = record.p_signal          # (n_samples, n_leads)
            original_fs = record.fs

            if original_fs != self.target_sampling_rate:
                signal = self._resample_signal(signal, original_fs, self.target_sampling_rate)

            # Ensure exactly 12 leads
            if signal.shape[1] < 12:
                padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
                signal = np.hstack([signal, padding])
            elif signal.shape[1] > 12:
                signal = signal[:, :12]

            return signal.astype(np.float32)

        except Exception as e:
            print(f"Error loading {path_str}: {e}")
            return np.zeros((self.target_length, 12), dtype=np.float32)

    def _resample_signal(
        self,
        signal: np.ndarray,
        original_fs: float,
        target_fs: float,
    ) -> np.ndarray:
        """Resample signal using scipy.signal.resample (per lead)."""
        n_target = int(signal.shape[0] * target_fs / original_fs)
        resampled = np.zeros((n_target, signal.shape[1]), dtype=np.float32)
        for i in range(signal.shape[1]):
            resampled[:, i] = scipy_signal.resample(signal[:, i], n_target)
        return resampled

    def _bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Zero-phase clinical ECG bandpass filter (0.5–40 Hz, 4th-order Butterworth).

        - 0.5 Hz high-pass removes baseline wander
        - 40 Hz low-pass removes EMG / powerline noise
        - filtfilt ensures zero phase shift (no ST morphology distortion)

        Applied after resampling so the filter always runs at target_sampling_rate.
        signal shape: (n_samples, n_leads)
        """
        nyq = self.target_sampling_rate / 2.0
        low  = 0.5  / nyq
        high = 40.0 / nyq

        if high >= 1.0:          # target_sampling_rate ≤ 80 Hz — skip
            return signal
        if signal.shape[0] < 27: # too short for filtfilt padding
            return signal

        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal, axis=0)

    def _standardize_length(self, signal: np.ndarray) -> np.ndarray:
        """Pad or truncate to target_length samples."""
        n = signal.shape[0]
        if n < self.target_length:
            padding = np.zeros((self.target_length - n, signal.shape[1]))
            signal = np.vstack([signal, padding])
        elif n > self.target_length:
            signal = signal[:self.target_length, :]
        return signal

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Z-score normalisation per lead."""
        mean = signal.mean(axis=0, keepdims=True)
        std  = signal.std(axis=0,  keepdims=True) + 1e-8
        return (signal - mean) / std

    # ── Label generation ──────────────────────────────────────────────────────

    def _generate_labels(self, metadata: ECGMetadata) -> Dict[str, torch.Tensor]:
        """
        Generate superclass and subclass multi-hot tensors.

        Brugada syndrome cases appear as:
          superclass → CD (index 3)
          subclass   → BRUG (wherever it falls in self.subclass_order)
        """
        # Superclass (5,)
        superclass_tensor = torch.zeros(len(self.superclass_order), dtype=torch.float)
        for sc in metadata.diagnostic_superclass:
            try:
                superclass_tensor[self.superclass_order.index(sc)] = 1.0
            except ValueError:
                pass

        # Subclass (N,) — N = len(subclass_order), typically 24
        subclass_tensor = torch.zeros(len(self.subclass_order), dtype=torch.float)
        for sub in metadata.diagnostic_subclass:
            if sub in self.subclass_order:
                subclass_tensor[self.subclass_order.index(sub)] = 1.0

        return {
            'superclass': superclass_tensor,
            'subclass':   subclass_tensor,
            # REMOVED: 'brugada'
        }
