"""
ECG inference engine — fully standalone, no dependency on training code.

Usage:
    from inference.engine import ECGInference

    engine = ECGInference.from_checkpoint(
        checkpoint="checkpoints/best.ckpt",
        config="config/train.yaml",
        thresholds="config/thresholds.json",
    )
    result = engine.predict_file("recordings/patient_001")
    print(result.active_superclasses)   # e.g. ["CD"]
    print(result.active_subclasses)     # e.g. ["BRUG"]
    print(f"Brugada p={result.brugada_probability:.3f}")
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import wfdb
import yaml
from scipy.signal import butter, filtfilt
from scipy.signal import resample as scipy_resample


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR: int = 100          # Hz — must match training config
TARGET_SECONDS: float = 10.0  # seconds — must match training config
TARGET_LEADS: int = 12

SUPERCLASS_NAMES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

# Canonical subclass order (sorted, matches _build_subclass_order in dataset.py).
# This is used as a fallback when the checkpoint does not carry subclass_names.
_FALLBACK_SUBCLASS_NAMES: List[str] = [
    "AMI", "BRUG", "CLBBB", "CRBBB", "ILBBB", "IMI", "IRBBB",
    "ISCA", "ISCI", "ISC_", "IVCD", "LAFB/LPFB", "LAO/LAE", "LMI",
    "LVH", "NST_", "PMI", "RAO/RAE", "RVH", "SEHYP", "STTC", "WPW", "_AVB",
]


# ── Self-contained preprocessing ──────────────────────────────────────────────

def _resample(signal: np.ndarray, original_sr: float, target_sr: float) -> np.ndarray:
    """Resample (n_samples, n_leads) from original_sr to target_sr."""
    if original_sr == target_sr:
        return signal
    n_target = int(round(signal.shape[0] * target_sr / original_sr))
    out = np.zeros((n_target, signal.shape[1]), dtype=np.float32)
    for i in range(signal.shape[1]):
        out[:, i] = scipy_resample(signal[:, i], n_target)
    return out


def _bandpass(signal: np.ndarray, fs: float,
              low_hz: float = 0.5, high_hz: float = 40.0) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    if high >= 1.0 or signal.shape[0] < 27:
        return signal
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0).astype(np.float32)


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalisation per lead."""
    mean = signal.mean(axis=0, keepdims=True)
    std  = signal.std(axis=0, keepdims=True) + 1e-8
    return ((signal - mean) / std).astype(np.float32)


def _standardize_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or truncate to target_length samples."""
    n = signal.shape[0]
    if n < target_length:
        pad = np.zeros((target_length - n, signal.shape[1]), dtype=np.float32)
        return np.vstack([signal, pad])
    return signal[:target_length]


def _fix_leads(signal: np.ndarray, n_leads: int = TARGET_LEADS) -> np.ndarray:
    """Ensure exactly n_leads by zero-padding or truncating."""
    if signal.shape[1] < n_leads:
        pad = np.zeros((signal.shape[0], n_leads - signal.shape[1]), dtype=np.float32)
        return np.hstack([signal, pad])
    return signal[:, :n_leads]


def preprocess(signal: np.ndarray, original_sr: float) -> np.ndarray:
    """
    Full preprocessing pipeline matching training:
        resample → fix leads → bandpass → standardise length → z-score

    Args:
        signal:      (n_samples, n_leads) float array in mV
        original_sr: sampling rate of the input signal

    Returns:
        (TARGET_SR * TARGET_SECONDS, 12) float32 array
    """
    signal = signal.astype(np.float32)
    signal = _resample(signal, original_sr, TARGET_SR)
    signal = _fix_leads(signal)
    signal = _bandpass(signal, TARGET_SR)
    signal = _standardize_length(signal, int(TARGET_SR * TARGET_SECONDS))
    signal = _normalize(signal)
    return signal


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def _load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[nn.Module, List[str]]:
    """
    Load a MultiTaskClassifier from checkpoint.

    Returns:
        (model, subclass_names) — subclass_names taken from the checkpoint's
        saved hyperparameters if available, otherwise derived from the config.
    """
    # Import here so the engine can still be imported even if the training
    # package is not on the path (it will only fail at load time).
    try:
        from models.ecg_transformer import create_ecg_transformer_rope
        from models.losses import get_loss_function
        from models.lightning_module import MultiTaskClassifier
    except ImportError as e:
        raise ImportError(
            "Could not import training modules. Make sure the agent/ directory "
            "is on sys.path, e.g. run from agent/ or add it with sys.path.insert()."
        ) from e

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mc        = cfg["model"]
    loss_cfg  = cfg["loss"]
    loss_type = loss_cfg["type"]

    # ── Try to recover subclass_names from checkpoint hparams ────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    subclass_names: List[str] = hparams.get("subclass_names", _FALLBACK_SUBCLASS_NAMES)

    # ── Build base model ─────────────────────────────────────────────────────
    base_model = create_ecg_transformer_rope(
        model_size=mc["size"],
        in_channels=mc["in_channels"],
        num_superclasses=mc["num_superclasses"],
        num_subclasses=len(subclass_names),
        dropout=mc.get("dropout", 0.1),
    )

    # ── Build loss (needed to restore Lightning module) ───────────────────────
    if loss_type == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        loss_fn = get_loss_function(
            "focal",
            task_weights=loss_cfg.get("task_weights", {}),
            alpha_superclass=focal_cfg.get("alpha_superclass", 0.25),
            alpha_subclass=focal_cfg.get("alpha_subclass", 0.25),
            gamma=focal_cfg.get("gamma", 2.0),
            subclass_alpha_overrides=focal_cfg.get("subclass_alpha_overrides"),
            subclass_names=subclass_names,
        )
    elif loss_type == "weighted_bce":
        loss_fn = get_loss_function(
            "weighted_bce",
            task_weights=loss_cfg.get("task_weights", {}),
            superclass_weights=torch.ones(mc["num_superclasses"]),
        )
    else:
        loss_fn = get_loss_function(loss_type, task_weights=loss_cfg.get("task_weights", {}))

    model = MultiTaskClassifier.load_from_checkpoint(
        checkpoint_path,
        model=base_model,
        loss_fn=loss_fn,
        subclass_names=subclass_names,
        map_location=device,
        strict=False,  # loss_fn weights are not used at inference; avoids
                       # alpha buffer mismatches between training/inference configs
    )
    model.eval()
    model.to(device)
    return model, subclass_names


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Holds probabilities and threshold-gated binary predictions for one recording.

    Attributes:
        sup_probs:   (5,)  superclass probabilities
        sub_probs:   (N,)  subclass probabilities
        sup_preds:   (5,)  binary superclass predictions (threshold-gated)
        sub_preds:   (N,)  binary subclass predictions
        superclass_names: ordered superclass names
        subclass_names:   ordered subclass names
        signal_length_seconds: duration of the preprocessed signal
    """
    sup_probs:            np.ndarray
    sub_probs:            np.ndarray
    sup_preds:            np.ndarray
    sub_preds:            np.ndarray
    superclass_names:     List[str]
    subclass_names:       List[str]
    signal_length_seconds: float

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def brugada_probability(self) -> float:
        idx = self.subclass_names.index("BRUG")
        return float(self.sub_probs[idx])

    @property
    def brugada_positive(self) -> bool:
        idx = self.subclass_names.index("BRUG")
        return bool(self.sub_preds[idx])

    @property
    def active_superclasses(self) -> List[str]:
        return [n for n, p in zip(self.superclass_names, self.sup_preds) if p]

    @property
    def active_subclasses(self) -> List[str]:
        return [n for n, p in zip(self.subclass_names, self.sub_preds) if p]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "signal_length_seconds": self.signal_length_seconds,
            "superclass": {
                name: {
                    "probability": float(self.sup_probs[i]),
                    "predicted":   bool(self.sup_preds[i]),
                }
                for i, name in enumerate(self.superclass_names)
            },
            "subclass": {
                name: {
                    "probability": float(self.sub_probs[i]),
                    "predicted":   bool(self.sub_preds[i]),
                }
                for i, name in enumerate(self.subclass_names)
            },
            "active_superclasses": self.active_superclasses,
            "active_subclasses":   self.active_subclasses,
            "brugada": {
                "probability": self.brugada_probability,
                "positive":    self.brugada_positive,
            },
        }

    def summary(self) -> str:
        """Human-readable one-block summary."""
        lines = [
            f"Signal length : {self.signal_length_seconds:.1f}s",
            "",
            "Superclass predictions:",
        ]
        for i, name in enumerate(self.superclass_names):
            flag = "✓" if self.sup_preds[i] else " "
            lines.append(f"  [{flag}] {name:<12} p={self.sup_probs[i]:.4f}")

        lines += ["", "Subclass predictions:"]
        for i, name in enumerate(self.subclass_names):
            flag = "✓" if self.sub_preds[i] else " "
            marker = " ◄ BRUGADA" if name == "BRUG" else ""
            lines.append(f"  [{flag}] {name:<14} p={self.sub_probs[i]:.4f}{marker}")

        lines += [
            "",
            f"Active superclasses : {self.active_superclasses or 'none'}",
            f"Active subclasses   : {self.active_subclasses or 'none'}",
            f"Brugada             : {'POSITIVE' if self.brugada_positive else 'negative'}"
            f"  (p={self.brugada_probability:.4f})",
        ]
        return "\n".join(lines)


# ── Inference engine ──────────────────────────────────────────────────────────

class ECGInference:
    """
    Wraps a loaded model + per-class thresholds for file or array inference.

    Instantiate via the class method:

        engine = ECGInference.from_checkpoint(
            checkpoint="checkpoints/best.ckpt",
            config="config/train.yaml",
            thresholds="config/thresholds.json",
        )

    Then run predictions:

        result = engine.predict_file("recordings/patient_001")
        result = engine.predict_array(signal_array, sampling_rate=500)
    """

    def __init__(
        self,
        model:           nn.Module,
        sup_thresholds:  Dict[str, float],
        sub_thresholds:  Dict[str, float],
        superclass_names: List[str],
        subclass_names:  List[str],
        device:          torch.device,
    ):
        self.model            = model
        self.sup_thresholds   = sup_thresholds
        self.sub_thresholds   = sub_thresholds
        self.superclass_names = superclass_names
        self.subclass_names   = subclass_names
        self.device           = device

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint:  Union[Path, str],
        config:      Union[Path, str],
        thresholds:  Union[Path, str],
        device:      Optional[str] = None,
    ) -> "ECGInference":
        """
        Load engine from a checkpoint, config YAML, and thresholds JSON.

        Args:
            checkpoint:  Path to .ckpt file produced by MultiTaskClassifier
            config:      Path to train.yaml used for this training run
            thresholds:  Path to thresholds.json produced by find_thresholds.py
            device:      'cuda', 'cpu', or None (auto-detect)
        """
        _device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with open(thresholds) as f:
            thresh_data = json.load(f)

        sup_thresholds: Dict[str, float] = thresh_data["superclass"]["thresholds"]
        sub_thresholds: Dict[str, float] = thresh_data["subclass"]["thresholds"]

        model, subclass_names = _load_model(Path(checkpoint), Path(config), _device)

        print(f"✓ Model loaded on {_device}")
        print(f"  Superclasses : {SUPERCLASS_NAMES}")
        print(f"  Subclasses   : {subclass_names}")

        return cls(
            model=model,
            sup_thresholds=sup_thresholds,
            sub_thresholds=sub_thresholds,
            superclass_names=SUPERCLASS_NAMES,
            subclass_names=subclass_names,
            device=_device,
        )

    # ── Core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_array(
        self,
        signal:       np.ndarray,
        sampling_rate: float,
    ) -> PredictionResult:
        """
        Run inference on a raw numpy ECG array.

        Args:
            signal:        (n_samples, n_leads) float array in mV
            sampling_rate: original sampling rate of the signal

        Returns:
            PredictionResult
        """
        processed = preprocess(signal, sampling_rate)
        # (time, leads) → (1, leads, time) batch tensor
        tensor = torch.from_numpy(processed.T).float().unsqueeze(0).to(self.device)
        preds  = self.model(tensor)

        sup_probs = torch.sigmoid(preds["superclass"]).cpu().numpy()[0]
        sub_probs = torch.sigmoid(preds["subclass"]).cpu().numpy()[0]

        sup_preds = np.array([
            int(sup_probs[i] >= self.sup_thresholds.get(name, 0.5))
            for i, name in enumerate(self.superclass_names)
        ])
        sub_preds = np.array([
            int(sub_probs[i] >= self.sub_thresholds.get(name, 0.5))
            for i, name in enumerate(self.subclass_names)
        ])

        return PredictionResult(
            sup_probs=sup_probs,
            sub_probs=sub_probs,
            sup_preds=sup_preds,
            sub_preds=sub_preds,
            superclass_names=self.superclass_names,
            subclass_names=self.subclass_names,
            signal_length_seconds=processed.shape[0] / TARGET_SR,
        )

    @torch.no_grad()
    def predict_file(self, wfdb_path: Union[Path, str]) -> PredictionResult:
        """
        Run inference on a WFDB recording (.hea / .dat pair).

        Args:
            wfdb_path: Path to the recording (with or without extension)

        Returns:
            PredictionResult
        """
        path_str = str(wfdb_path).removesuffix(".hea").removesuffix(".dat")
        record   = wfdb.rdrecord(path_str)
        signal   = record.p_signal.astype(np.float32)
        return self.predict_array(signal, float(record.fs))

    def predict_batch(
        self,
        paths: List[Union[Path, str]],
        verbose: bool = True,
    ) -> List[PredictionResult]:
        """
        Run inference on a list of WFDB file paths.

        Args:
            paths:   List of recording paths
            verbose: Print progress

        Returns:
            List of PredictionResult, one per path (None on load error)
        """
        results = []
        for i, p in enumerate(paths):
            if verbose:
                print(f"  [{i+1}/{len(paths)}] {p}", end="\r")
            try:
                results.append(self.predict_file(p))
            except Exception as e:
                print(f"\n  Warning: failed on {p}: {e}")
                results.append(None)
        if verbose:
            print()
        return results
