"""Unified data models for Multi-Dataset ECG Classification (Brugada + PTB-XL)."""

from pathlib import Path
from typing import Optional, Dict, Union, List
from enum import Enum
import ast

import torch
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DatasetSource(str, Enum):
    """Source of the ECG recording."""
    BRUGADA_HUCA = "brugada_huca"
    PTB_XL = "ptb_xl"


class DiagnosticSuperclass(str, Enum):
    """PTB-XL Diagnostic Superclasses."""
    NORM = "NORM"
    MI   = "MI"
    STTC = "STTC"
    CD   = "CD"
    HYP  = "HYP"

    @property
    def description(self) -> str:
        mapping = {
            "NORM": "Normal ECG",
            "MI":   "Myocardial Infarction",
            "STTC": "ST/T Change",
            "CD":   "Conduction Disturbance",
            "HYP":  "Hypertrophy",
        }
        return mapping.get(self.value, self.value)


class ECGMetadata(BaseModel):
    """Unified metadata for Brugada-HUCA and PTB-XL."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    ecg_id:     Optional[float] = None   # float to tolerate NaN from CSV
    patient_id: Optional[float] = None

    # ── Demographics ──────────────────────────────────────────────────────────
    age:    Optional[float] = None
    sex:    Optional[float] = None   # float to tolerate NaN from CSV
    height: Optional[float] = None
    weight: Optional[float] = None

    # ── Recording ─────────────────────────────────────────────────────────────
    recording_date: Optional[str] = None
    report:         Optional[str] = None
    device:         Optional[str] = None

    # ── Diagnostic labels ─────────────────────────────────────────────────────
    scp_codes:             Optional[str]       = None
    diagnostic_superclass: List[DiagnosticSuperclass] = Field(default_factory=list)
    diagnostic_subclass:   List[str]           = Field(default_factory=list)

    # ── File paths ────────────────────────────────────────────────────────────
    filename_lr:  Optional[str] = None
    filename_hr:  Optional[str] = None
    final_path:   Optional[str] = None

    # ── Dataset source ────────────────────────────────────────────────────────
    dataset_source: DatasetSource = DatasetSource.PTB_XL

    # ── Human-readable label ──────────────────────────────────────────────────
    diagnosis_readable: List[str] = Field(default_factory=list)

    # ── PTB-XL fold (for cross-validation) ───────────────────────────────────
    strat_fold: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ECGSample(BaseModel):
    """A single processed ECG sample ready for model input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: torch.Tensor                # (n_leads, n_samples) — variable length
    label_superclass: torch.Tensor      # multi-hot (5,)
    label_subclass:   torch.Tensor      # multi-hot (N,)  — incl. BRUG

    patient_id: str
    source: DatasetSource
    original_metadata: ECGMetadata
    readable_label: List[str] = Field(default_factory=list)


class AugmentationConfig(BaseModel):
    """ECG signal augmentation configuration."""

    # ── Existing augmentations ────────────────────────────────────────────────
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    amplitude_scale_prob:  float = 0.3

    noise_std:  float = 0.05
    noise_prob: float = 0.3

    baseline_wander_amplitude: float = 0.1
    baseline_wander_frequency: tuple[float, float] = (0.1, 0.5)
    baseline_wander_prob:      float = 0.2

    time_warp_sigma: float = 0.2
    time_warp_knots: int   = 4
    time_warp_prob:  float = 0.15

    lead_scale_range: tuple[float, float] = (0.9, 1.1)
    lead_scale_prob:  float = 0.2

    lead_masking_prob:      float = 0.3
    lead_masking_max_leads: int   = 6

    # ── Random crop (variable-length training) ────────────────────────────────
    # When crop_prob > 0, each training sample is randomly cropped to a length
    # drawn uniformly from [min_crop_seconds, signal_length].  This exposes the
    # model to sub-full-length inputs during training so it generalises to
    # shorter recordings at inference time.
    #
    # Recommended starting values:
    #   crop_prob:         0.5   — apply to half of training samples
    #   min_crop_seconds:  3.0   — shortest crop (300 samples @ 100 Hz)
    #
    # Set crop_prob: 0.0 to disable and reproduce the original behaviour.
    crop_prob:         float = 0.5
    min_crop_seconds:  float = 3.0


class DataConfig(BaseModel):
    """Unified data configuration for multi-dataset training."""

    # ── Dataset flags & paths ─────────────────────────────────────────────────
    use_brugada: bool = True
    brugada_metadata_path:      Optional[Path] = None
    brugada_scp_statements_path: Optional[Path] = None
    brugada_data_root:          Optional[Path] = None

    use_ptbxl: bool = True
    ptbxl_metadata_path:      Optional[Path] = None
    ptbxl_scp_statements_path: Optional[Path] = None
    ptbxl_data_root:          Optional[Path] = None
    ptbxl_sampling_ratio: float = 1.0

    # ── Signal processing ─────────────────────────────────────────────────────
    target_sampling_rate:   int   = 100
    target_length_seconds:  float = 10.0   # used for val/test standardisation

    # ── Splits ────────────────────────────────────────────────────────────────
    train_split: float = Field(default=0.7,  ge=0.0, le=1.0)
    val_split:   float = Field(default=0.15, ge=0.0, le=1.0)
    test_split:  float = Field(default=0.15, ge=0.0, le=1.0)
    n_folds:     int   = Field(default=5, ge=2)
    stratified:  bool  = True

    # ── Data loading ──────────────────────────────────────────────────────────
    batch_size:  int  = Field(default=32, ge=1)
    num_workers: int  = Field(default=4,  ge=0)
    pin_memory:  bool = True

    # ── Preprocessing ─────────────────────────────────────────────────────────
    normalize:            bool = True
    normalization_method: str  = "standardize"

    # ── Augmentation ──────────────────────────────────────────────────────────
    augment_train: bool = True
    augment_val:   bool = False

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed: int = 42

    @field_validator(
        'brugada_metadata_path', 'brugada_scp_statements_path', 'brugada_data_root',
        'ptbxl_metadata_path', 'ptbxl_scp_statements_path', 'ptbxl_data_root',
        mode='before'
    )
    @classmethod
    def convert_path(cls, v):
        return Path(v) if isinstance(v, str) and v else v


class DatasetStatistics(BaseModel):
    """Dataset statistics computed from training metadata."""

    total_samples: int

    # By source
    brugada_samples: int = 0
    ptbxl_samples:   int = 0

    # By superclass
    normal_samples: int = 0
    mi_samples:     int = 0
    sttc_samples:   int = 0
    cd_samples:     int = 0
    hyp_samples:    int = 0

    # Superclass pos_weights for BCEWithLogitsLoss  (neg / pos ratio)
    superclass_weights: Dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_metadata_list(cls, metadata_list: List["ECGMetadata"]) -> "DatasetStatistics":
        """Compute statistics from a list of ECGMetadata."""
        total = len(metadata_list)

        brugada_count = sum(1 for m in metadata_list if m.dataset_source == DatasetSource.BRUGADA_HUCA)
        ptbxl_count   = sum(1 for m in metadata_list if m.dataset_source == DatasetSource.PTB_XL)

        normal_count = sum(1 for m in metadata_list if DiagnosticSuperclass.NORM in m.diagnostic_superclass)
        mi_count     = sum(1 for m in metadata_list if DiagnosticSuperclass.MI   in m.diagnostic_superclass)
        sttc_count   = sum(1 for m in metadata_list if DiagnosticSuperclass.STTC in m.diagnostic_superclass)
        cd_count     = sum(1 for m in metadata_list if DiagnosticSuperclass.CD   in m.diagnostic_superclass)
        hyp_count    = sum(1 for m in metadata_list if DiagnosticSuperclass.HYP  in m.diagnostic_superclass)

        # pos_weight = n_negative / n_positive  (BCEWithLogitsLoss convention)
        superclass_weights = {}
        for name, count in [
            ("NORM", normal_count), ("MI", mi_count), ("STTC", sttc_count),
            ("CD",   cd_count),     ("HYP", hyp_count),
        ]:
            neg = total - count
            superclass_weights[name] = (neg / count) if count > 0 else 1.0

        return cls(
            total_samples=total,
            brugada_samples=brugada_count,
            ptbxl_samples=ptbxl_count,
            normal_samples=normal_count,
            mi_samples=mi_count,
            sttc_samples=sttc_count,
            cd_samples=cd_count,
            hyp_samples=hyp_count,
            superclass_weights=superclass_weights,
        )
