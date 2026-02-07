"""Pydantic data models for Brugada ECG classification."""

from pathlib import Path
from typing import Optional, List
from enum import Enum

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DiagnosisLabel(str, Enum):
    """Binary diagnosis label."""
    NORMAL = "normal"
    BRUGADA = "brugada"


class ECGMetadata(BaseModel):
    """Metadata for a single ECG recording."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    patient_id: int
    basal_pattern: int = Field(ge=0, le=1)
    sudden_death: int = Field(ge=0, le=1)
    brugada: int = Field(ge=0, le=2)  # 0=Normal, 1=Brugada, 2=Other/Suspected
    ecg_header_path: Path
    ecg_signal_path: Path
    files_exist: bool
    
    @field_validator('ecg_header_path', 'ecg_signal_path', mode='before')
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v
    
    @property
    def diagnosis_label(self) -> DiagnosisLabel:
        # Treat brugada=2 as brugada for binary classification
        return DiagnosisLabel.BRUGADA if self.brugada >= 1 else DiagnosisLabel.NORMAL


class ECGSample(BaseModel):
    """Complete ECG sample for model input."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    signal: torch.Tensor  # Shape: (n_leads, n_samples)
    label: torch.Tensor
    patient_id: int
    basal_pattern: Optional[int] = None
    sudden_death: Optional[int] = None


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    
    # Paths
    metadata_path: Path
    data_root: Optional[Path] = None
    
    # Splits
    train_split: float = Field(default=0.7, ge=0.0, le=1.0)
    val_split: float = Field(default=0.15, ge=0.0, le=1.0)
    test_split: float = Field(default=0.15, ge=0.0, le=1.0)
    
    # Cross-validation
    n_folds: int = Field(default=5, ge=2)
    stratified: bool = True
    
    # Data loading
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    
    # Preprocessing
    normalize: bool = True
    normalization_method: str = "standardize"
    
    # Augmentation
    augment_train: bool = True
    augment_val: bool = False
    
    # Reproducibility
    random_seed: int = 42
    
    @field_validator('metadata_path', mode='before')
    @classmethod
    def convert_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class AugmentationConfig(BaseModel):
    """ECG signal augmentation configuration."""
    
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    amplitude_scale_prob: float = 0.5
    noise_std: float = 0.05
    noise_prob: float = 0.5
    baseline_wander_amplitude: float = 0.1
    baseline_wander_frequency: tuple[float, float] = (0.1, 0.5)
    baseline_wander_prob: float = 0.3
    time_warp_sigma: float = 0.2
    time_warp_knots: int = 4
    time_warp_prob: float = 0.2
    lead_scale_range: tuple[float, float] = (0.9, 1.1)
    lead_scale_prob: float = 0.3


class DatasetStatistics(BaseModel):
    """Dataset statistics and class weights."""
    
    total_samples: int
    normal_samples: int
    brugada_samples: int
    class_balance_ratio: float
    normal_weight: float
    brugada_weight: float
    pos_weight: float
    missing_files: int = 0
    
    @classmethod
    def from_metadata_list(cls, metadata_list: List[ECGMetadata]) -> "DatasetStatistics":
        """Compute statistics from metadata."""
        total = len(metadata_list)
        # Count brugada >= 1 as positive class for binary classification
        brugada = sum(1 for m in metadata_list if m.brugada >= 1)
        normal = total - brugada
        missing = sum(not m.files_exist for m in metadata_list)
        
        # Inverse frequency weighting
        normal_weight = total / (2 * normal) if normal > 0 else 0.0
        brugada_weight = total / (2 * brugada) if brugada > 0 else 0.0
        pos_weight = brugada_weight / normal_weight if normal_weight > 0 else 1.0
        
        return cls(
            total_samples=total,
            normal_samples=normal,
            brugada_samples=brugada,
            class_balance_ratio=brugada / normal if normal > 0 else 0.0,
            normal_weight=normal_weight,
            brugada_weight=brugada_weight,
            pos_weight=pos_weight,
            missing_files=missing
        )
