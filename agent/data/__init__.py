"""Unified data module for multi-dataset ECG classification."""

from .models import (
    DatasetSource,
    DiagnosticSuperclass,
    ECGMetadata,
    ECGSample,
    DataConfig,
    AugmentationConfig,
    DatasetStatistics
)
from .dataset import UnifiedECGDataset, ECGAugmentation
from .datamodule import UnifiedDataModule, load_brugada_metadata, load_ptbxl_metadata

__all__ = [
    'DatasetSource',
    'DiagnosticSuperclass',
    'ECGMetadata',
    'ECGSample',
    'DataConfig',
    'AugmentationConfig',
    'DatasetStatistics',
    'UnifiedECGDataset',
    'ECGAugmentation',
    'UnifiedDataModule',
    'load_brugada_metadata',
    'load_ptbxl_metadata',
]
