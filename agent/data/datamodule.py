"""PyTorch Lightning DataModule for Brugada dataset."""

from typing import Optional, List

import pandas as pd
import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .models import ECGMetadata, DataConfig, AugmentationConfig, DatasetStatistics
from .dataset import BrugadaDataset


class BrugadaDataModule(pl.LightningDataModule):
    """Lightning DataModule for Brugada ECG classification."""
    
    def __init__(self, config: DataConfig, augmentation_config: Optional[AugmentationConfig] = None):
        super().__init__()
        self.config = config
        self.augmentation_config = augmentation_config
        
        self.metadata_list: List[ECGMetadata] = []
        self.train_metadata: List[ECGMetadata] = []
        self.val_metadata: List[ECGMetadata] = []
        self.test_metadata: List[ECGMetadata] = []
        
        self.statistics: Optional[DatasetStatistics] = None
    
    def setup(self, stage: Optional[str] = None):
        """Load and split data."""
        # Load metadata
        df = pd.read_csv(self.config.metadata_path)
        self.metadata_list = [
            ECGMetadata(**row) for _, row in df.iterrows()
        ]
        
        # Compute statistics
        self.statistics = DatasetStatistics.from_metadata_list(self.metadata_list)
        
        # Split data - use binary labels for stratification (brugada >= 1)
        labels = [1 if m.brugada >= 1 else 0 for m in self.metadata_list]
        train_val, self.test_metadata = train_test_split(
            self.metadata_list,
            test_size=self.config.test_split,
            stratify=labels if self.config.stratified else None,
            random_state=self.config.random_seed
        )
        
        train_val_labels = [1 if m.brugada >= 1 else 0 for m in train_val]
        val_size = self.config.val_split / (1 - self.config.test_split)
        self.train_metadata, self.val_metadata = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val_labels if self.config.stratified else None,
            random_state=self.config.random_seed
        )
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = BrugadaDataset(
                self.train_metadata,
                self.augmentation_config if self.config.augment_train else None,
                self.config.normalize,
                self.config.normalization_method
            )
            self.val_dataset = BrugadaDataset(
                self.val_metadata,
                None,
                self.config.normalize,
                self.config.normalization_method
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = BrugadaDataset(
                self.test_metadata,
                None,
                self.config.normalize,
                self.config.normalization_method
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=lambda x: x
        )

    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=lambda x: x  # Return list of ECGSample objects as-is
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=lambda x: x  # Return list of ECGSample objects as-is
        )
    
    def get_pos_weight(self) -> float:
        """Get pos_weight for BCEWithLogitsLoss."""
        if self.statistics is None:
            raise ValueError("Must call setup() first")
        return self.statistics.pos_weight
