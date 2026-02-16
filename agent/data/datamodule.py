"""PyTorch Lightning DataModule for multi-dataset ECG classification."""

from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.model_selection import train_test_split

from .models import (
    ECGMetadata, DataConfig, AugmentationConfig, DatasetStatistics, 
    DatasetSource, DiagnosticSuperclass
)
from .dataset import UnifiedECGDataset


def load_ptbxl_metadata(
    metadata_path: Path,
    scp_statements_path: Path
) -> tuple[List[ECGMetadata], pd.DataFrame]:
    """
    Load PTB-XL metadata and SCP statements.
    
    Returns:
        metadata_list: List of ECGMetadata objects
        scp_statements_df: DataFrame mapping SCP codes to diagnostic classes
    """
    # Load main metadata
    df = pd.read_csv(metadata_path, index_col='ecg_id')
    df.reset_index(inplace=True)
    
    # Load SCP statements (code to diagnostic class mapping)
    scp_statements = pd.read_csv(scp_statements_path, index_col=0)
    
    metadata_list = []
    for _, row in df.iterrows():
        # Skip records with missing filenames
        if pd.isna(row.get('filename_hr')) or row.get('filename_hr') is None:
            continue
        
        # Parse SCP codes from string
        scp_codes_str = row['scp_codes']
        scp_codes = {}
        if isinstance(scp_codes_str, str) and scp_codes_str.strip():
            try:
                # Format: {'CODE1': prob1, 'CODE2': prob2, ...}
                import ast
                scp_codes = ast.literal_eval(scp_codes_str)
            except:
                pass
        
        # Skip if no SCP codes (no diagnosis information)
        if len(scp_codes) == 0:
            continue
        
        # Map SCP codes to diagnostic superclasses
        diagnostic_superclass = []
        for scp_code in scp_codes.keys():
            if scp_code in scp_statements.index:
                # Get the diagnostic class for this SCP code
                diagnostic_class = scp_statements.loc[scp_code, 'diagnostic_class']
                
                # Map to superclass enum (avoid duplicates)
                if diagnostic_class == 'NORM' and DiagnosticSuperclass.NORM not in diagnostic_superclass:
                    diagnostic_superclass.append(DiagnosticSuperclass.NORM)
                elif diagnostic_class == 'MI' and DiagnosticSuperclass.MI not in diagnostic_superclass:
                    diagnostic_superclass.append(DiagnosticSuperclass.MI)
                elif diagnostic_class == 'STTC' and DiagnosticSuperclass.STTC not in diagnostic_superclass:
                    diagnostic_superclass.append(DiagnosticSuperclass.STTC)
                elif diagnostic_class == 'CD' and DiagnosticSuperclass.CD not in diagnostic_superclass:
                    diagnostic_superclass.append(DiagnosticSuperclass.CD)
                elif diagnostic_class == 'HYP' and DiagnosticSuperclass.HYP not in diagnostic_superclass:
                    diagnostic_superclass.append(DiagnosticSuperclass.HYP)
        
        # If no superclass assigned, default to NORM
        if len(diagnostic_superclass) == 0:
            diagnostic_superclass.append(DiagnosticSuperclass.NORM)
        
        # Get filename (use hr if available)
        filename = row.get('filename_hr')
        if pd.isna(filename) or filename is None:
            filename = row.get('filename_lr')
        
        # Skip if still no filename
        if pd.isna(filename) or filename is None:
            continue
        
        metadata = ECGMetadata(
            patient_id=int(row['ecg_id']),
            dataset_source=DatasetSource.PTB_XL,
            filename_hr=filename,  # Store the filename
            ecg_header_path=f"records100/00000/{filename}.hea",
            ecg_signal_path=f"records100/00000/{filename}.dat",
            files_exist=True,
            scp_codes=scp_codes,
            diagnostic_superclass=diagnostic_superclass
        )
        metadata_list.append(metadata)
    
    return metadata_list, scp_statements


def load_brugada_metadata(metadata_path: Path) -> List[ECGMetadata]:
    """Load Brugada-HUCA metadata."""
    df = pd.read_csv(metadata_path)
    
    metadata_list = []
    for _, row in df.iterrows():
        metadata = ECGMetadata(
            patient_id=int(row['patient_id']),
            dataset_source=DatasetSource.BRUGADA_HUCA,
            basal_pattern=row.get('basal_pattern', 0),
            sudden_death=row.get('sudden_death', 0),
            brugada=row.get('brugada', 0),
            ecg_header_path=row['ecg_header_path'],
            ecg_signal_path=row['ecg_signal_path'],
            files_exist=bool(row.get('files_exist', True))
        )
        metadata_list.append(metadata)
    
    return metadata_list


class UnifiedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule supporting multiple ECG datasets.
    
    Handles:
    - Loading Brugada-HUCA and PTB-XL datasets
    - Creating train/val/test splits with stratification
    - Computing dataset statistics and class weights
    - Providing DataLoaders with proper batching
    """
    
    def __init__(self, config: DataConfig, augmentation_config: Optional[AugmentationConfig] = None):
        super().__init__()
        self.config = config
        self.augmentation_config = augmentation_config
        
        self.metadata_list: List[ECGMetadata] = []
        self.train_metadata: List[ECGMetadata] = []
        self.val_metadata: List[ECGMetadata] = []
        self.test_metadata: List[ECGMetadata] = []
        
        self.scp_statements_df: Optional[pd.DataFrame] = None
        self.statistics: Optional[DatasetStatistics] = None
        
        # Data roots mapping
        self.data_roots = {}
    
    def setup(self, stage: Optional[str] = None):
        """Load and split data."""
        all_metadata = []
        
        # Load Brugada-HUCA if enabled
        if self.config.use_brugada:
            brugada_metadata = load_brugada_metadata(self.config.brugada_metadata_path)
            all_metadata.extend(brugada_metadata)
            self.data_roots[DatasetSource.BRUGADA_HUCA] = self.config.brugada_data_root
            print(f"Loaded {len(brugada_metadata)} Brugada samples")
        
        # Load PTB-XL if enabled
        if self.config.use_ptbxl:
            ptbxl_metadata, scp_statements = load_ptbxl_metadata(
                self.config.ptbxl_metadata_path,
                self.config.ptbxl_scp_statements_path
            )
            
            # Sample PTB-XL if needed
            if self.config.ptbxl_sampling_ratio < 1.0:
                import random
                random.seed(self.config.random_seed)
                n_samples = int(len(ptbxl_metadata) * self.config.ptbxl_sampling_ratio)
                ptbxl_metadata = random.sample(ptbxl_metadata, n_samples)
            
            all_metadata.extend(ptbxl_metadata)
            self.data_roots[DatasetSource.PTB_XL] = self.config.ptbxl_data_root
            self.scp_statements_df = scp_statements
            print(f"Loaded {len(ptbxl_metadata)} PTB-XL samples")
        
        self.metadata_list = all_metadata
        
        # Compute statistics
        self.statistics = DatasetStatistics.from_metadata_list(
            self.metadata_list
        )
        
        print(f"\nDataset Statistics:")
        print(f"  Total: {self.statistics.total_samples}")
        print(f"  Normal: {self.statistics.normal_samples}")
        print(f"  MI: {self.statistics.mi_samples}")
        print(f"  STTC: {self.statistics.sttc_samples}")
        print(f"  CD: {self.statistics.cd_samples}")
        print(f"  HYP: {self.statistics.hyp_samples}")
        print(f"  Brugada+: {self.statistics.brugada_positive_samples}")
        
        # Split data
        # For stratification, use brugada label (simpler than multi-hot)
        labels = [1 if m.brugada and m.brugada >= 1 else 0 for m in self.metadata_list]
        
        train_val, self.test_metadata = train_test_split(
            self.metadata_list,
            test_size=self.config.test_split,
            stratify=labels if self.config.stratified else None,
            random_state=self.config.random_seed
        )
        
        train_val_labels = [1 if m.brugada and m.brugada >= 1 else 0 for m in train_val]
        val_size = self.config.val_split / (1 - self.config.test_split)
        self.train_metadata, self.val_metadata = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val_labels if self.config.stratified else None,
            random_state=self.config.random_seed
        )
        
        print(f"\nData Splits:")
        print(f"  Train: {len(self.train_metadata)}")
        print(f"  Val: {len(self.val_metadata)}")
        print(f"  Test: {len(self.test_metadata)}")
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = UnifiedECGDataset(
                self.train_metadata,
                self.data_roots,
                self.scp_statements_df,
                self.augmentation_config if self.config.augment_train else None,
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds
            )
            
            self.val_dataset = UnifiedECGDataset(
                self.val_metadata,
                self.data_roots,
                self.scp_statements_df,
                None,  # No augmentation for validation
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = UnifiedECGDataset(
                self.test_metadata,
                self.data_roots,
                self.scp_statements_df,
                None,
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds
            )
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """
        Custom collate function.
        
        Converts list of ECGSample objects into batched tensors.
        """
        signals = torch.stack([sample.signal for sample in batch])
        
        labels_superclass = torch.stack([sample.label_superclass for sample in batch])
        labels_subclass = torch.stack([sample.label_subclass for sample in batch])
        labels_brugada = torch.stack([sample.label_brugada for sample in batch])
        
        return {
            'signal': signals,
            'labels': {
                'superclass': labels_superclass,
                'subclass': labels_subclass,
                'brugada': labels_brugada
            },
            'metadata': [sample.original_metadata for sample in batch]
        }
    
    def get_class_weights(self) -> Dict:
        """Get class weights for loss functions."""
        if self.statistics is None:
            raise ValueError("Must call setup() first")
        
        return {
            'superclass_weights': self.statistics.superclass_weights,
            'brugada_pos_weight': self.statistics.brugada_pos_weight
        }
