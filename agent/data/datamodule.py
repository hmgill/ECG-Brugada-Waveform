"""PyTorch Lightning DataModule for multi-dataset ECG classification."""

import ast
import random
from typing import Optional, List, Dict
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .models import (
    ECGMetadata, DataConfig, AugmentationConfig, DatasetStatistics,
    DatasetSource, DiagnosticSuperclass
)
from .dataset import UnifiedECGDataset


# ── Metadata loaders ──────────────────────────────────────────────────────────

def load_metadata_unified(
    metadata_path: Path,
    scp_statements_path: Path,
    dataset_source: DatasetSource,
) -> tuple[List[ECGMetadata], pd.DataFrame]:
    """
    Load metadata in PTB-XL format for either PTB-XL or Brugada-HUCA.

    Both datasets now share the same CSV structure (produced by
    convert_brugada_to_ptbxl_structure.py for Brugada), so a single loader
    handles both.  SCP codes are resolved to superclass / subclass via the
    accompanying scp_statements CSV.

    Returns:
        metadata_list: List of ECGMetadata objects
        scp_statements_df: DataFrame indexed by SCP code
    """
    df = pd.read_csv(metadata_path)
    scp_statements = pd.read_csv(scp_statements_path, index_col=0)

    metadata_list = []
    for _, row in df.iterrows():
        # ── Parse SCP codes ──────────────────────────────────────────────────
        scp_codes_str = row.get('scp_codes', '')
        scp_codes: Dict[str, float] = {}
        if isinstance(scp_codes_str, str) and scp_codes_str.strip():
            try:
                scp_codes = ast.literal_eval(scp_codes_str)
            except (ValueError, SyntaxError):
                pass

        if not scp_codes:
            continue

        # ── Map SCP codes → superclass / subclass ────────────────────────────
        diagnostic_superclass: List[DiagnosticSuperclass] = []
        diagnostic_subclass: List[str] = []

        for scp_code in scp_codes:
            if scp_code not in scp_statements.index:
                continue

            diag_class    = scp_statements.loc[scp_code, 'diagnostic_class']
            diag_subclass = scp_statements.loc[scp_code, 'diagnostic_subclass']

            # Superclass
            try:
                sc = DiagnosticSuperclass(diag_class)
                if sc not in diagnostic_superclass:
                    diagnostic_superclass.append(sc)
            except ValueError:
                pass

            # Subclass
            if pd.notna(diag_subclass) and str(diag_subclass).strip():
                sub = str(diag_subclass).strip()
                if sub not in diagnostic_subclass:
                    diagnostic_subclass.append(sub)

        if not diagnostic_superclass:
            continue

        # ── Build ECGMetadata ─────────────────────────────────────────────────
        metadata = ECGMetadata(
            patient_id=int(row['patient_id']),
            ecg_id=int(row['ecg_id']) if pd.notna(row.get('ecg_id')) else int(row['patient_id']),
            dataset_source=dataset_source,
            filename_lr=row.get('filename_lr'),
            filename_hr=row.get('filename_hr'),
            scp_codes=scp_codes,
            diagnostic_superclass=diagnostic_superclass,
            diagnostic_subclass=diagnostic_subclass,
            strat_fold=int(row['strat_fold']) if pd.notna(row.get('strat_fold')) else None,
            # Demographics (NaN for Brugada dataset)
            age=float(row['age']) if pd.notna(row.get('age')) else None,
            sex=int(row['sex']) if pd.notna(row.get('sex')) else None,
            # Brugada-specific columns (present only in brugada_database.csv)
            brugada=int(row['brugada_label']) if pd.notna(row.get('brugada_label')) else None,
            basal_pattern=int(row['basal_pattern']) if pd.notna(row.get('basal_pattern')) else None,
            sudden_death=int(row['sudden_death']) if pd.notna(row.get('sudden_death')) else None,
        )
        metadata_list.append(metadata)

    return metadata_list, scp_statements


# ── DataModule ────────────────────────────────────────────────────────────────

class UnifiedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule supporting PTB-XL and Brugada-HUCA datasets.

    Both datasets are loaded through the same PTB-XL-structured CSV format,
    producing a unified metadata list with superclass / subclass labels.
    The Brugada syndrome cases appear as CD superclass + BRUG subclass.
    """

    def __init__(
        self,
        config: DataConfig,
        augmentation_config: Optional[AugmentationConfig] = None
    ):
        super().__init__()
        self.config = config
        self.augmentation_config = augmentation_config

        self.metadata_list:  List[ECGMetadata] = []
        self.train_metadata: List[ECGMetadata] = []
        self.val_metadata:   List[ECGMetadata] = []
        self.test_metadata:  List[ECGMetadata] = []

        self.scp_statements_df: Optional[pd.DataFrame] = None
        self.statistics:        Optional[DatasetStatistics] = None
        self.data_roots:        Dict[DatasetSource, Path] = {}

        # Populated after setup() — used by train.py to build the model
        self.train_dataset: Optional[UnifiedECGDataset] = None
        self.val_dataset:   Optional[UnifiedECGDataset] = None
        self.test_dataset:  Optional[UnifiedECGDataset] = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        """Load datasets, compute statistics, and create train/val/test splits."""
        all_metadata: List[ECGMetadata] = []
        all_scp: List[pd.DataFrame] = []

        # ── Brugada-HUCA ──────────────────────────────────────────────────────
        if self.config.use_brugada:
            brugada_metadata, brugada_scp = load_metadata_unified(
                self.config.brugada_metadata_path,
                self.config.brugada_scp_statements_path,
                DatasetSource.BRUGADA_HUCA,
            )
            all_metadata.extend(brugada_metadata)
            all_scp.append(brugada_scp)
            self.data_roots[DatasetSource.BRUGADA_HUCA] = self.config.brugada_data_root
            print(f"Loaded {len(brugada_metadata)} Brugada-HUCA samples")

        # ── PTB-XL ────────────────────────────────────────────────────────────
        if self.config.use_ptbxl:
            ptbxl_metadata, ptbxl_scp = load_metadata_unified(
                self.config.ptbxl_metadata_path,
                self.config.ptbxl_scp_statements_path,
                DatasetSource.PTB_XL,
            )

            if self.config.ptbxl_sampling_ratio < 1.0:
                random.seed(self.config.random_seed)
                n = int(len(ptbxl_metadata) * self.config.ptbxl_sampling_ratio)
                ptbxl_metadata = random.sample(ptbxl_metadata, n)

            all_metadata.extend(ptbxl_metadata)
            all_scp.append(ptbxl_scp)
            self.data_roots[DatasetSource.PTB_XL] = self.config.ptbxl_data_root
            print(f"Loaded {len(ptbxl_metadata)} PTB-XL samples")

        # ── Merge SCP statements (deduplicated) ───────────────────────────────
        self.scp_statements_df = (
            pd.concat(all_scp).loc[~pd.concat(all_scp).index.duplicated()]
            if all_scp else None
        )

        self.metadata_list = all_metadata

        # ── Statistics ────────────────────────────────────────────────────────
        self.statistics = DatasetStatistics.from_metadata_list(self.metadata_list)

        print(f"\nDataset Statistics:")
        print(f"  Total:  {self.statistics.total_samples}")
        print(f"  NORM:   {self.statistics.normal_samples}")
        print(f"  MI:     {self.statistics.mi_samples}")
        print(f"  STTC:   {self.statistics.sttc_samples}")
        print(f"  CD:     {self.statistics.cd_samples}  (incl. Brugada positive)")
        print(f"  HYP:    {self.statistics.hyp_samples}")

        # ── Stratified train / val / test split ───────────────────────────────
        # Stratify by whether the sample has the BRUG subclass label, so that
        # the rare positive class is proportionally represented in every split.
        strat_labels = [
            1 if 'BRUG' in m.diagnostic_subclass else 0
            for m in self.metadata_list
        ]

        train_val, self.test_metadata = train_test_split(
            self.metadata_list,
            test_size=self.config.test_split,
            stratify=strat_labels if self.config.stratified else None,
            random_state=self.config.random_seed,
        )
        train_val_labels = [
            1 if 'BRUG' in m.diagnostic_subclass else 0
            for m in train_val
        ]
        val_size = self.config.val_split / (1 - self.config.test_split)
        self.train_metadata, self.val_metadata = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val_labels if self.config.stratified else None,
            random_state=self.config.random_seed,
        )

        print(f"\nData Splits:")
        print(f"  Train: {len(self.train_metadata)}")
        print(f"  Val:   {len(self.val_metadata)}")
        print(f"  Test:  {len(self.test_metadata)}")

        # ── Create datasets ───────────────────────────────────────────────────
        if stage in ("fit", None):
            self.train_dataset = UnifiedECGDataset(
                self.train_metadata,
                self.data_roots,
                self.scp_statements_df,
                self.augmentation_config if self.config.augment_train else None,
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds,
            )
            self.val_dataset = UnifiedECGDataset(
                self.val_metadata,
                self.data_roots,
                self.scp_statements_df,
                None,
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds,
            )

        if stage in ("test", None):
            self.test_dataset = UnifiedECGDataset(
                self.test_metadata,
                self.data_roots,
                self.scp_statements_df,
                None,
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds,
            )

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    # ── Collate ───────────────────────────────────────────────────────────────

    def _collate_fn(self, batch) -> Dict:
        """Convert a list of ECGSample objects into batched tensors."""
        return {
            'signal': torch.stack([s.signal for s in batch]),
            'labels': {
                'superclass': torch.stack([s.label_superclass for s in batch]),
                'subclass':   torch.stack([s.label_subclass   for s in batch]),
                # REMOVED: 'brugada'
            },
            'metadata': [s.original_metadata for s in batch],
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_class_weights(self) -> Dict:
        """
        Return class weights for loss functions.

        superclass_weights: dict mapping class name → pos_weight (neg/pos ratio)
        """
        if self.statistics is None:
            raise ValueError("Must call setup() first")

        return {
            'superclass_weights': self.statistics.superclass_weights,
            # REMOVED: 'brugada_pos_weight'
        }
