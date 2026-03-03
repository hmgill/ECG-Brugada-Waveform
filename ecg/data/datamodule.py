"""PyTorch Lightning DataModule for multi-dataset ECG classification."""

import ast
import random
from typing import Optional, List, Dict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
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

    Both datasets share the same CSV structure (produced by
    convert_brugada_to_ptbxl_structure.py for Brugada), so a single loader
    handles both.  SCP codes are resolved to superclass / subclass via the
    accompanying scp_statements CSV.

    Returns:
        metadata_list:     List of ECGMetadata objects
        scp_statements_df: DataFrame indexed by SCP code
    """
    df = pd.read_csv(metadata_path)
    scp_statements = pd.read_csv(scp_statements_path, index_col=0)

    metadata_list = []
    for _, row in df.iterrows():
        # ── Parse SCP codes ──────────────────────────────────────────────────
        scp_codes_str = row.get("scp_codes", "")
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
        diagnostic_subclass:   List[str] = []

        for scp_code in scp_codes:
            if scp_code not in scp_statements.index:
                continue

            diag_class    = scp_statements.loc[scp_code, "diagnostic_class"]
            diag_subclass = scp_statements.loc[scp_code, "diagnostic_subclass"]
            is_diagnostic = scp_statements.loc[scp_code, "diagnostic"]

            if is_diagnostic != 1:
                continue

            if pd.notna(diag_class) and diag_class:
                try:
                    diagnostic_superclass.append(DiagnosticSuperclass(str(diag_class).strip()))
                except ValueError:
                    pass

            if pd.notna(diag_subclass) and diag_subclass:
                diagnostic_subclass.append(str(diag_subclass).strip())

        # ── Build ECGMetadata ─────────────────────────────────────────────────
        final_path = row.get("filename_hr") or row.get("filename_lr")

        readable_labels = list(set(
            [sc.value for sc in diagnostic_superclass] + diagnostic_subclass
        ))

        metadata = ECGMetadata(
            ecg_id=row.get("ecg_id"),
            patient_id=row.get("patient_id"),
            age=row.get("age"),
            sex=row.get("sex"),
            height=row.get("height"),
            weight=row.get("weight"),
            recording_date=str(row.get("recording_date", "")),
            report=row.get("report", ""),
            device=row.get("device", ""),
            scp_codes=scp_codes_str,
            diagnostic_superclass=list(set(diagnostic_superclass)),
            diagnostic_subclass=list(set(diagnostic_subclass)),
            filename_lr=row.get("filename_lr"),
            filename_hr=row.get("filename_hr"),
            final_path=str(final_path) if pd.notna(final_path) else None,
            dataset_source=dataset_source,
            diagnosis_readable=readable_labels,
            strat_fold=row.get("strat_fold"),
        )
        metadata_list.append(metadata)

    return metadata_list, scp_statements


# ── DataModule ────────────────────────────────────────────────────────────────

class UnifiedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for multi-dataset ECG classification.

    Both datasets are loaded through the same PTB-XL-structured CSV format,
    producing a unified metadata list with superclass / subclass labels.
    The Brugada syndrome cases appear as CD superclass + BRUG subclass.

    Variable-length training
    ------------------------
    Training datasets are constructed with is_training=True, which enables
    random cropping via AugmentationConfig.crop_prob / min_crop_seconds.
    Validation and test datasets always use fixed-length standardisation so
    that threshold calibration remains reproducible.

    The training collate function pads each batch to the longest sample
    within that batch rather than a global fixed length, so the model sees
    genuinely variable-length tensors during training.
    """

    def __init__(
        self,
        config: DataConfig,
        augmentation_config: Optional[AugmentationConfig] = None
    ):
        super().__init__()
        self.config               = config
        self.augmentation_config  = augmentation_config

        self.metadata_list:  List[ECGMetadata] = []
        self.train_metadata: List[ECGMetadata] = []
        self.val_metadata:   List[ECGMetadata] = []
        self.test_metadata:  List[ECGMetadata] = []

        self.scp_statements_df: Optional[pd.DataFrame] = None
        self.statistics:        Optional[DatasetStatistics] = None
        self.data_roots:        Dict[DatasetSource, Path] = {}

        self.train_dataset: Optional[UnifiedECGDataset] = None
        self.val_dataset:   Optional[UnifiedECGDataset] = None
        self.test_dataset:  Optional[UnifiedECGDataset] = None

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        """Load datasets, compute statistics, and create train/val/test splits."""
        all_metadata:    List[ECGMetadata] = []
        all_scp: List[pd.DataFrame] = []

        all_scp: List[pd.DataFrame] = []

        # ── Load Brugada-HUCA ─────────────────────────────────────────────────
        if self.config.use_brugada and self.config.brugada_metadata_path:
            brugada_metadata, brugada_scp = load_metadata_unified(
                self.config.brugada_metadata_path,
                self.config.brugada_scp_statements_path,
                DatasetSource.BRUGADA_HUCA,
            )
            all_metadata.extend(brugada_metadata)
            all_scp.append(brugada_scp)
            self.data_roots[DatasetSource.BRUGADA_HUCA] = self.config.brugada_data_root
            print(f"Loaded {len(brugada_metadata)} Brugada-HUCA samples")

        # ── Load PTB-XL ───────────────────────────────────────────────────────
        if self.config.use_ptbxl and self.config.ptbxl_metadata_path:
            ptbxl_metadata, ptbxl_scp = load_metadata_unified(
                self.config.ptbxl_metadata_path,
                self.config.ptbxl_scp_statements_path,
                DatasetSource.PTB_XL,
            )
            # Optional downsampling
            if self.config.ptbxl_sampling_ratio < 1.0:
                random.seed(self.config.random_seed)
                k = int(len(ptbxl_metadata) * self.config.ptbxl_sampling_ratio)
                ptbxl_metadata = random.sample(ptbxl_metadata, k)
            all_metadata.extend(ptbxl_metadata)
            all_scp.append(ptbxl_scp)
            self.data_roots[DatasetSource.PTB_XL] = self.config.ptbxl_data_root
            print(f"Loaded {len(ptbxl_metadata)} PTB-XL samples")

        self.metadata_list = all_metadata

        # Concatenate SCP statements from all datasets, deduplicating by index,
        # so _build_subclass_order sees the full 23-class PTB-XL vocabulary + BRUG
        self.scp_statements_df = (
            pd.concat(all_scp).loc[~pd.concat(all_scp).index.duplicated(keep="first")]
            if all_scp else None
        )

        # ── Statistics ────────────────────────────────────────────────────────
        self.statistics = DatasetStatistics.from_metadata_list(all_metadata)

        print(f"\nDataset Statistics:")
        print(f"  Total:  {self.statistics.total_samples}")
        print(f"  NORM:   {self.statistics.normal_samples}")
        print(f"  MI:     {self.statistics.mi_samples}")
        print(f"  STTC:   {self.statistics.sttc_samples}")
        print(f"  CD:     {self.statistics.cd_samples}  (incl. Brugada positive)")
        print(f"  HYP:    {self.statistics.hyp_samples}")

        # ── Stratified train / val / test split ───────────────────────────────
        strat_labels = [
            1 if "BRUG" in m.diagnostic_subclass else 0
            for m in self.metadata_list
        ]

        train_val, self.test_metadata = train_test_split(
            self.metadata_list,
            test_size=self.config.test_split,
            stratify=strat_labels if self.config.stratified else None,
            random_state=self.config.random_seed,
        )
        train_val_labels = [
            1 if "BRUG" in m.diagnostic_subclass else 0
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
                # Augmentation config passed only for training — enables crop
                self.augmentation_config if self.config.augment_train else None,
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds,
                is_training=True,   # ← enables random cropping
            )
            self.val_dataset = UnifiedECGDataset(
                self.val_metadata,
                self.data_roots,
                self.scp_statements_df,
                None,               # No augmentation for val
                self.config.normalize,
                self.config.target_sampling_rate,
                self.config.target_length_seconds,
                is_training=False,  # ← fixed-length standardisation
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
                is_training=False,
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
        """Pad variable-length signals to longest in batch, then stack."""
        lengths = [s.signal.shape[1] for s in batch]
        t_max   = max(lengths)
        n_leads = batch[0].signal.shape[0]

        signals = torch.zeros(len(batch), n_leads, t_max)
        mask    = torch.zeros(len(batch), t_max, dtype=torch.bool)

        for i, sample in enumerate(batch):
            t = sample.signal.shape[1]
            signals[i, :, :t] = sample.signal
            mask[i, :t] = True

        return {
            'signal':  signals,
            'padding_mask': mask,
            'lengths': torch.tensor(lengths, dtype=torch.long),
            'labels': {
                'superclass': torch.stack([s.label_superclass for s in batch]),
                'subclass': torch.stack([s.label_subclass   for s in batch]),
            },
            'metadata': [s.original_metadata for s in batch],
        }
    
    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_class_weights(self) -> Dict:
        """Return class weights for loss functions."""
        if self.statistics is None:
            raise ValueError("Must call setup() first")
        return {
            "superclass_weights": self.statistics.superclass_weights,
        }
