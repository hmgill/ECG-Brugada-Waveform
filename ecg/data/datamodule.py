"""PyTorch Lightning DataModule for multi-dataset ECG classification."""

import ast
import random
from typing import Optional, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# iterstrat: multi-label stratification for Brugada-HUCA records.
# PTB-XL uses its built-in strat_fold (patient-aware, human-validated splits).
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    _ITERSTRAT_AVAILABLE = True
except ImportError:
    _ITERSTRAT_AVAILABLE = False

from .models import (
    ECGMetadata, DataConfig, AugmentationConfig, DatasetStatistics,
    DatasetSource, DiagnosticSuperclass
)
from .dataset import UnifiedECGDataset


# ── Superclass order (must match dataset.py) ──────────────────────────────────
_SUPERCLASS_ORDER = [
    DiagnosticSuperclass.NORM,
    DiagnosticSuperclass.MI,
    DiagnosticSuperclass.STTC,
    DiagnosticSuperclass.CD,
    DiagnosticSuperclass.HYP,
]
_SUPERCLASS_NAMES = [sc.value for sc in _SUPERCLASS_ORDER]


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
        # CHANGE 1: filter out uncertain annotations (0 < likelihood < 50).
        # likelihood=0  → unknown/ungraded but present; keep.
        # likelihood=100 → confirmed; keep.
        # 0 < likelihood < 50 → annotator uncertainty; skip.
        diagnostic_superclass: List[DiagnosticSuperclass] = []
        diagnostic_subclass:   List[str] = []

        for scp_code, likelihood in scp_codes.items():
            if 0 < likelihood < 50:
                continue

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


# ── CHANGE 2: stratification helpers ─────────────────────────────────────────

def _build_multilabel_matrix(metadata: List[ECGMetadata]) -> np.ndarray:
    """(N, 6) binary matrix: 5 superclasses + BRUG, for iterstrat."""
    cols = _SUPERCLASS_NAMES + ["BRUG"]
    mat = np.zeros((len(metadata), len(cols)), dtype=np.int8)
    for i, m in enumerate(metadata):
        for j, sc_name in enumerate(_SUPERCLASS_NAMES):
            if any(sc.value == sc_name for sc in m.diagnostic_superclass):
                mat[i, j] = 1
        if "BRUG" in m.diagnostic_subclass:
            mat[i, 5] = 1
    return mat


def _split_ptbxl_by_fold(
    metadata: List[ECGMetadata],
    val_folds: tuple = (9,),
    test_folds: tuple = (10,),
) -> tuple[List[ECGMetadata], List[ECGMetadata], List[ECGMetadata]]:
    """
    Split PTB-XL using the official strat_fold column.
    Folds 1-8 → train, fold 9 → val, fold 10 → test.
    All folds 9/10 records are human-validated (highest label quality).
    """
    train, val, test = [], [], []
    for m in metadata:
        f = m.strat_fold
        if f in test_folds:
            test.append(m)
        elif f in val_folds:
            val.append(m)
        else:
            train.append(m)
    return train, val, test


def _split_multilabel_stratified(
    metadata: List[ECGMetadata],
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[List[ECGMetadata], List[ECGMetadata], List[ECGMetadata]]:
    """
    Multi-label stratified split for datasets without pre-defined folds
    (i.e. Brugada-HUCA). Falls back to random split if iterstrat is absent.
    """
    n = len(metadata)

    if not _ITERSTRAT_AVAILABLE or n < 20:
        if not _ITERSTRAT_AVAILABLE:
            print(
                "  ⚠ iterstrat not installed — falling back to random split.\n"
                "  Install with: pip install iterative-stratification"
            )
        rng = random.Random(random_seed)
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test  = [metadata[i] for i in idxs[:n_test]]
        val   = [metadata[i] for i in idxs[n_test:n_test + n_val]]
        train = [metadata[i] for i in idxs[n_test + n_val:]]
        return train, val, test

    label_matrix = _build_multilabel_matrix(metadata)
    indices = np.arange(n)

    msss_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_seed
    )
    train_val_idx, test_idx = next(
        msss_test.split(indices.reshape(-1, 1), label_matrix)
    )

    adjusted_val = val_ratio / (1.0 - test_ratio)
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=adjusted_val, random_state=random_seed
    )
    train_rel_idx, val_rel_idx = next(
        msss_val.split(train_val_idx.reshape(-1, 1), label_matrix[train_val_idx])
    )

    return (
        [metadata[i] for i in train_val_idx[train_rel_idx]],
        [metadata[i] for i in train_val_idx[val_rel_idx]],
        [metadata[i] for i in test_idx],
    )


def _print_split_stats(name: str, metadata: List[ECGMetadata]) -> None:
    n = len(metadata)
    counts = {sc: sum(1 for m in metadata if any(s.value == sc for s in m.diagnostic_superclass))
              for sc in _SUPERCLASS_NAMES}
    counts["BRUG"] = sum(1 for m in metadata if "BRUG" in m.diagnostic_subclass)
    parts = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"  {name:6s}: {n:5d} samples  [{parts}]")


# ── DataModule ────────────────────────────────────────────────────────────────

class UnifiedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for multi-dataset ECG classification.

    Both datasets are loaded through the same PTB-XL-structured CSV format,
    producing a unified metadata list with superclass / subclass labels.
    The Brugada syndrome cases appear as CD superclass + BRUG subclass.

    Splitting strategy
    ------------------
    PTB-XL: uses the official strat_fold column (patient-disjoint).
      Folds 1-8 → train, fold 9 → val, fold 10 → test.
    Brugada-HUCA: multi-label stratified shuffle split via iterstrat,
      stratifying jointly over the 5 superclasses + BRUG.

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
        augmentation_config: Optional[AugmentationConfig] = None,
        ptbxl_val_folds:  tuple = (9,),
        ptbxl_test_folds: tuple = (10,),
    ):
        super().__init__()
        self.config               = config
        self.augmentation_config  = augmentation_config
        self.ptbxl_val_folds      = ptbxl_val_folds
        self.ptbxl_test_folds     = ptbxl_test_folds

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
        ptbxl_metadata:   List[ECGMetadata] = []
        brugada_metadata: List[ECGMetadata] = []
        all_scp: List[pd.DataFrame] = []

        # ── Load Brugada-HUCA ─────────────────────────────────────────────────
        if self.config.use_brugada and self.config.brugada_metadata_path:
            brugada_metadata, brugada_scp = load_metadata_unified(
                self.config.brugada_metadata_path,
                self.config.brugada_scp_statements_path,
                DatasetSource.BRUGADA_HUCA,
            )
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
            if self.config.ptbxl_sampling_ratio < 1.0:
                random.seed(self.config.random_seed)
                k = int(len(ptbxl_metadata) * self.config.ptbxl_sampling_ratio)
                ptbxl_metadata = random.sample(ptbxl_metadata, k)
            all_scp.append(ptbxl_scp)
            self.data_roots[DatasetSource.PTB_XL] = self.config.ptbxl_data_root
            print(f"Loaded {len(ptbxl_metadata)} PTB-XL samples")

        self.metadata_list = ptbxl_metadata + brugada_metadata

        # Merged SCP statements — full vocabulary for subclass order building
        self.scp_statements_df = (
            pd.concat(all_scp).loc[~pd.concat(all_scp).index.duplicated(keep="first")]
            if all_scp else None
        )

        # ── Statistics ────────────────────────────────────────────────────────
        self.statistics = DatasetStatistics.from_metadata_list(self.metadata_list)

        print(f"\nDataset Statistics:")
        print(f"  Total:  {self.statistics.total_samples}")
        print(f"  NORM:   {self.statistics.normal_samples}")
        print(f"  MI:     {self.statistics.mi_samples}")
        print(f"  STTC:   {self.statistics.sttc_samples}")
        print(f"  CD:     {self.statistics.cd_samples}  (incl. Brugada positive)")
        print(f"  HYP:    {self.statistics.hyp_samples}")

        # ── CHANGE 2: stratified splits ───────────────────────────────────────
        if self.config.stratified:
            ptbxl_train, ptbxl_val, ptbxl_test = [], [], []
            brug_train,  brug_val,  brug_test  = [], [], []

            if ptbxl_metadata:
                ptbxl_train, ptbxl_val, ptbxl_test = _split_ptbxl_by_fold(
                    ptbxl_metadata,
                    val_folds=self.ptbxl_val_folds,
                    test_folds=self.ptbxl_test_folds,
                )
                print(f"\nPTB-XL splits (strat_fold):")
                _print_split_stats("train", ptbxl_train)
                _print_split_stats("val",   ptbxl_val)
                _print_split_stats("test",  ptbxl_test)

            if brugada_metadata:
                brug_train, brug_val, brug_test = _split_multilabel_stratified(
                    brugada_metadata,
                    val_ratio=self.config.val_split,
                    test_ratio=self.config.test_split,
                    random_seed=self.config.random_seed,
                )
                print(f"\nBrugada-HUCA splits (multi-label stratified):")
                _print_split_stats("train", brug_train)
                _print_split_stats("val",   brug_val)
                _print_split_stats("test",  brug_test)

            self.train_metadata = ptbxl_train + brug_train
            self.val_metadata   = ptbxl_val   + brug_val
            self.test_metadata  = ptbxl_test  + brug_test

        else:
            all_meta = self.metadata_list[:]
            random.seed(self.config.random_seed)
            random.shuffle(all_meta)
            n = len(all_meta)
            n_test = int(n * self.config.test_split)
            n_val  = int(n * self.config.val_split)
            self.test_metadata  = all_meta[:n_test]
            self.val_metadata   = all_meta[n_test:n_test + n_val]
            self.train_metadata = all_meta[n_test + n_val:]

        print(f"\nFinal Data Splits:")
        _print_split_stats("train", self.train_metadata)
        _print_split_stats("val",   self.val_metadata)
        _print_split_stats("test",  self.test_metadata)

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
                'subclass':   torch.stack([s.label_subclass   for s in batch]),
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
