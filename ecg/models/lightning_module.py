"""
PyTorch Lightning module for multi-task ECG disease detection.

Changes from the original lightning_module.py
----------------------------------------------
1.  EncoderUnfreezeCallback  — new callback that unfreezes the backbone
    after `freeze_encoder_epochs` epochs and optionally lowers the encoder LR.
    Add it to your trainer callbacks list in train.py.

2.  MultiTaskClassifier.configure_optimizers  — now uses a two-param-group
    AdamW when the model exposes freeze/unfreeze (i.e. HuBERTECGMultiTask):
        •  heads:   lr = learning_rate         (same as before)
        •  encoder: lr = encoder_lr            (typically 10× smaller)
    Falls back to single-group for the old ECGTransformerRoPE model.

Everything else (metrics, logging, loss, checkpoint monitor) is unchanged.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision,
    MetricCollection,
)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: unfreeze encoder after warm-up
# ─────────────────────────────────────────────────────────────────────────────

class EncoderUnfreezeCallback(Callback):
    """
    Unfreeze the pretrained encoder after `freeze_epochs` training epochs.

    Add this to your trainer callbacks:

        callbacks.append(EncoderUnfreezeCallback(freeze_epochs=10))

    After unfreezing, the encoder LR is set to `encoder_lr` in every
    param group whose name starts with "encoder".  The head LR is unchanged.

    If the model does not have a `freeze_encoder` / `unfreeze_encoder`
    attribute (e.g. the old ECGTransformerRoPE) this callback is a no-op.
    """

    def __init__(self, freeze_epochs: int = 10, encoder_lr: float = 1e-5):
        self.freeze_epochs = freeze_epochs
        self.encoder_lr    = encoder_lr
        self._unfrozen     = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._unfrozen:
            return
        if trainer.current_epoch < self.freeze_epochs:
            return

        model = pl_module.model
        if not hasattr(model, "unfreeze_encoder"):
            return

        model.unfreeze_encoder()
        self._unfrozen = True

        # Adjust encoder param group LR without restarting the optimizer
        for pg in trainer.optimizers[0].param_groups:
            if pg.get("name") == "encoder":
                pg["lr"] = self.encoder_lr
                print(f"  Encoder LR set to {self.encoder_lr}")


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskClassifier(pl.LightningModule):
    """
    Lightning module for multi-task ECG classification.

    Tasks:
    - Superclass classification (5 classes, multi-label)
    - Subclass classification (N classes, multi-label, incl. BRUG)

    Checkpoint monitor
    ------------------
    Logs `val_combined_auroc` each validation epoch:

        val_combined_auroc = sup_weight * val_sup_auroc_macro
                           + sub_weight * val_sub_auroc_macro

    Set `checkpoint.monitor: val_combined_auroc` in train.yaml.

    Differential learning rates (HuBERT-ECG)
    -----------------------------------------
    When the wrapped model exposes `freeze_encoder` / `unfreeze_encoder`
    (i.e. HuBERTECGMultiTask), two AdamW param groups are created:
        •  "heads"   — lr = learning_rate   (default 1e-4)
        •  "encoder" — lr = encoder_lr      (default 1e-5, only active
                                             after EncoderUnfreezeCallback fires)
    For the old ECGTransformerRoPE a single param group is used as before.
    """

    def __init__(
        self,
        model:          nn.Module,
        loss_fn:        nn.Module,
        learning_rate:  float = 1e-4,
        weight_decay:   float = 1e-2,
        # Encoder-specific LR used after unfreeze (HuBERT-ECG only)
        encoder_lr:     float = 1e-5,
        scheduler_params: Optional[Dict[str, Any]] = None,
        subclass_names:   Optional[List[str]] = None,
        combined_auroc_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.model         = model
        self.loss_fn       = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.encoder_lr    = encoder_lr
        self.scheduler_params = scheduler_params or {}

        # ── Class names ───────────────────────────────────────────────────────
        self.superclass_names = ["NORM", "MI", "STTC", "CD", "HYP"]
        self.subclass_names   = subclass_names or [
            "AMI", "BRUG", "CLBBB", "CRBBB", "ILBBB", "IMI", "IRBBB",
            "ISCA", "ISCI", "ISC_", "IVCD", "LAFB/LPFB", "LAO/LAE", "LMI",
            "LVH", "NST_", "PMI", "RAO/RAE", "RVH", "SEHYP", "STTC", "WPW",
            "_AVB",
        ]

        num_subclasses = len(self.subclass_names)
        self.brug_subclass_idx = (
            self.subclass_names.index("BRUG")
            if "BRUG" in self.subclass_names else None
        )

        # ── Combined AUROC monitor weights ────────────────────────────────────
        cw = combined_auroc_weights or {"superclass": 0.5, "subclass": 0.5}
        self.combined_sup_weight = cw.get("superclass", 0.5)
        self.combined_sub_weight = cw.get("subclass",   0.5)

        self.save_hyperparameters(ignore=["model", "loss_fn"])

        # ── Metrics ───────────────────────────────────────────────────────────
        self.train_metrics_superclass = self._create_multilabel_metrics(5, "train_sup_")
        self.val_metrics_superclass   = self._create_multilabel_metrics(5, "val_sup_")
        self.train_metrics_subclass   = self._create_multilabel_metrics(num_subclasses, "train_sub_")
        self.val_metrics_subclass     = self._create_multilabel_metrics(num_subclasses, "val_sub_")

    # ── Metric factory ────────────────────────────────────────────────────────

    def _create_multilabel_metrics(self, num_labels: int, prefix: str) -> MetricCollection:
        return MetricCollection({
            "auroc_macro":        AUROC(task="multilabel", num_labels=num_labels, average="macro"),
            "auroc_per_class":    AUROC(task="multilabel", num_labels=num_labels, average=None),
            "auprc_macro":        AveragePrecision(task="multilabel", num_labels=num_labels, average="macro"),
            "acc_macro":          Accuracy(task="multilabel", num_labels=num_labels, average="macro"),
            "acc_micro":          Accuracy(task="multilabel", num_labels=num_labels, average="micro"),
            "f1_macro":           F1Score(task="multilabel", num_labels=num_labels, average="macro"),
            "f1_per_class":       F1Score(task="multilabel", num_labels=num_labels, average=None),
            "precision_macro":    Precision(task="multilabel", num_labels=num_labels, average="macro"),
            "precision_per_class":Precision(task="multilabel", num_labels=num_labels, average=None),
            "recall_macro":       Recall(task="multilabel", num_labels=num_labels, average="macro"),
            "recall_per_class":   Recall(task="multilabel", num_labels=num_labels, average=None),
        }, prefix=prefix)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        signals = batch["signal"]
        labels  = batch["labels"]

        predictions = self(signals)
        losses      = self.loss_fn(predictions, labels)

        for name, val in losses.items():
            self.log(f"train_{name}_loss", val, on_step=False, on_epoch=True,
                     prog_bar=(name == "total"))

        sup_probs = torch.sigmoid(predictions["superclass"])
        sub_probs = torch.sigmoid(predictions["subclass"])
        self.train_metrics_superclass.update(sup_probs, labels["superclass"].int())
        self.train_metrics_subclass.update(sub_probs,   labels["subclass"].int())

        return losses["total"]

    def on_train_epoch_end(self):
        self._log_metric_collection(
            self.train_metrics_superclass, self.superclass_names, "train_sup_"
        )
        self._log_metric_collection(
            self.train_metrics_subclass, self.subclass_names, "train_sub_"
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        signals = batch["signal"]
        labels  = batch["labels"]

        predictions = self(signals)
        losses      = self.loss_fn(predictions, labels)

        for name, val in losses.items():
            self.log(f"val_{name}_loss", val, on_step=False, on_epoch=True,
                     prog_bar=(name == "total"))

        sup_probs = torch.sigmoid(predictions["superclass"])
        sub_probs = torch.sigmoid(predictions["subclass"])
        self.val_metrics_superclass.update(sup_probs, labels["superclass"].int())
        self.val_metrics_subclass.update(sub_probs,   labels["subclass"].int())

        return losses["total"]

    def on_validation_epoch_end(self):
        sup_auroc_macro = self._log_metric_collection(
            self.val_metrics_superclass, self.superclass_names, "val_sup_",
            prog_bar_key="val_sup_auroc_macro",
        )
        sub_auroc_macro = self._log_metric_collection(
            self.val_metrics_subclass, self.subclass_names, "val_sub_",
        )

        if sup_auroc_macro is not None and sub_auroc_macro is not None:
            combined = (
                self.combined_sup_weight * sup_auroc_macro
                + self.combined_sub_weight * sub_auroc_macro
            )
            self.log("val_combined_auroc", combined, prog_bar=True, sync_dist=True)

    # ── Test ─────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    # ── Logging helper ────────────────────────────────────────────────────────

    def _log_metric_collection(
        self,
        collection: MetricCollection,
        class_names: List[str],
        prefix:      str,
        prog_bar_key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Compute, log, and reset a MetricCollection.  Returns auroc_macro."""
        metrics = collection.compute()
        auroc_macro = None

        for metric_name, metric_value in metrics.items():
            if "per_class" in metric_name:
                base = metric_name.replace(prefix, "").replace("_per_class", "")
                for i, cls in enumerate(class_names):
                    self.log(f"{prefix}{base}_{cls}", metric_value[i], prog_bar=False)
            else:
                is_prog = (metric_name == prog_bar_key)
                self.log(metric_name, metric_value, prog_bar=is_prog)
                if metric_name.endswith("auroc_macro"):
                    auroc_macro = metric_value

        collection.reset()
        return auroc_macro

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        Two-group AdamW for HuBERT-ECG (encoder + heads at different LRs).
        Single-group AdamW for legacy ECGTransformerRoPE.
        """
        model = self.model
        has_pretrained_encoder = hasattr(model, "freeze_encoder")

        if has_pretrained_encoder:
            # Separate encoder params from head + norm params
            encoder_params = list(model.encoder.parameters())
            encoder_ids    = {id(p) for p in encoder_params}
            head_params    = [p for p in model.parameters() if id(p) not in encoder_ids]

            param_groups = [
                {"params": head_params,    "lr": self.learning_rate, "name": "heads"},
                {"params": encoder_params, "lr": self.encoder_lr,    "name": "encoder"},
            ]
            print(
                f"  Optimiser: two param groups — "
                f"heads lr={self.learning_rate}, encoder lr={self.encoder_lr}"
            )
        else:
            param_groups = [{"params": model.parameters(), "lr": self.learning_rate}]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        if not self.scheduler_params:
            return optimizer

        scheduler_type = self.scheduler_params.get("type", "cosine")

        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_epochs = self.scheduler_params.get("warmup_epochs", 10)
            total_epochs  = self.scheduler_params.get("t_max", 200)

            warmup   = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine   = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
            return [optimizer], [scheduler]

        elif scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_params.get("factor", 0.5),
                patience=self.scheduler_params.get("patience", 5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor":   "val_total_loss",
                    "interval":  "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
