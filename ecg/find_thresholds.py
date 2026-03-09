#!/usr/bin/env python3
"""
Per-class threshold calibration for the multi-task ECG classifier.

Loads the best checkpoint, runs the validation set, and finds optimal
per-class thresholds for both superclass and subclass heads.

For BRUG specifically, the threshold is chosen to satisfy a minimum
sensitivity (recall) constraint rather than maximising F1, reflecting
the clinical priority of not missing Brugada cases.

Usage:
    python eval/find_thresholds.py \\
        --checkpoint checkpoints/run_XXXXXXXX/multitask-epoch=XX-*.ckpt \\
        --config config/train.yaml \\
        --output thresholds.json

    # Adjust minimum BRUG sensitivity (default 0.90)
    python eval/find_thresholds.py \\
        --checkpoint checkpoints/run_XXXXXXXX/multitask-epoch=XX-*.ckpt \\
        --config config/train.yaml \\
        --min-brug-sensitivity 0.95 \\
        --output thresholds.json
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless/HPC
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.models import DataConfig, AugmentationConfig
from data.datamodule import UnifiedDataModule
from models.ecg_transformer import create_ecg_transformer_rope
from models.losses import get_loss_function
from models.lightning_module import MultiTaskClassifier
from models.ecg_transformer import create_ecg_transformer_rope
from models.hubert_ecg_multitask import create_hubert_multitask
from models.losses import get_loss_function
from models.lightning_module import MultiTaskClassifier


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find optimal per-class thresholds from a trained checkpoint."
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the .ckpt file (best checkpoint)."
    )
    p.add_argument(
        "--config", type=Path, default=Path("config/train.yaml"),
        help="Training config YAML (same one used for training)."
    )
    p.add_argument(
        "--output", type=Path, default=Path("thresholds.json"),
        help="Where to save the per-class threshold JSON."
    )
    p.add_argument(
        "--plots-dir", type=Path, default=Path("threshold_plots"),
        help="Directory for diagnostic plots."
    )
    p.add_argument(
        "--min-brug-sensitivity", type=float, default=0.90,
        help="Minimum sensitivity (recall) to guarantee for BRUG (default 0.90)."
    )
    p.add_argument(
        "--split", choices=["val", "test"], default="val",
        help="Which split to calibrate on. Always use val — never test."
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'cpu', 'cuda', 'cuda:0', etc. Auto-detected if omitted."
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config + data setup  (mirrors train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_datamodule(cfg: dict) -> UnifiedDataModule:
    """
    Construct DataConfig / AugmentationConfig and return a set-up datamodule.
    Mirrors the construction in agent/scripts/train.py line-for-line so that
    the same splits and subclass_order are produced.
    """
    dc = cfg["data"]

    data_config = DataConfig(
        use_brugada=dc["use_brugada"],
        brugada_metadata_path=Path(dc["brugada_metadata_path"]),
        brugada_scp_statements_path=Path(dc["brugada_scp_statements_path"]),
        brugada_data_root=Path(dc["brugada_data_root"]),
        use_ptbxl=dc["use_ptbxl"],
        ptbxl_metadata_path=Path(dc["ptbxl_metadata_path"]),
        ptbxl_data_root=Path(dc["ptbxl_data_root"]),
        ptbxl_scp_statements_path=Path(dc["ptbxl_scp_statements_path"]),
        ptbxl_sampling_ratio=dc["ptbxl_sampling_ratio"],
        target_sampling_rate=dc["target_sampling_rate"],
        target_length_seconds=dc["target_length_seconds"],
        train_split=dc["train_split"],
        val_split=dc["val_split"],
        test_split=dc["test_split"],
        stratified=dc["stratified"],
        batch_size=dc["batch_size"],
        num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"],
        normalize=dc["normalize"],
        normalization_method=dc["normalization_method"],
        augment_train=False,   # No augmentation during eval
        augment_val=False,
        random_seed=dc["random_seed"],
    )

    aug_params = dc.get("augmentation", {})
    augmentation_config = AugmentationConfig(
        amplitude_scale_range=tuple(aug_params["amplitude_scale_range"]),
        amplitude_scale_prob=0.0,
        noise_std=aug_params["noise_std"],
        noise_prob=0.0,
        baseline_wander_amplitude=aug_params["baseline_wander_amplitude"],
        baseline_wander_frequency=tuple(aug_params["baseline_wander_frequency"]),
        baseline_wander_prob=0.0,
        time_warp_sigma=aug_params["time_warp_sigma"],
        time_warp_knots=aug_params["time_warp_knots"],
        time_warp_prob=0.0,
        lead_scale_range=tuple(aug_params["lead_scale_range"]),
        lead_scale_prob=0.0,
        lead_masking_prob=0.0,
        lead_masking_max_leads=aug_params.get("lead_masking_max_leads", 6),
    )

    datamodule = UnifiedDataModule(
        config=data_config,
        augmentation_config=augmentation_config,
    )
    datamodule.setup(stage="fit")   # populates train_dataset + val_dataset
    return datamodule


def build_model_and_load_checkpoint(cfg, checkpoint_path, subclass_names, device):
    mc = cfg["model"]
    
    architecture = mc.get("architecture", "rope_transformer")
    if architecture == "hubert_ecg":
        base_model = create_hubert_multitask(
            num_superclasses=mc.get("num_superclasses", 5),
            num_subclasses=len(subclass_names),
            hubert_size=mc.get("hubert_size", "base"),
            dropout=mc.get("dropout", 0.1),
            window_seconds=mc.get("window_seconds", 5.0),
            pool_mode=mc.get("pool_mode", "mean"),
            freeze_on_init=False,  # no freezing needed for inference
        )
    else:
        base_model = create_ecg_transformer_rope(
            model_size=mc["size"],
            in_channels=mc["in_channels"],
            num_superclasses=mc["num_superclasses"],
            num_subclasses=len(subclass_names),
            dropout=mc["dropout"],
        )
    # Build a dummy loss — weights aren't used during inference, only the
    # model weights matter. We match the type used at training so that the
    # checkpoint loads without warnings.
    loss_cfg = cfg["loss"]
    loss_type = loss_cfg["type"]
    if loss_type == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        loss_fn = get_loss_function(
            loss_type,
            task_weights=loss_cfg.get("task_weights", {}),
            alpha_superclass=focal_cfg.get("alpha_superclass", 0.75),
            alpha_subclass=focal_cfg.get("alpha_subclass", 0.25),
            gamma=focal_cfg.get("gamma", 2.0),
        )
    elif loss_type == "weighted_bce":
        # Weights don't matter for inference; use uniform weights
        loss_fn = get_loss_function(
            loss_type,
            task_weights=loss_cfg.get("task_weights", {}),
            superclass_weights=torch.ones(mc["num_superclasses"]),
        )
    else:
        loss_fn = get_loss_function(
            loss_type,
            task_weights=loss_cfg.get("task_weights", {}),
        )

    lightning_model = MultiTaskClassifier.load_from_checkpoint(
        checkpoint_path,
        model=base_model,
        loss_fn=loss_fn,
        subclass_names=subclass_names,
        map_location=device,
        strict=False,
    )
    lightning_model.eval()
    return lightning_model

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: MultiTaskClassifier,
    dataloader,
    device: torch.device,
) -> tuple:
    """
    Run the full dataloader through the model.

    Returns:
        sup_probs  : (N, 5)  — superclass sigmoid probabilities
        sup_labels : (N, 5)  — superclass ground-truth multi-hot
        sub_probs  : (N, C)  — subclass sigmoid probabilities
        sub_labels : (N, C)  — subclass ground-truth multi-hot
    """
    model.to(device)
    model.eval()

    sup_probs_list, sup_labels_list = [], []
    sub_probs_list, sub_labels_list = [], []

    n_batches = 0
    for batch in dataloader:
        signals    = batch["signal"].to(device)
        sup_labels = batch["labels"]["superclass"]
        sub_labels = batch["labels"]["subclass"]

        preds     = model(signals)
        sup_probs = torch.sigmoid(preds["superclass"]).cpu()
        sub_probs = torch.sigmoid(preds["subclass"]).cpu()

        sup_probs_list.append(sup_probs)
        sup_labels_list.append(sup_labels)
        sub_probs_list.append(sub_probs)
        sub_labels_list.append(sub_labels)
        n_batches += 1

    print(f"  Processed {n_batches} batches")

    return (
        torch.cat(sup_probs_list).numpy(),
        torch.cat(sup_labels_list).numpy(),
        torch.cat(sub_probs_list).numpy(),
        torch.cat(sub_labels_list).numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Threshold search
# ─────────────────────────────────────────────────────────────────────────────

def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return the threshold on the precision-recall curve that maximises F1."""
    prec, rec, thresh = precision_recall_curve(labels, probs)
    # precision_recall_curve appends a final sentinel point with no threshold
    denom = prec[:-1] + rec[:-1]
    f1 = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / denom, 0.0)
    if len(f1) == 0:
        return 0.5
    return float(thresh[np.argmax(f1)])


def sensitivity_constrained_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    min_sensitivity: float = 0.90,
) -> tuple:
    """
    Find the highest threshold that still achieves >= min_sensitivity.

    Returns (threshold, achieved_sensitivity, achieved_specificity).
    Falls back to the threshold giving maximum sensitivity if the constraint
    cannot be met anywhere on the ROC curve.
    """
    fpr, tpr, thresh = roc_curve(labels, probs)

    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        # Constraint cannot be satisfied — take maximum sensitivity point
        best_idx = np.argmax(tpr)
    else:
        # Among valid points, pick the one with highest specificity (lowest FPR)
        best_idx = valid[np.argmin(fpr[valid])]

    opt_thresh    = float(thresh[best_idx])
    achieved_sens = float(tpr[best_idx])
    achieved_spec = float(1.0 - fpr[best_idx])

    return opt_thresh, achieved_sens, achieved_spec


def find_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    min_brug_sensitivity: float = 0.90,
    priority_classes: set = None,
) -> tuple:
    """
    Find per-class optimal thresholds.

    Priority classes (default: BRUG) use a sensitivity-constrained strategy.
    All other classes maximise F1.

    Returns:
        thresholds : {class_name: float}
        metrics    : {class_name: dict}
    """
    if priority_classes is None:
        priority_classes = {"BRUG"}

    thresholds = {}
    metrics    = {}

    for i, name in enumerate(class_names):
        p = probs[:, i]
        l = labels[:, i].astype(int)
        n_pos = int(l.sum())

        if n_pos == 0:
            print(f"  [{name}] No positive samples in split — default threshold=0.5")
            thresholds[name] = 0.5
            metrics[name] = {
                "auroc": None, "auprc": None,
                "opt_threshold": 0.5,
                "sensitivity": None, "specificity": None,
                "f1_at_threshold": None,
                "n_positive": 0,
            }
            continue

        auroc = float(roc_auc_score(l, p))
        auprc = float(average_precision_score(l, p))

        if name in priority_classes:
            opt_t, sens, spec = sensitivity_constrained_threshold(
                p, l, min_sensitivity=min_brug_sensitivity
            )
            preds_at_t = (p >= opt_t).astype(int)
            f1_at_t    = float(f1_score(l, preds_at_t, zero_division=0))

            print(
                f"  [{name}] AUROC={auroc:.4f} | AUPRC={auprc:.4f} | "
                f"Threshold={opt_t:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | "
                f"F1={f1_at_t:.4f} | n_pos={n_pos}"
            )
        else:
            opt_t      = best_f1_threshold(p, l)
            preds_at_t = (p >= opt_t).astype(int)
            f1_at_t    = float(f1_score(l, preds_at_t, zero_division=0))

            # Compute sens/spec at the chosen threshold for the summary table
            fpr_arr, tpr_arr, thresh_arr = roc_curve(l, p)
            idx  = np.argmin(np.abs(thresh_arr - opt_t)) if len(thresh_arr) else 0
            sens = float(tpr_arr[min(idx, len(tpr_arr) - 1)])
            spec = float(1.0 - fpr_arr[min(idx, len(fpr_arr) - 1)])

            print(
                f"  [{name}] AUROC={auroc:.4f} | AUPRC={auprc:.4f} | "
                f"Threshold={opt_t:.4f} | F1={f1_at_t:.4f} | n_pos={n_pos}"
            )

        thresholds[name] = opt_t
        metrics[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "opt_threshold": opt_t,
            "sensitivity": sens,
            "specificity": spec,
            "f1_at_threshold": f1_at_t,
            "n_positive": n_pos,
        }

    return thresholds, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_brug_distribution(
    probs: np.ndarray,
    labels: np.ndarray,
    opt_threshold: float,
    save_path: Path,
) -> None:
    """Probability density plot for BRUG positives vs negatives + ROC curve."""
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: histogram ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(neg_probs, bins=60, alpha=0.6, color="steelblue",
            label=f"Normal  (n={len(neg_probs)})", density=True)
    ax.hist(pos_probs, bins=max(10, len(pos_probs) // 2), alpha=0.7, color="crimson",
            label=f"Brugada (n={len(pos_probs)})", density=True)
    ax.axvline(opt_threshold, color="black", lw=2, linestyle="--",
               label=f"Optimal  threshold={opt_threshold:.3f}")
    ax.axvline(0.3, color="gray", lw=1, linestyle=":",
               label="Default  threshold=0.30")
    ax.set_xlabel("BRUG Predicted Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("BRUG Probability Distribution (Validation Set)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    # ── Right: ROC curve ──────────────────────────────────────────────────────
    ax = axes[1]
    fpr, tpr, thresh = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC (AUC={auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    # Mark operating point
    idx = int(np.argmin(np.abs(thresh - opt_threshold)))
    ax.scatter(fpr[idx], tpr[idx], color="black", zorder=5, s=80,
               label=f"Op. point  Sens={tpr[idx]:.3f}, Spec={1-fpr[idx]:.3f}")

    ax.set_xlabel("1 - Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title("BRUG ROC Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_precision_recall_brug(
    probs: np.ndarray,
    labels: np.ndarray,
    opt_threshold: float,
    save_path: Path,
) -> None:
    """Precision-Recall curve for BRUG with operating point marked."""
    prec, rec, thresh = precision_recall_curve(labels, probs)
    aurpc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="crimson", lw=2, label=f"PR curve (AUPRC={aurpc:.4f})")

    baseline = labels.mean()
    ax.axhline(baseline, color="gray", linestyle="--", lw=1,
               label=f"No-skill baseline ({baseline:.4f})")

    # Mark operating point (sentinel at end of thresh has no corresponding point)
    if len(thresh) > 0:
        idx = int(np.argmin(np.abs(thresh - opt_threshold)))
        ax.scatter(rec[idx], prec[idx], color="black", zorder=5, s=80,
                   label=f"Op. point  Prec={prec[idx]:.3f}, Rec={rec[idx]:.3f}")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("BRUG Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_threshold_summary(
    metrics: dict,
    class_names: list,
    save_path: Path,
    title_prefix: str = "",
) -> None:
    """Bar chart: AUROC and F1-at-threshold for every class."""
    names      = [n for n in class_names if metrics[n]["auroc"] is not None]
    aurocs     = [metrics[n]["auroc"]           for n in names]
    f1s        = [metrics[n]["f1_at_threshold"] for n in names]
    thresholds = [metrics[n]["opt_threshold"]   for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 6))
    ax.bar(x - width / 2, aurocs, width, label="AUROC",
           color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1 @ threshold",
                   color="darkorange", alpha=0.8)

    # Annotate thresholds above F1 bars
    for bar, t in zip(bars2, thresholds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{t:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{title_prefix}AUROC vs F1 at Optimal Threshold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(metrics: dict, class_names: list, title: str) -> None:
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")
    header = (
        f"{'Class':<20} {'AUROC':>7} {'AUPRC':>7} {'Threshold':>10} "
        f"{'Sensitivity':>12} {'Specificity':>12} {'F1':>7} {'N+':>6}"
    )
    print(header)
    print("-" * 88)
    for name in class_names:
        m = metrics[name]
        if m["auroc"] is None:
            print(
                f"  {name:<18} {'—':>7} {'—':>7} {'—':>10} "
                f"{'—':>12} {'—':>12} {'—':>7} {m['n_positive']:>6}"
            )
        else:
            print(
                f"  {name:<18} "
                f"{m['auroc']:>7.4f} "
                f"{m['auprc']:>7.4f} "
                f"{m['opt_threshold']:>10.4f} "
                f"{m['sensitivity']:>12.4f} "
                f"{m['specificity']:>12.4f} "
                f"{m['f1_at_threshold']:>7.4f} "
                f"{m['n_positive']:>6}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading config...")
    cfg = load_yaml(args.config)
    print(f"  Config: {args.config}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[2/5] Setting up datamodule...")
    datamodule = build_datamodule(cfg)

    # subclass_order lives on the dataset, not the datamodule itself
    subclass_names: list = datamodule.train_dataset.subclass_order
    superclass_names: list = ["NORM", "MI", "STTC", "CD", "HYP"]

    print(f"  Subclasses ({len(subclass_names)}): {subclass_names}")

    if args.split == "val":
        dataloader = datamodule.val_dataloader()
        n_samples  = len(datamodule.val_metadata)
        print(f"  Calibrating on VAL set ({n_samples} samples)")
    else:
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()
        n_samples  = len(datamodule.test_metadata)
        print(f"  Calibrating on TEST set ({n_samples} samples)")
        print("  ⚠ Warning: calibrating on test set may inflate reported performance.")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Loading model from checkpoint...")
    print(f"  Checkpoint: {args.checkpoint}")
    lightning_model = build_model_and_load_checkpoint(
        cfg, args.checkpoint, subclass_names, device
    )
    print("  ✓ Checkpoint loaded successfully")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\n[4/5] Running inference...")
    sup_probs, sup_labels, sub_probs, sub_labels = collect_predictions(
        lightning_model, dataloader, device
    )
    print(f"  ✓ Collected predictions: {sup_probs.shape[0]} samples")

    # ── Threshold search ──────────────────────────────────────────────────────
    print("\n[5/5] Finding optimal thresholds...")
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n  — Superclass —")
    sup_thresholds, sup_metrics = find_thresholds(
        sup_probs, sup_labels, superclass_names,
        min_brug_sensitivity=args.min_brug_sensitivity,
        priority_classes=set(),   # No priority constraint at superclass level
    )

    print("\n  — Subclass —")
    sub_thresholds, sub_metrics = find_thresholds(
        sub_probs, sub_labels, subclass_names,
        min_brug_sensitivity=args.min_brug_sensitivity,
        priority_classes={"BRUG"},
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")

    if "BRUG" in subclass_names:
        brug_idx    = subclass_names.index("BRUG")
        brug_probs  = sub_probs[:, brug_idx]
        brug_labels = sub_labels[:, brug_idx].astype(int)
        brug_thresh = sub_thresholds["BRUG"]

        if brug_labels.sum() > 0:
            plot_brug_distribution(
                brug_probs, brug_labels, brug_thresh,
                save_path=args.plots_dir / "brug_distribution.png",
            )
            plot_precision_recall_brug(
                brug_probs, brug_labels, brug_thresh,
                save_path=args.plots_dir / "brug_precision_recall.png",
            )
        else:
            print("  ⚠ No BRUG positives in this split — skipping BRUG plots")

    plot_threshold_summary(
        sub_metrics, subclass_names,
        save_path=args.plots_dir / "subclass_threshold_summary.png",
        title_prefix="Subclass — ",
    )
    plot_threshold_summary(
        sup_metrics, superclass_names,
        save_path=args.plots_dir / "superclass_threshold_summary.png",
        title_prefix="Superclass — ",
    )

    # ── Console tables ────────────────────────────────────────────────────────
    print_summary_table(sup_metrics, superclass_names, "SUPERCLASS THRESHOLDS")
    print_summary_table(sub_metrics, subclass_names,   "SUBCLASS THRESHOLDS")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "checkpoint": str(args.checkpoint),
        "calibration_split": args.split,
        "min_brug_sensitivity": args.min_brug_sensitivity,
        "superclass": {
            "thresholds": sup_thresholds,
            "metrics": sup_metrics,
        },
        "subclass": {
            "thresholds": sub_thresholds,
            "metrics": sub_metrics,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Thresholds saved to: {args.output}")
    print(f"✓ Plots saved to:      {args.plots_dir}/")

    # ── BRUG highlight ────────────────────────────────────────────────────────
    if "BRUG" in sub_metrics and sub_metrics["BRUG"]["auroc"] is not None:
        bm = sub_metrics["BRUG"]
        print(f"\n{'=' * 50}")
        print("  BRUG SUMMARY")
        print(f"{'=' * 50}")
        print(f"  AUROC:             {bm['auroc']:.4f}")
        print(f"  AUPRC:             {bm['auprc']:.4f}")
        print(f"  Optimal threshold: {bm['opt_threshold']:.4f}  (was 0.3000)")
        print(f"  Sensitivity:       {bm['sensitivity']:.4f}")
        print(f"  Specificity:       {bm['specificity']:.4f}")
        print(f"  F1 @ threshold:    {bm['f1_at_threshold']:.4f}")
        print(f"  N positives (val): {bm['n_positive']}")
        print(f"{'=' * 50}")
        print(
            "\n  To apply these thresholds at inference, load thresholds.json\n"
            "  and compare predicted probabilities against per-class thresholds\n"
            "  instead of the default 0.3.\n"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
