"""
Standalone inference για το SplitLawin checkpoint.

Φορτώνει pre-trained ckpt, τρέχει σε test set, υπολογίζει per-class metrics
και (optional) σώζει visualizations.

Usage:
    # Run στο πλήρες test set + report metrics
    python scripts/inference.py \
        --ckpt experiments/.../ckpt_best.pth \
        --dataset-root /workspace/datasets/weedsgalore/weedsgalore-dataset

    # Επιπλέον σώσε visualizations για κάθε test image
    python scripts/inference.py \
        --ckpt .../ckpt_best.pth \
        --dataset-root .../weedsgalore-dataset \
        --save-viz outputs/inf_viz/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# NumPy 2.0 shim για super-gradients
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan


def load_model(ckpt_path: str, device: str = "cuda"):
    """Φορτώνει το SplitLawin model + ckpt weights."""
    from wd.models import SplitLawin
    from super_gradients.training.utils import HpmStruct

    arch_params = HpmStruct(
        backbone="MiT-B0",
        backbone_pretrained=False,  # δεν χρειάζεται — φορτώνουμε δικά μας weights
        main_channels=3,
        main_pretrained=['R', 'G', 'B'],
        side_pretrained='G',
        fusion_type='squeeze_excite',
        num_classes=3,
        input_channels=5,
        side_channels=2,
    )
    model = SplitLawin(arch_params)

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # ckpt περιέχει 'net', 'optimizer_state_dict', etc. Παίρνουμε μόνο το net.
    if isinstance(state, dict):
        weights = state.get("net", state.get("state_dict", state))
    else:
        weights = state
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    print(f"  Loaded successfully.")

    model = model.to(device).eval()
    return model


def get_test_loader(dataset_root: str, batch_size: int = 2):
    """Φτιάχνει το test DataLoader μέσω του WeedsGaloreDatasetInterface."""
    from wd.data.weedsgalore import WeedsGaloreDatasetInterface
    from easydict import EasyDict

    params = EasyDict({
        "root": dataset_root,
        "tile": False,
        "hor_flip": False,  # eval mode
        "ver_flip": False,
        "channels": ['R', 'G', 'B', 'NIR', 'RE'],
        "batch_size": batch_size,
        "val_batch_size": batch_size,
        "test_batch_size": batch_size,
        "num_workers": 0,
        "num_classes": 3,
        "return_path": True,
        "size": "same",
        "crop_size": "same",
    })
    interface = WeedsGaloreDatasetInterface(params)
    interface.build_data_loaders(batch_size_factor=1)
    return interface.test_loader


def evaluate(model, loader, device: str = "cuda", num_classes: int = 3):
    """Forward pass σε test set + aggregated CM."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_images = 0

    with torch.no_grad():
        for batch in loader:
            # batch μπορεί να είναι (img, mask) ή (img, mask, path)
            if len(batch) == 3:
                img, mask, _ = batch
            else:
                img, mask = batch
            img = img.to(device)
            mask = mask.to(device)

            logits = model(img)
            # Αν το model επιστρέφει tuple/dict, παίρνουμε το main output
            if isinstance(logits, tuple):
                logits = logits[0]
            elif hasattr(logits, "main"):
                logits = logits.main
            elif isinstance(logits, dict):
                logits = logits["out"]

            pred = logits.argmax(dim=1)  # [B, H, W]

            # Update CM
            for t in range(num_classes):
                for p in range(num_classes):
                    cm[t, p] += int(((mask == t) & (pred == p)).sum().item())
            n_images += img.size(0)

    return cm, n_images


def report_metrics(cm: np.ndarray, class_names: list[str]):
    """Per-class IoU/F1 από aggregated CM + macro averages."""
    n = cm.shape[0]
    print(f"\n=== Aggregated Confusion Matrix ===")
    header = "             " + "".join(f"  pred_{name:<8}" for name in class_names)
    print(header)
    for i in range(n):
        row = " ".join(f"{cm[i, j]:>13}" for j in range(n))
        print(f"true_{class_names[i]:<8}  {row}")

    print(f"\n=== Per-class metrics ===")
    print(f"{'class':<14} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    ious, f1s = [], []
    for c in range(n):
        tp = cm[c, c]
        fp = sum(cm[i, c] for i in range(n) if i != c)
        fn = sum(cm[c, j] for j in range(n) if j != c)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou); f1s.append(f1)
        print(f"{class_names[c]:<14} {iou:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")

    print(f"\n=== Macro ===")
    print(f"  mIoU      = {sum(ious) / n:.4f}")
    print(f"  macro F1  = {sum(f1s) / n:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to ckpt_best.pth")
    p.add_argument("--dataset-root", required=True, help="WeedsGalore dataset root")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-viz", type=Path, help="(optional) folder για viz PNGs")
    args = p.parse_args()

    print(f"Device: {args.device}")
    model = load_model(args.ckpt, args.device)
    loader = get_test_loader(args.dataset_root, args.batch_size)
    print(f"Test loader: {len(loader)} batches")

    cm, n = evaluate(model, loader, args.device)
    print(f"\nProcessed {n} test images")
    report_metrics(cm, ["background", "crop", "weed"])


if __name__ == "__main__":
    main()
