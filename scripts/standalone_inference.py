"""
Standalone inference για WeedsGalore SplitLawin checkpoints.

Bypass-άρει τελείως το ezdl resume mechanism. Φορτώνει το ckpt απευθείας
με το ίδιο model instantiation path που χρησιμοποιεί το training framework
(super-gradients models.get + wd.models registry).

Usage:
    # Inference + test metrics (matches wandb test phase output)
    python scripts/standalone_inference.py \\
        --ckpt experiments/.../ckpt_best.pth \\
        --yaml params/WeedsGalore/Best_CIR_B_RE_FT_Lovasz.yaml \\
        --dataset-root /workspace/datasets/weedsgalore/weedsgalore-dataset

    # + test-time augmentation (h-flip + v-flip averaging)
    python scripts/standalone_inference.py \\
        --ckpt .../ckpt_best.pth \\
        --yaml .../config.yaml \\
        --dataset-root .../weedsgalore-dataset \\
        --tta

Outputs:
  - stdout: per-class IoU/F1, macro mIoU/F1
  - {ckpt_dir}/standalone_metrics.json: full results
  - (optional with --save-viz) PNG visualizations per test image
"""
import argparse
import json
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

# Repo root στο sys.path ώστε να βλέπει το `wd/` και `ezdl/`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# NumPy 2.0 compat shim για super-gradients
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import torch
import torch.nn.functional as F
import yaml


# ────────────────────────────────────────────────────────────────────────────
# YAML / params helpers
# ────────────────────────────────────────────────────────────────────────────

def _unwrap_grid_lists(d):
    """YAML grids wrap single values σε lists. Unwrap recursively."""
    if isinstance(d, dict):
        return {k: _unwrap_grid_lists(v) for k, v in d.items()}
    if isinstance(d, list) and len(d) == 1:
        return _unwrap_grid_lists(d[0])
    if isinstance(d, list):
        return [_unwrap_grid_lists(x) for x in d]
    return d


def load_yaml_params(yaml_path: Path) -> dict:
    """Load training YAML και unwrap grid format."""
    with yaml_path.open() as f:
        raw = yaml.safe_load(f)
    # YAML structure: top-level 'parameters' key contains grid-format
    params = raw.get('parameters', raw)
    return _unwrap_grid_lists(params)


def count_input_channels(channels_spec):
    """Πανομοιότυπο logic με ez_trainer._load_model."""
    if isinstance(channels_spec, str):
        return 3 if channels_spec == 'CIR' else 1
    return sum(3 if c == 'CIR' else 1 for c in channels_spec)


# ────────────────────────────────────────────────────────────────────────────
# Model instantiation (matches ez_trainer._load_model exactly)
# ────────────────────────────────────────────────────────────────────────────

def build_model_from_yaml(params: dict, device: str = "cuda"):
    """Replicate ez_trainer._load_model logic to produce identical architecture."""
    from ezdl.utils.utilities import instantiate_class
    from super_gradients.training import models

    model_params = params["model"]
    channels_spec = params["dataset"]["channels"]
    input_channels = count_input_channels(channels_spec)
    output_channels = params["dataset"]["num_classes"]

    arch_params = {
        "input_channels": input_channels,
        "output_channels": output_channels,
        "in_channels": input_channels,
        "out_channels": output_channels,
        "num_classes": output_channels,
        **model_params["params"],
    }
    print(f"[build_model] arch_params keys: {list(arch_params.keys())}")
    print(f"[build_model] input_channels={input_channels}, output_channels={output_channels}")

    # Same instantiation cascade as ez_trainer._load_model
    try:
        model = instantiate_class(model_params["name"], arch_params)
        print(f"[build_model] instantiated via instantiate_class('{model_params['name']}')")
    except (AttributeError, ValueError) as ex:
        print(f"[build_model] instantiate_class failed: {ex}, trying models.get()")
        try:
            model = models.get(model_name=model_params["name"], arch_params=arch_params)
        except Exception as ex2:
            raise RuntimeError(
                f"Could not build model {model_params['name']}: {ex2}"
            ) from ex2
        print(f"[build_model] instantiated via models.get('{model_params['name']}')")

    model = model.to(device)
    return model, arch_params


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (with prefix stripping + diagnostics)
# ────────────────────────────────────────────────────────────────────────────

def strip_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    """Recursively strip 'module.' prefix from DataParallel-saved state dicts."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        # repeatedly strip 'module.' (might be nested)
        while k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    return new_sd


def load_checkpoint(ckpt_path: Path, model: torch.nn.Module, device: str = "cuda"):
    """Load ckpt weights into model, με όλα τα κλασικά prefix issues handled."""
    print(f"\n[load_ckpt] Loading: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    # super-gradients saves a dict with multiple keys ('net', 'optimizer_state_dict', etc)
    if isinstance(raw, dict) and "net" in raw:
        sd = raw["net"]
        print(f"[load_ckpt] Using 'net' key from saved dict (other keys: "
              f"{[k for k in raw.keys() if k != 'net']})")
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        print(f"[load_ckpt] Using 'state_dict' key")
    else:
        sd = raw

    # Strip DataParallel prefix
    sd_clean = strip_module_prefix(sd)
    n_stripped = sum(1 for k in sd if k.startswith("module."))
    if n_stripped > 0:
        print(f"[load_ckpt] Stripped 'module.' prefix from {n_stripped}/{len(sd)} keys")

    # Diagnostic: show first 3 keys after stripping
    print(f"[load_ckpt] Sample loaded keys: {list(sd_clean.keys())[:3]}")
    print(f"[load_ckpt] Sample model keys:  {list(model.state_dict().keys())[:3]}")

    # Try strict load first; if mismatches, fall back to non-strict and report
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)

    if missing:
        print(f"[load_ckpt] ⚠️  {len(missing)} MISSING keys (model has them, ckpt doesn't):")
        for k in missing[:5]:
            print(f"    - {k}")
        if len(missing) > 5:
            print(f"    ... ({len(missing) - 5} more)")
    if unexpected:
        print(f"[load_ckpt] ⚠️  {len(unexpected)} UNEXPECTED keys (in ckpt, model doesn't expect):")
        for k in unexpected[:5]:
            print(f"    - {k}")
        if len(unexpected) > 5:
            print(f"    ... ({len(unexpected) - 5} more)")

    if not missing and not unexpected:
        print(f"[load_ckpt] ✅ Perfect match — all {len(sd_clean)} keys loaded")

    return raw  # return full ckpt dict in case caller wants epoch/best_metric


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

def build_test_loader(params: dict, dataset_root_override: str = None, batch_size: int = 2):
    """Build test DataLoader using the project's WeedsGaloreDatasetInterface."""
    from wd.data.weedsgalore import WeedsGaloreDatasetInterface
    from easydict import EasyDict

    ds_params = deepcopy(params["dataset"])
    if dataset_root_override:
        ds_params["root"] = dataset_root_override
    # Force eval-mode flags
    ds_params["hor_flip"] = False
    ds_params["ver_flip"] = False
    ds_params["batch_size"] = batch_size
    ds_params["val_batch_size"] = batch_size
    ds_params["test_batch_size"] = batch_size

    interface = WeedsGaloreDatasetInterface(EasyDict(ds_params))
    # Force num_workers=0 to avoid Docker shared-memory issues with worker processes
    interface.build_data_loaders(batch_size_factor=1, num_workers=0)
    return interface.test_loader


# ────────────────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────────────────

def _model_forward(model, x):
    """Handle different output formats (tuple, dict, ComposedOutput, plain tensor)."""
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("out", out.get("main", out))
    elif hasattr(out, "main"):
        out = out.main
    return out


@torch.no_grad()
def predict_single(model, x, tta: bool = False):
    """Single forward με optional TTA (h-flip + v-flip averaging)."""
    if not tta:
        return F.softmax(_model_forward(model, x), dim=1)
    # 4 augmentations averaged
    p_orig = F.softmax(_model_forward(model, x), dim=1)
    p_h = F.softmax(_model_forward(model, torch.flip(x, dims=[3])), dim=1)
    p_h = torch.flip(p_h, dims=[3])
    p_v = F.softmax(_model_forward(model, torch.flip(x, dims=[2])), dim=1)
    p_v = torch.flip(p_v, dims=[2])
    p_hv = F.softmax(_model_forward(model, torch.flip(x, dims=[2, 3])), dim=1)
    p_hv = torch.flip(p_hv, dims=[2, 3])
    return (p_orig + p_h + p_v + p_hv) / 4.0


def evaluate(model, loader, device, num_classes=3, tta=False):
    """Aggregated CM-based metrics (matches paper convention)."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_images = 0

    for batch_idx, batch in enumerate(loader):
        # batch can be (img, mask), (img, mask, path)
        if len(batch) == 3:
            img, mask, _ = batch
        else:
            img, mask = batch
        img = img.to(device)
        mask = mask.to(device)

        probs = predict_single(model, img, tta=tta)
        pred = probs.argmax(dim=1)

        for t in range(num_classes):
            for p in range(num_classes):
                cm[t, p] += int(((mask == t) & (pred == p)).sum().item())
        n_images += img.size(0)
        if (batch_idx + 1) % 5 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)} processed")

    return cm, n_images


def compute_metrics(cm: np.ndarray, class_names: list) -> dict:
    """Per-class + macro IoU/F1/Precision/Recall από aggregated CM."""
    n = cm.shape[0]
    out = {"per_class": {}, "macro": {}}
    ious, f1s, precs, recs = [], [], [], []
    for c in range(n):
        tp = int(cm[c, c])
        fp = int(sum(cm[i, c] for i in range(n) if i != c))
        fn = int(sum(cm[c, j] for j in range(n) if j != c))
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou); f1s.append(f1); precs.append(prec); recs.append(rec)
        out["per_class"][class_names[c]] = {
            "iou": iou, "f1": f1, "precision": prec, "recall": rec
        }
    out["macro"] = {
        "mIoU": sum(ious) / n,
        "F1": sum(f1s) / n,
        "Precision": sum(precs) / n,
        "Recall": sum(recs) / n,
    }
    out["confusion_matrix"] = cm.tolist()
    return out


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="Path σε ckpt_best.pth")
    p.add_argument("--yaml", type=Path, required=True, help="YAML που χρησιμοποιήθηκε στο training (για identical arch)")
    p.add_argument("--dataset-root", default=None, help="Override dataset.root (αν διαφορετικό από YAML)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (h+v flip)")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--class-names", nargs="+", default=["background", "crop", "weed"])
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"ERROR: ckpt not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)
    if not args.yaml.exists():
        print(f"ERROR: yaml not found: {args.yaml}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}\nStandalone WeedsGalore Inference\n{'='*70}")
    print(f"Ckpt:       {args.ckpt}")
    print(f"Yaml:       {args.yaml}")
    print(f"Device:     {args.device}")
    print(f"TTA:        {'ENABLED (4× inference time)' if args.tta else 'disabled'}")
    print(f"{'='*70}\n")

    # 1. Load YAML
    params = load_yaml_params(args.yaml)
    print(f"[main] Loaded YAML params, model={params['model']['name']}, "
          f"channels={params['dataset']['channels']}")

    # 2. Build model
    model, arch_params = build_model_from_yaml(params, device=args.device)

    # 3. Load checkpoint
    raw_ckpt = load_checkpoint(args.ckpt, model, device=args.device)
    if isinstance(raw_ckpt, dict):
        if "epoch" in raw_ckpt:
            print(f"[main] Loaded ckpt was saved at epoch {raw_ckpt['epoch']}")
        if "acc" in raw_ckpt:
            print(f"[main] Loaded ckpt saved best_metric (acc): {raw_ckpt['acc']}")

    # 4. Build test loader
    loader = build_test_loader(params, args.dataset_root, args.batch_size)
    print(f"[main] Test loader ready: {len(loader)} batches\n")

    # 5. Evaluate
    print(f"{'='*70}\nRunning inference{'  + TTA' if args.tta else ''}...\n{'='*70}")
    cm, n_images = evaluate(model, loader, args.device, args.num_classes, args.tta)
    print(f"\n[main] Processed {n_images} test images")

    # 6. Metrics
    metrics = compute_metrics(cm, args.class_names)

    print(f"\n{'='*70}\nRESULTS{'  (with TTA)' if args.tta else ''}\n{'='*70}")
    print(f"\n{'class':<14} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    for name, m in metrics["per_class"].items():
        print(f"{name:<14} {m['iou']:>8.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f}")
    print(f"\n{'macro':<14} {metrics['macro']['mIoU']:>8.4f} "
          f"{metrics['macro']['F1']:>8.4f} {metrics['macro']['Precision']:>8.4f} "
          f"{metrics['macro']['Recall']:>8.4f}")
    print(f"\n  >> mIoU = {metrics['macro']['mIoU']:.4f}")
    print(f"  >> F1   = {metrics['macro']['F1']:.4f}")

    # 7. Save metrics JSON
    out_path = args.ckpt.parent / f"standalone_metrics{'_tta' if args.tta else ''}.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[main] Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
