"""
Aggregate test_results.csv (per-image breakdown) → global per-class IoU/F1.

Το CSV γράφεται από το PerExampleMetricCallback στο τέλος του test phase.
Κάθε γραμμή = 1 test image. Αυτό το script:
  1. Αθροίζει το confusion matrix από όλα τα images
  2. Υπολογίζει global per-class IoU, F1, Precision, Recall
  3. Τυπώνει macro averages (mIoU, mF1)

Usage:
    python scripts/aggregate_test_results.py path/to/test_results.csv
    python scripts/aggregate_test_results.py path/to/test_results.csv --classes background crop weed
    python scripts/aggregate_test_results.py path/to/test_results.csv --num-classes 3
"""
import argparse
import csv
import sys
from pathlib import Path


def aggregate(csv_path: Path, num_classes: int, class_names: list[str] | None = None):
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"ERROR: {csv_path} είναι κενό.", file=sys.stderr)
        sys.exit(1)

    # Sum confusion matrix across all images
    cm = [[0] * num_classes for _ in range(num_classes)]
    for r in rows:
        for i in range(num_classes):
            for j in range(num_classes):
                key = f"cf_{i}_{j}"
                if key not in r:
                    print(f"ERROR: λείπει στήλη '{key}' στο CSV. "
                          f"Έχεις σωστό --num-classes (έδωσες {num_classes});",
                          file=sys.stderr)
                    sys.exit(1)
                cm[i][j] += int(r[key])

    print(f"Test images: {len(rows)}")
    print(f"\nAggregated confusion matrix (rows=true, cols=pred):")
    header = "             " + "".join(f"  pred_{n[:8]:>10}" for n in (class_names or [str(c) for c in range(num_classes)]))
    print(header)
    for i in range(num_classes):
        label = (class_names[i] if class_names else f"class_{i}").ljust(10)
        row = " ".join(f"{cm[i][j]:>14}" for j in range(num_classes))
        print(f"true_{label}  {row}")

    # Per-class metrics
    print(f"\n=== Per-class metrics (από aggregated CM) ===")
    print(f"{'class':<14} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    ious, f1s, precs, recs = [], [], [], []
    for c in range(num_classes):
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(num_classes) if j != c)
        fp = sum(cm[i][c] for i in range(num_classes) if i != c)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou); f1s.append(f1); precs.append(prec); recs.append(rec)
        name = class_names[c] if class_names else f"class_{c}"
        print(f"{name:<14} {iou:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}")

    # Macro averages
    print(f"\n=== Macro averages ===")
    print(f"  mIoU      = {sum(ious) / num_classes:.4f}")
    print(f"  macro F1  = {sum(f1s) / num_classes:.4f}")
    print(f"  macro Prec= {sum(precs) / num_classes:.4f}")
    print(f"  macro Rec = {sum(recs) / num_classes:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path", type=Path, help="Διαδρομή στο test_results.csv")
    p.add_argument("--num-classes", type=int, default=3, help="Αριθμός κλάσεων (default: 3)")
    p.add_argument("--classes", nargs="+", default=["background", "crop", "weed"],
                   help="Ονόματα κλάσεων (default: background crop weed)")
    args = p.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: δεν βρέθηκε {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    if len(args.classes) != args.num_classes:
        print(f"WARNING: δόθηκαν {len(args.classes)} class names αλλά --num-classes={args.num_classes}. "
              f"Αγνοώ τα names.", file=sys.stderr)
        args.classes = None

    aggregate(args.csv_path, args.num_classes, args.classes)


if __name__ == "__main__":
    main()
