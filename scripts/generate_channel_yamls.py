"""
Generator για channel ablation YAMLs (full + smoke versions).
Όλες οι configs χρησιμοποιούν Focal Tversky loss + SplitLawin B0 + ίδια
hyperparams για fair comparison. Διαφέρει μόνο το `dataset.channels` +
`model.params.main_channels`/`side_pretrained`/`main_pretrained`.

Run once:
    python scripts/generate_channel_yamls.py

Παράγει: params/WeedsGalore/Channels_<NAME>.yaml + smoke version.
"""
from pathlib import Path

OUT = Path("params/WeedsGalore")

# (name, channels, main_channels, main_pretrained, side_pretrained, comment)
COMBOS = [
    ("RGB_NDVI",
     ['R', 'G', 'B', 'NDVI'], 3,
     ['R', 'G', 'B'], 'G',
     "main=RGB pretrained, side=NDVI (1ch G-pretrained)"),

    ("RGB_CIR",
     ['R', 'G', 'B', 'NIR', 'G', 'R'], 3,
     ['R', 'G', 'B'], 'G',
     "main=RGB pretrained, side=CIR composite (NIR,G,R) with G-pretrained"),

    ("RGB_NDVI_CIR",
     ['R', 'G', 'B', 'NDVI', 'NIR', 'G', 'R'], 3,
     ['R', 'G', 'B'], 'G',
     "main=RGB, side=NDVI+CIR (4ch) all G-pretrained - max spectral info"),

    ("RGB_NDVI_CIR_RE",
     ['R', 'G', 'B', 'NDVI', 'NIR', 'G', 'R', 'RE'], 3,
     ['R', 'G', 'B'], 'G',
     "main=RGB, side=NDVI+CIR+RE (5ch) all G-pretrained - full spectral"),

    ("CIR_NDVI",
     ['NIR', 'G', 'R', 'NDVI'], 3,
     ['B', 'G', 'R'], 'G',
     "main=CIR composite (NIR,G,R) with B-G-R pretrained slots, side=NDVI"),
]


FULL_TEMPLATE = """
# Channel ablation: {name}
# {comment}
# SplitLawin MiT-B0 | Focal Tversky | NoTile 608 | mixed_precision (1080 Ti)
# NOTE: ASCII-only YAML (ezdl loader needs UTF-8, Windows defaults to cp1253)


experiment:
  name: weedmapping-weedsgalore
  group: ChannelAblations-FocalTversky
  continue_with_errors: true
  start_from_grid: 0
  start_from_run: 0
  tracking_dir: null
  logger: wandb
  entity: null
  excluded_files: "*.pth"

parameters:
  tags: [[channel_ablation, {tag}, focal_tversky, no_tile, weedsgalore, splitlawin_b0]]
  phases: [[train, test]]
  dataset_interface: [wd/data/WeedsGaloreDatasetInterface]

  train_params:
    max_epochs: [500]
    initial_lr: [0.0001]
    optimizer: [Adam]
    optimizer_params:
      weight_decay: [0]
    loss:
      name: [focal_tversky]
      params:
        alpha: [0.3]
        beta: [0.7]
        gamma: [1.3333]
    seed: [42]
    zero_weight_decay_on_bias_and_bn: [True]
    average_best_models: [False]
    greater_metric_to_watch_is_better: [False]
    metric_to_watch: [loss]
    freeze_pretrained: [False]
    mixed_precision: [True]

  early_stopping:
    patience: [30]
    monitor: [loss]
    mode: [min]

  train_metrics:
    f1: &metric_params
      num_classes: &num_classes [3]
    iou: *metric_params
  test_metrics:
    iou: *metric_params
    jaccard: *metric_params
    conf_mat: *metric_params
    auc: *metric_params
    f1: *metric_params
    precision: *metric_params
    recall: *metric_params
    perclassauc:
      discriminator:
        [[
          ['auc_background', 0],
          ['auc_crop', 1],
          ['auc_weed', 2]
        ]]

  model:
    name: [wd/models/splitlawin]
    params:
      backbone: [MiT-B0]
      backbone_pretrained: [True]
      main_channels: [{main_channels}]
      main_pretrained: [{main_pretrained!r}]
      side_pretrained: [{side_pretrained!r}]
      fusion_type: ['squeeze_excite']

  dataset:
    root: ["/workspace/datasets/weedsgalore/weedsgalore-dataset"]
    tile: [False]
    hor_flip: [True]
    ver_flip: [True]
    channels: [{channels!r}]
    batch_size: [4]
    val_batch_size: [2]
    test_batch_size: [2]
    num_workers: [0]
    num_classes: [3]
    return_path: [True]
    size: [same]
    crop_size: [same]

  test_callbacks:
    PerExampleMetricCallback:
      phase: [TEST_BATCH_END]

other_grids: []
"""


SMOKE_TEMPLATE = """
# Smoke test (1 epoch) για channel ablation: {name}
# {comment}


experiment:
  name: weedmapping-weedsgalore
  group: ChannelAblations-SMOKE
  continue_with_errors: true
  start_from_grid: 0
  start_from_run: 0
  tracking_dir: null
  logger: wandb
  entity: null
  excluded_files: "*.pth"

parameters:
  tags: [[smoke, channel_ablation, {tag}, focal_tversky, no_tile, weedsgalore, splitlawin_b0]]
  phases: [[train, test]]
  dataset_interface: [wd/data/WeedsGaloreDatasetInterface]

  train_params:
    max_epochs: [1]
    initial_lr: [0.0001]
    optimizer: [Adam]
    optimizer_params:
      weight_decay: [0]
    loss:
      name: [focal_tversky]
      params:
        alpha: [0.3]
        beta: [0.7]
        gamma: [1.3333]
    seed: [42]
    zero_weight_decay_on_bias_and_bn: [True]
    average_best_models: [False]
    greater_metric_to_watch_is_better: [False]
    metric_to_watch: [loss]
    freeze_pretrained: [False]
    mixed_precision: [True]

  early_stopping:
    patience: [5]
    monitor: [loss]
    mode: [min]

  train_metrics:
    f1: &metric_params
      num_classes: &num_classes [3]
    iou: *metric_params
  test_metrics:
    iou: *metric_params
    f1: *metric_params

  model:
    name: [wd/models/splitlawin]
    params:
      backbone: [MiT-B0]
      backbone_pretrained: [True]
      main_channels: [{main_channels}]
      main_pretrained: [{main_pretrained!r}]
      side_pretrained: [{side_pretrained!r}]
      fusion_type: ['squeeze_excite']

  dataset:
    root: ["/workspace/datasets/weedsgalore/weedsgalore-dataset"]
    tile: [False]
    hor_flip: [True]
    ver_flip: [True]
    channels: [{channels!r}]
    batch_size: [4]
    val_batch_size: [2]
    test_batch_size: [2]
    num_workers: [0]
    num_classes: [3]
    return_path: [True]
    size: [same]
    crop_size: [same]

  test_callbacks:
    PerExampleMetricCallback:
      phase: [TEST_BATCH_END]

other_grids: []
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, channels, main_channels, main_pretrained, side_pretrained, comment in COMBOS:
        ctx = {
            "name": name,
            "tag": name.lower(),
            "comment": comment,
            "channels": channels,
            "main_channels": main_channels,
            "main_pretrained": main_pretrained,
            "side_pretrained": side_pretrained,
        }
        full_path = OUT / f"Channels_{name}.yaml"
        smoke_path = OUT / f"Channels_{name}_smoke.yaml"
        full_path.write_text(FULL_TEMPLATE.format(**ctx), encoding="utf-8")
        smoke_path.write_text(SMOKE_TEMPLATE.format(**ctx), encoding="utf-8")
        print(f"  Wrote: {full_path}")
        print(f"  Wrote: {smoke_path}")


if __name__ == "__main__":
    main()
