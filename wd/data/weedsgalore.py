"""
WeedsGalore dataset integration for LWViTs.

Key differences from WeedMap:
- 16-bit PNGs (÷65535 for [0,1])
- 600×600 images → 9 tiles (3×3, stride=172, size=256×256)
- NDVI and CIR computed on-the-fly
- Split-file based (train.txt / val.txt / test.txt)
- Labels >2 collapsed to 2 (weed subclasses → single weed class)
"""
import itertools
import os
from typing import Any, Union, Iterable

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from ezdl.transforms import (
    PairRandomCrop, ToLong, Denormalize, PairRandomFlip,
    PairFlip, PairFourCrop, squeeze0
)
from ezdl.data import DatasetInterface

from torch.utils.data.distributed import DistributedSampler
from super_gradients.training import utils as core_utils
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.common.exceptions.dataset_exceptions import IllegalDatasetParameterException
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

# ---------- Constants ----------
TILE_SIZE = 256
GRID_POSITIONS = [0, 172, 344]  # 3×3 grid, stride=172, last crop ends at 344+256=600
RAW_BANDS = ['R', 'G', 'B', 'NIR', 'RE']

# No-tile mode: resize 600x600 -> FULL_IMAGE_SIZE (divisible by 32 for MiT)
FULL_IMAGE_SIZE = 608


# ---------- Transforms ----------
class CollapseWeedClasses:
    """Map all weed sub-classes (labels > 2) to a single weed class (2)."""
    def __call__(self, x):
        x[x > 2] = 2
        return x


class _ResizeLabel:
    """Resize a 2D long-tensor label (H, W) using NEAREST interpolation.

    torchvision's Resize expects (C, H, W); wrap/unwrap the channel dim and
    preserve dtype (long) since NEAREST does not change class ids.
    """
    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(self, x):
        # x: (H, W) long tensor
        x = x.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        x = torch.nn.functional.interpolate(x, size=self.size, mode='nearest')
        return x.squeeze(0).squeeze(0).long()


# ---------- Dataset Interface ----------
class WeedsGaloreDatasetInterface(DatasetInterface):
    """
    Manages train/val/test datasets and data loaders for WeedsGalore.
    Follows the same pattern as WeedMapDatasetInterface.
    """
    size = (5, TILE_SIZE, TILE_SIZE)

    def __init__(self, dataset_params, name=None):
        super().__init__(dataset_params)
        channels = dataset_params['channels']

        # Tiling flag: default True (legacy 3×3 256 tiles); False = full 600→608 images
        tile = core_utils.get_param(dataset_params, 'tile', default_val=True)
        self.tile = tile

        # Resolve how many actual tensor channels we have
        # NOTE: 'CIR' counts as 3 channels (NIR, G, R composite) ακόμα και μέσα σε list
        if channels == 'CIR':
            n_channels = 3
        elif isinstance(channels, str):
            n_channels = 1
        else:
            n_channels = sum(3 if c == 'CIR' else 1 for c in channels)
        spatial = TILE_SIZE if tile else FULL_IMAGE_SIZE
        self.size = (n_channels, spatial, spatial)

        # Compute normalization stats from training scenes
        mean, std = self._get_mean_std(dataset_params)

        self.lib_dataset_params = {
            'mean': mean,
            'std': std,
            'channels': channels,
        }

        # --- Build transforms ---
        input_transform = [
            transforms.Normalize(mean, std),
        ]

        test_transform = [
            transforms.Normalize(mean, std),
        ]

        target_transform = [
            ToLong(),
            CollapseWeedClasses(),
        ]

        test_target_transform = [
            ToLong(),
            CollapseWeedClasses(),
        ]

        period = 1

        crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val='same')
        if crop_size != 'same':
            crop = PairRandomCrop(crop_size)
            test_crop = PairFourCrop(crop_size, periodicity=period)
            input_transform.append(crop)
            target_transform.append(crop)
            test_transform.append(test_crop)
            test_target_transform.append(test_crop)
            period *= 4

        if dataset_params.get('hor_flip', False):
            flip_hor = PairRandomFlip(orientation="horizontal")
            input_transform.append(flip_hor)
            target_transform.append(flip_hor)

        if dataset_params.get('ver_flip', False):
            flip_ver = PairRandomFlip(orientation="vertical")
            input_transform.append(flip_ver)
            target_transform.append(flip_ver)

        if core_utils.get_param(self.dataset_params, 'size', default_val='same') != 'same':
            resize = transforms.Resize(
                size=core_utils.get_param(self.dataset_params, 'size', default_val='same')
            )
            input_transform.append(resize)
            target_transform.append(resize)
            test_transform.append(resize)
            test_target_transform.append(resize)

        # No-tile mode: auto-resize 600 -> FULL_IMAGE_SIZE (divisible by 32 for MiT)
        if not tile:
            # BILINEAR for input imagery, NEAREST for labels
            input_resize = transforms.Resize(
                size=(FULL_IMAGE_SIZE, FULL_IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            label_resize = transforms.Resize(
                size=(FULL_IMAGE_SIZE, FULL_IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            input_transform.append(input_resize)
            test_transform.append(input_resize)
            # Label is (H, W) long; Resize expects (C, H, W) or PIL — wrap to add/remove channel dim
            target_transform.append(_ResizeLabel(FULL_IMAGE_SIZE))
            test_target_transform.append(_ResizeLabel(FULL_IMAGE_SIZE))

        target_transform = transforms.Compose(target_transform)
        input_transform = transforms.Compose(input_transform)
        test_transform = transforms.Compose(test_transform)
        test_target_transform = transforms.Compose(test_target_transform)

        # --- Build indices from split files ---
        root = dataset_params['root']
        splits_dir = os.path.join(root, 'splits')

        train_ids = _read_split(splits_dir, 'train')
        val_ids = _read_split(splits_dir, 'val')
        test_ids = _read_split(splits_dir, 'test')

        train_index = WeedsGaloreDataset.build_index(train_ids, tile=tile)
        val_index = WeedsGaloreDataset.build_index(val_ids, tile=tile)
        test_index = WeedsGaloreDataset.build_index(test_ids, tile=tile)

        self.trainset = WeedsGaloreDataset(
            root=root, channels=channels,
            batch_size=dataset_params['batch_size'],
            index=train_index,
            transform=input_transform, target_transform=target_transform,
            return_path=dataset_params.get('return_path', False),
            tile=tile,
        )

        self.valset = WeedsGaloreDataset(
            root=root, channels=channels,
            batch_size=dataset_params.get('val_batch_size', dataset_params['batch_size']),
            index=val_index,
            transform=input_transform, target_transform=target_transform,
            return_path=dataset_params.get('return_path', False),
            tile=tile,
        )

        self.testset = WeedsGaloreDataset(
            root=root, channels=channels,
            batch_size=dataset_params.get('test_batch_size', dataset_params['batch_size']),
            index=test_index,
            transform=test_transform, target_transform=test_target_transform,
            return_path=dataset_params.get('return_path', False),
            period=period,
            tile=tile,
        )

    def undo_preprocess(self, x):
        return (
            Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255
        ).type(torch.uint8)

    @staticmethod
    def _get_mean_std(dataset_params):
        """
        Compute mean/std from pre-computed stats file, aggregated across training dates.
        Follows the same approach as WeedMapDatasetInterface.get_mean_std.
        """
        from wd.data.weedsgalore_stats import STATS

        channels = dataset_params['channels']
        if channels == 'CIR':
            channels = ['NIR', 'G', 'R']
        elif isinstance(channels, (list, tuple)):
            # Expand 'CIR' tokens μέσα σε list (συμβατό με _get_image)
            expanded = []
            for c in channels:
                if c == 'CIR':
                    expanded.extend(['NIR', 'G', 'R'])
                else:
                    expanded.append(c)
            channels = expanded

        # OSAVI and MSAVI don't have pre-computed stats. Use NDVI stats as approximation
        # (similar formula structure & dynamic range ~[-1, 1] for vegetation imagery).
        def stats_lookup(date_dict, name):
            if name in ('OSAVI', 'MSAVI'):
                return date_dict['NDVI']
            return date_dict[name]

        # Aggregate across all dates in the stats dict
        dates = list(STATS.keys())
        if len(dates) == 1:
            d = dates[0]
            return list(zip(*[(stats_lookup(STATS[d], c)['mean'],
                               stats_lookup(STATS[d], c)['std']) for c in channels]))

        sums = {
            **{c + '_sum': sum(stats_lookup(STATS[d], c)['sum'] for d in dates) for c in channels},
            **{c + '_sum_sq': sum(stats_lookup(STATS[d], c)['sum_sq'] for d in dates) for c in channels},
        }
        count = sum(STATS[d]['count'] for d in dates)
        means = [sums[c + '_sum'] / count for c in channels]
        stds = [
            np.sqrt((sums[c + '_sum_sq'] / count) - (means[i] ** 2))
            for i, c in enumerate(channels)
        ]
        return means, stds

    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None,
                           val_batch_size=None, test_batch_size=None,
                           distributed_sampler: bool = False, **kwargs):
        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = DistributedSampler(self.trainset)
            val_sampler = DistributedSampler(self.valset)
            test_sampler = DistributedSampler(self.testset) if self.testset is not None else None
            train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params.batch_size * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params.val_batch_size * self.batch_size_factor
        if test_batch_size is None:
            test_batch_size = self.dataset_params.test_batch_size * self.batch_size_factor

        train_loader_drop_last = core_utils.get_param(
            self.dataset_params, 'train_loader_drop_last', default_val=False
        )

        cutmix = core_utils.get_param(self.dataset_params, 'cutmix', False)
        cutmix_params = core_utils.get_param(self.dataset_params, 'cutmix_params')

        train_collate_fn = core_utils.get_param(self.trainset, 'collate_fn')
        val_collate_fn = core_utils.get_param(self.valset, 'collate_fn')
        test_collate_fn = core_utils.get_param(self.testset, 'collate_fn')

        if cutmix and train_collate_fn is not None:
            raise IllegalDatasetParameterException("cutmix and collate function cannot be used together")
        if cutmix:
            train_collate_fn = CollateMixup(**cutmix_params)

        pw = num_workers > 0
        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=train_batch_size, shuffle=train_shuffle,
            num_workers=num_workers, pin_memory=True, sampler=train_sampler,
            collate_fn=train_collate_fn, drop_last=train_loader_drop_last, persistent_workers=pw,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.valset, batch_size=val_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, sampler=val_sampler,
            collate_fn=val_collate_fn, persistent_workers=pw,
        )
        if self.testset is not None:
            self.test_loader = torch.utils.data.DataLoader(
                self.testset, batch_size=test_batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, sampler=test_sampler,
                collate_fn=test_collate_fn, persistent_workers=pw,
            )
        self.classes = self.trainset.classes


# ---------- Utility functions ----------
def _read_split(splits_dir, split_name):
    """Read scene IDs from split file."""
    path = os.path.join(splits_dir, f"{split_name}.txt")
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _scene_id_to_date(scene_id):
    """'2023-06-15_0472' -> '2023-06-15'"""
    return scene_id[:10]


# ---------- Dataset ----------
class WeedsGaloreDataset(VisionDataset):
    """
    WeedsGalore dataset with 3×3 tiling of 600×600 images into 256×256 crops.

    Index structure: list of (scene_id, row, col) tuples where
    row, col in {0, 1, 2} map to pixel offsets via GRID_POSITIONS.
    """
    CLASS_LABELS = {0: "background", 1: "crop", 2: "weed"}
    classes = ['background', 'crop', 'weed']
    id2label = {0: "background", 1: "crop", 2: "weed"}

    def __init__(
        self,
        root: str,
        transform: callable,
        target_transform: callable,
        channels: Union[str, Iterable] = 'CIR',
        index: Iterable = None,
        batch_size: int = 4,
        return_path: bool = False,
        period: int = None,
        tile: bool = True,
    ):
        super().__init__(root=root)
        self.batch_size = batch_size
        self.tile = tile

        # Resolve channel loading strategy
        if channels == 'CIR':
            self._load_channels = ['NIR', 'G', 'R']
            self._compute_ndvi = False
        elif isinstance(channels, list) or isinstance(channels, tuple):
            # Expand 'CIR' tokens μέσα σε list → [NIR, G, R] composite.
            # Π.χ. ['R','G','B','CIR'] → ['R','G','B','NIR','G','R'] (6 ch)
            expanded = []
            for c in channels:
                if c == 'CIR':
                    expanded.extend(['NIR', 'G', 'R'])
                else:
                    expanded.append(c)
            # Separate raw bands from derived ones (deduped via _get_image)
            self._raw_bands = [c for c in expanded if c in RAW_BANDS]
            self._compute_ndvi = 'NDVI' in expanded
            self._compute_osavi = 'OSAVI' in expanded
            self._compute_msavi = 'MSAVI' in expanded
            self._compute_cir = False
            self._channel_order = expanded  # use expanded list για stacking
            self._load_channels = None
        else:
            raise ValueError(f"Unsupported channels spec: {channels}")

        self.channels = channels

        if index is not None:
            self.index = index
        else:
            raise ValueError("Index must be provided for WeedsGaloreDataset")

        if period is not None:
            self.index = list(
                itertools.chain.from_iterable(itertools.repeat(x, period) for x in self.index)
            )

        self.len = len(self.index)
        self.path = root
        self.transform = transform
        self.target_transform = target_transform
        self.return_name = return_path

    @classmethod
    def build_index(cls, scene_ids, tile: bool = True):
        """
        Build dataset index.
        - If tile=True: 9 tiles per scene (3×3 grid) -> (scene_id, row, col) tuples.
        - If tile=False: 1 full image per scene -> (scene_id, None, None) tuples.
        """
        if not tile:
            return [(sid, None, None) for sid in scene_ids]
        index = []
        for sid in scene_ids:
            for row in range(3):
                for col in range(3):
                    index.append((sid, row, col))
        return index

    def _load_band(self, scene_id, band):
        """Load a single 16-bit PNG band, return float32 tensor in [0, 1]."""
        date = _scene_id_to_date(scene_id)
        path = os.path.join(self.path, date, "images", f"{scene_id}_{band}.png")
        img = Image.open(path)
        # 16-bit PNG: convert to numpy first, then normalize
        arr = np.array(img, dtype=np.float32) / 65535.0
        return torch.from_numpy(arr)  # shape: (H, W)

    def _load_label(self, scene_id):
        """Load semantic label PNG, return as tensor."""
        date = _scene_id_to_date(scene_id)
        path = os.path.join(self.path, date, "semantics", f"{scene_id}.png")
        img = Image.open(path)
        arr = np.array(img, dtype=np.int64)
        return torch.from_numpy(arr)  # shape: (H, W)

    def _crop_tile(self, tensor, row, col):
        """Crop a 256×256 tile from the 3×3 grid position."""
        y = GRID_POSITIONS[row]
        x = GRID_POSITIONS[col]
        return tensor[..., y:y + TILE_SIZE, x:x + TILE_SIZE]

    def _get_image(self, scene_id):
        """Load and stack all requested channels for a scene."""
        if self.channels == 'CIR':
            # CIR = stack(NIR, G, R)
            bands = [self._load_band(scene_id, b) for b in ['NIR', 'G', 'R']]
            return torch.stack(bands, dim=0)  # (3, H, W)

        # Multi-channel: load raw bands + compute derived ones
        band_data = {}
        # Load only the raw bands we need
        needed_raw = set(self._raw_bands)
        if self._compute_ndvi or self._compute_osavi or self._compute_msavi:
            needed_raw.add('NIR')
            needed_raw.add('R')

        for b in needed_raw:
            band_data[b] = self._load_band(scene_id, b)

        # Compute NDVI if requested: (NIR - R) / (NIR + R)
        if self._compute_ndvi:
            nir = band_data['NIR']
            red = band_data['R']
            band_data['NDVI'] = (nir - red) / (nir + red + 1e-10)

        # Compute OSAVI: (NIR - R) / (NIR + R + 0.16) — soil-adjusted, reduces soil bias
        if self._compute_osavi:
            nir = band_data['NIR']
            red = band_data['R']
            band_data['OSAVI'] = (nir - red) / (nir + red + 0.16)

        # Compute MSAVI: 0.5 · (2·NIR + 1 - √((2·NIR + 1)² - 8·(NIR - R)))
        # Self-calibrating SAVI, καλύτερο σε bare soil scenarios.
        if self._compute_msavi:
            nir = band_data['NIR']
            red = band_data['R']
            inner = (2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)
            # Clamp για numerical stability (rare edge case όπου το inner γίνεται <0)
            inner = torch.clamp(inner, min=0.0)
            band_data['MSAVI'] = 0.5 * (2.0 * nir + 1.0 - torch.sqrt(inner))

        # Stack in user-specified order
        stacked = torch.stack([band_data[c] for c in self._channel_order], dim=0)
        return stacked  # (C, H, W)

    def __getitem__(self, index: int) -> Any:
        scene_id, row, col = self.index[index]

        # Load full image and label
        img = self._get_image(scene_id)    # (C, 600, 600)
        label = self._load_label(scene_id)  # (600, 600)

        if self.tile:
            # Crop tile
            img = self._crop_tile(img, row, col)      # (C, 256, 256)
            label = self._crop_tile(label, row, col)   # (256, 256)
            sample_name = f"{scene_id}_{row}_{col}"
        else:
            # Full-image mode: keep 600×600; transforms will resize to FULL_IMAGE_SIZE
            sample_name = scene_id

        # Apply transforms
        img = self.transform(img)
        label = self.target_transform(label)

        if self.return_name:
            return img, label, {'input_name': sample_name}
        return img, label

    def __len__(self) -> int:
        return self.len
