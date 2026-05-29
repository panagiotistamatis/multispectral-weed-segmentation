import torch
from super_gradients.training.utils import get_param, HpmStruct
from super_gradients.training import utils as sg_utils
from torch import Tensor
from torch.nn import functional as F

from ezdl.utils.utilities import filter_none
from ezdl.models.backbones.mit import MiTFusion
from ezdl.models.base import BaseModel
from ezdl.models.heads.lawin import LawinHead
from ezdl.models.heads.laweed import LaweedHead


class BaseLawin(BaseModel):
    """
    Abstract base lawin class with free decoder head lawin based
    """

    def __init__(self, arch_params, lawin_class) -> None:
        num_classes = get_param(arch_params, "num_classes")
        input_channels = get_param(arch_params, "input_channels", 3)
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        backbone_pretrained = get_param(arch_params, "backbone_pretrained", False)
        pretrained_channels = get_param(arch_params, "main_pretrained", None)
        super().__init__(backbone, input_channels, backbone_pretrained)
        self.decode_head = lawin_class(self.backbone.channels, 256 if 'B0' in backbone else 512, num_classes)
        self.apply(self._init_weights)
        if backbone_pretrained:
            self.main_pretrained = pretrained_channels
            if isinstance(pretrained_channels, str):
                self.main_pretrained = [pretrained_channels] * input_channels
            else:
                self.main_pretrained = pretrained_channels
            # Optional per-channel scaling (I3D-style: NIR/RE ← 0.6·RGB pretrained)
            main_scales = get_param(arch_params, "main_pretrained_scales", None)
            self.backbone.init_pretrained_weights(
                channels_to_load=self.main_pretrained,
                channel_scales=main_scales,
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y


class Lawin(BaseLawin):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """

    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LawinHead)


class Laweed(BaseLawin):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """

    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LaweedHead)


class BaseDoubleLawin(BaseLawin):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """

    def __init__(self, arch_params, lawin_class) -> None:
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        main_channels = get_param(arch_params, "main_channels", None)
        if main_channels is None:
            raise ValueError("Please provide main_channels")
        self.side_channels = arch_params['input_channels'] - main_channels
        self.side_pretrained = get_param(arch_params, "side_pretrained", None)
        self.main_channels = main_channels
        arch_params['input_channels'] = arch_params['main_channels']
        super().__init__(arch_params, lawin_class)
        self.side_backbone = self.eval_backbone(backbone, self.side_channels, pretrained=bool(self.side_pretrained))
        if self.side_pretrained is not None:
            if isinstance(self.side_pretrained, str):
                self.side_pretrained = [self.side_pretrained] * self.side_channels
            # Optional I3D-style per-channel scaling για side branch
            _side_scales = get_param(arch_params, "side_pretrained_scales", None)
            self.side_backbone.init_pretrained_weights(
                channels_to_load=self.side_pretrained,
                channel_scales=_side_scales,
            )
        p_local = get_param(arch_params, "p_local", None)
        p_glob = get_param(arch_params, "p_glob", None)
        fusion_type = get_param(arch_params, "fusion_type", None)
        self.fusion = MiTFusion(self.backbone.channels,
                                **filter_none({"p_local": p_local, "p_glob": p_glob, "fusion_type": fusion_type}))

    def forward(self, x: Tensor) -> Tensor:
        main_channels = x[:, :self.main_channels, ::].contiguous()
        side_channels = x[:, self.main_channels:, ::].contiguous()
        feat_main = self.backbone(main_channels)
        feat_side = self.side_backbone(side_channels)
        feat = self.fusion((feat_main, feat_side))
        y = self.decode_head(feat)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y


class DoubleLawin(BaseDoubleLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LawinHead)


class DoubleLaweed(BaseDoubleLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LaweedHead)


class ASVFInputModule(torch.nn.Module):
    """Adaptive Spectral-Vegetation Fusion at the input stage.

    Adapted from ASVLB-Net (Dong et al., Pest Manag Sci 2026, Eq. 1-6) for SplitLawin.
    Computes NDVI from NIR & R bands, then adaptively fuses raw spectral features
    με τον vegetation-index prior μέσω channel + spatial attention (learnable α).
    Το output ΔΙΑΤΗΡΕΙ το input channel count (residual), ώστε τα downstream backbones
    να κρατούν τα ImageNet-pretrained weights τους.
    """
    def __init__(self, in_channels: int, nir_idx: int, red_idx: int,
                 main_dim: int = 32, veg_dim: int = 16):
        super().__init__()
        self.nir_idx = nir_idx
        self.red_idx = red_idx
        # Main spectral branch (preliminary encoding, Eq. 1-context)
        self.main_conv = torch.nn.Conv2d(in_channels, main_dim, 3, padding=1)
        # Vegetation branch: NDVI → conv1x1 (Eq. 2)
        self.veg_conv = torch.nn.Conv2d(1, veg_dim, 1)
        cat_dim = main_dim + veg_dim
        # Channel attention (Eq. 3-4): GAP → conv-relu-conv → sigmoid
        self.ca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(cat_dim, max(cat_dim // 4, 1), 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(max(cat_dim // 4, 1), cat_dim, 1),
            torch.nn.Sigmoid(),
        )
        # Spatial attention (Eq. 5): per-branch 1x1 → learnable α blend → sigmoid
        self.main_sp = torch.nn.Conv2d(main_dim, 1, 1)
        self.veg_sp = torch.nn.Conv2d(veg_dim, 1, 1)
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        # Project back σε input channel count + residual (Eq. 6)
        self.proj = torch.nn.Conv2d(cat_dim, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        nir = x[:, self.nir_idx:self.nir_idx + 1]
        red = x[:, self.red_idx:self.red_idx + 1]
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = (ndvi + 1.0) * 0.5  # [-1,1] → [0,1]
        f_main = self.main_conv(x)
        f_v = self.veg_conv(ndvi)
        f_cat = torch.cat([f_main, f_v], dim=1)
        # channel attention (Eq. 3-4)
        f_c = f_cat * self.ca(f_cat)
        # spatial attention με learnable α (Eq. 5)
        m_s = torch.sigmoid(
            self.alpha * self.main_sp(f_main) + (1.0 - self.alpha) * self.veg_sp(f_v)
        )
        f_c = f_c * m_s
        # project back + residual (Eq. 6)
        return self.proj(f_c) + x


class BaseSplitLawin(BaseLawin):
    def __init__(self, arch_params, lawin_class) -> None:
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        main_channels = get_param(arch_params, "main_channels", None)
        if main_channels is None:
            raise ValueError("Please provide main_channels")
        self.side_channels = arch_params['input_channels'] - main_channels
        self.side_pretrained = get_param(arch_params, "side_pretrained", None)
        self.main_channels = main_channels
        # Total input channels (πριν το overwrite στη γραμμή που ακολουθεί)
        _total_input_channels = arch_params['input_channels']
        arch_params['input_channels'] = arch_params['main_channels']
        super().__init__(arch_params, lawin_class)
        self.side_backbone = self.eval_backbone(backbone, self.side_channels,
                                                n_blocks=1,
                                                pretrained=bool(self.side_pretrained))
        if self.side_pretrained is not None:
            if isinstance(self.side_pretrained, str):
                self.side_pretrained = [self.side_pretrained] * self.side_channels
            # Optional I3D-style per-channel scaling για side branch
            _side_scales = get_param(arch_params, "side_pretrained_scales", None)
            self.side_backbone.init_pretrained_weights(
                channels_to_load=self.side_pretrained,
                channel_scales=_side_scales,
            )
        p_local = get_param(arch_params, "p_local", None)
        p_glob = get_param(arch_params, "p_glob", None)
        fusion_type = get_param(arch_params, "fusion_type", None)
        self.fusion = MiTFusion(self.backbone.channels,
                                **filter_none({"p_local": p_local, "p_glob": p_glob, "fusion_type": fusion_type}))
        # Optional ASVF input module (ASVLB-Net inspired). Created AFTER super().__init__
        # ώστε να μην επηρεαστεί από το _init_weights pass του BaseLawin.
        if get_param(arch_params, "use_asvf", False):
            # Default indices για CIR+B+RE → expanded [NIR, G, R, B, RE]: NIR=0, R=2
            nir_idx = get_param(arch_params, "asvf_nir_idx", 0)
            red_idx = get_param(arch_params, "asvf_red_idx", 2)
            self.asvf = ASVFInputModule(_total_input_channels, nir_idx, red_idx)
        else:
            self.asvf = None

    def forward(self, x: Tensor) -> Tensor:
        if self.asvf is not None:
            x = self.asvf(x)
        main_channels = x[:, :self.main_channels, ::].contiguous()
        side_channels = x[:, self.main_channels:, ::].contiguous()
        first_feat_side = self.side_backbone(side_channels)
        first_feat_main = self.backbone.partial_forward(main_channels, slice(0, 1))
        first_feat = self.fusion((first_feat_main, first_feat_side))[0]
        feat = (first_feat,) + self.backbone.partial_forward(first_feat, slice(1, 4))
        y = self.decode_head(feat)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """

        def f(x):
            return not (x[0].startswith('backbone') and int(x[0].split('.')[4]) == 0)

        freeze_pretrained = sg_utils.get_param(training_params, 'freeze_pretrained', False)
        if self.backbone_pretrained and freeze_pretrained:
            return [{'named_params': list(filter(f, list(self.named_parameters())))}]
        return [{'named_params': self.named_parameters()}]


class SplitLawin(BaseSplitLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LawinHead)


class SplitLaweed(BaseSplitLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LaweedHead)


# Legacy names
lawin = Lawin
laweed = Laweed
doublelawin = DoubleLawin
doublelaweed = DoubleLaweed
splitlawin = SplitLawin
splitlaweed = SplitLaweed
