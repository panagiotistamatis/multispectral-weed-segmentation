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
        # Optional configurable Lawin attention ratios + patch_size.
        # Default behavior: ratios=[8,4,2], patch_size=8 (paper-original, 256×256 input).
        # Για 608×608: π.χ. lawin_ratios=[12,4,2] ή patch_size=12 με ratios=[6,4,2].
        # Conditional pass — μόνο LawinHead δέχεται τα κλειδιά (LaweedHead δεν).
        lawin_ratios = get_param(arch_params, "lawin_ratios", None)
        lawin_patch_size = get_param(arch_params, "lawin_patch_size", None)
        head_extra = {}
        if lawin_ratios is not None:
            head_extra["ratios"] = lawin_ratios
        if lawin_patch_size is not None:
            head_extra["patch_size"] = lawin_patch_size
        self.decode_head = lawin_class(
            self.backbone.channels,
            256 if 'B0' in backbone else 512,
            num_classes,
            **head_extra,
        )
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

    Author's own proposed module for SplitLawin. Computes NDVI from the NIR & R
    bands, then adaptively fuses the raw spectral features with the vegetation-index
    prior via channel attention (inspired by Squeeze-and-Excitation, Hu et al.) and
    spatial attention (in the style of CBAM, Woo et al.), balanced by a learnable α.
    Το output ΔΙΑΤΗΡΕΙ το input channel count (residual), ώστε τα downstream backbones
    να κρατούν τα ImageNet-pretrained weights τους.
    """
    def __init__(self, in_channels: int, nir_idx: int, red_idx: int,
                 main_dim: int = 32, veg_dim: int = 16,
                 alpha_init: float = 0.5,
                 use_ndre: bool = False, re_idx: int = None,
                 beta_init: float = 0.5):
        """
        Args:
            nir_idx, red_idx: input channel positions για NDVI = (NIR-R)/(NIR+R)
            main_dim, veg_dim: feature dims για main + each vegetation branch
            alpha_init: learnable α initial value (main vs vegetation spatial blend, Eq.5)
            use_ndre: αν True, προστίθεται 2η vegetation branch με NDRE = (NIR-RE)/(NIR+RE)
            re_idx: input channel position για RE (required αν use_ndre=True)
            beta_init: learnable β initial value (NDVI vs NDRE blend, μόνο αν use_ndre)
        """
        super().__init__()
        self.nir_idx = nir_idx
        self.red_idx = red_idx
        self.use_ndre = use_ndre
        self.re_idx = re_idx
        if use_ndre and re_idx is None:
            raise ValueError("use_ndre=True requires re_idx")
        # Main spectral branch (preliminary encoding, Eq. 1-context)
        self.main_conv = torch.nn.Conv2d(in_channels, main_dim, 3, padding=1)
        # Vegetation branch: NDVI → conv1x1 (Eq. 2)
        self.ndvi_conv = torch.nn.Conv2d(1, veg_dim, 1)
        # Optional NDRE branch — complementary chlorophyll-sensitive index
        cat_dim = main_dim + veg_dim
        if use_ndre:
            self.ndre_conv = torch.nn.Conv2d(1, veg_dim, 1)
            self.ndre_sp = torch.nn.Conv2d(veg_dim, 1, 1)
            self.beta = torch.nn.Parameter(torch.tensor(float(beta_init)))
            cat_dim += veg_dim
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
        self.ndvi_sp = torch.nn.Conv2d(veg_dim, 1, 1)
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init)))
        # Project back σε input channel count + residual (Eq. 6)
        self.proj = torch.nn.Conv2d(cat_dim, in_channels, 1)

    @staticmethod
    def _normalized_diff(band_a: Tensor, band_b: Tensor) -> Tensor:
        """Compute (a-b)/(a+b), clamp σε [-1,1] → re-normalize σε [0,1].
        Computed in FP32 για AMP numerical stability (FP16 ε=1e-6 underflows)."""
        denom = (band_a + band_b).clamp(min=1e-3)
        ndi = (band_a - band_b) / denom
        return ((ndi + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        # NDVI/NDRE υπολογίζονται σε FP32 για numerical stability υπό mixed precision (AMP).
        x32 = x.float()
        nir = x32[:, self.nir_idx:self.nir_idx + 1]
        red = x32[:, self.red_idx:self.red_idx + 1]
        ndvi = self._normalized_diff(nir, red).to(x.dtype)

        f_main = self.main_conv(x)
        f_ndvi = self.ndvi_conv(ndvi)
        feats = [f_main, f_ndvi]

        if self.use_ndre:
            re_band = x32[:, self.re_idx:self.re_idx + 1]
            ndre = self._normalized_diff(nir, re_band).to(x.dtype)
            f_ndre = self.ndre_conv(ndre)
            feats.append(f_ndre)

        f_cat = torch.cat(feats, dim=1)
        # channel attention over all branches (Eq. 3-4)
        f_c = f_cat * self.ca(f_cat)

        # Spatial attention (Eq. 5 — extended για dual-index)
        m_main = self.main_sp(f_main)
        m_ndvi = self.ndvi_sp(f_ndvi)
        if self.use_ndre:
            # Two-level blend: first combine NDVI+NDRE με learnable β,
            # μετά combine main vs combined-vegetation με α.
            m_ndre = self.ndre_sp(f_ndre)
            m_veg = self.beta * m_ndvi + (1.0 - self.beta) * m_ndre
        else:
            m_veg = m_ndvi
        m_s = torch.sigmoid(self.alpha * m_main + (1.0 - self.alpha) * m_veg)
        f_c = f_c * m_s
        # project back + residual (Eq. 6)
        return self.proj(f_c) + x


class BottleneckSCSAModule(torch.nn.Module):
    """Spatial-Channel Synergistic Attention στο bottleneck (deepest encoder F4).

    SCSA attention mechanism by Si et al. (thesis ref [64]). The author's
    contribution is integrating it at the bottleneck of the SplitLawin pipeline.
    Εφαρμόζεται στο deepest feature (F4) μεταξύ encoder
    και Lawin decoder. Multi-scale DWConv branches (kernels 3/5/7/9) για spatial
    attention, μετά channel self-attention. Residual connection — preserves
    input channel count.
    """
    def __init__(self, channels: int):
        super().__init__()
        compressed = max(channels // 2, 4)
        # Compress (Eq. 9): Conv-BN-ReLU 2C → C concept
        self.compress = torch.nn.Sequential(
            torch.nn.Conv2d(channels, compressed, 1, bias=False),
            torch.nn.BatchNorm2d(compressed),
            torch.nn.ReLU(inplace=True),
        )
        # Split σε 4 groups + multi-scale DWConv για spatial attention map W
        # Uneven split αν compressed % 4 != 0 — τελευταίο group παίρνει το remainder
        c_split = compressed // 4
        c_rem = compressed - 3 * c_split
        self.split_sizes = [c_split, c_split, c_split, c_rem]
        self.dw3 = torch.nn.Conv2d(c_split, c_split, 3, padding=1, groups=c_split)
        self.dw5 = torch.nn.Conv2d(c_split, c_split, 5, padding=2, groups=c_split)
        self.dw7 = torch.nn.Conv2d(c_split, c_split, 7, padding=3, groups=c_split)
        self.dw9 = torch.nn.Conv2d(c_rem, c_rem, 9, padding=4, groups=c_rem)
        # Channel attention (Eq. 10): GAP → bottleneck conv → sigmoid
        ca_hidden = max(compressed // 4, 1)
        self.ca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(compressed, ca_hidden, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ca_hidden, compressed, 1),
            torch.nn.Sigmoid(),
        )
        # Restore (Eq. 11): C → 2C concept (compressed → channels) + residual
        self.restore = torch.nn.Sequential(
            torch.nn.Conv2d(compressed, channels, 1, bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        f_conv = self.compress(x)  # [B, compressed, H, W]
        # Multi-scale DWConv για spatial attention W (Eq. 10 context)
        groups = torch.split(f_conv, self.split_sizes, dim=1)
        sp = torch.cat([
            self.dw3(groups[0]),
            self.dw5(groups[1]),
            self.dw7(groups[2]),
            self.dw9(groups[3]),
        ], dim=1)
        W = torch.sigmoid(sp)
        SA = W * f_conv                # spatial-attended features
        CA = self.ca(SA)               # channel attention vector [B, compressed, 1, 1]
        f_scsa = SA * CA               # synergistic spatial+channel
        return self.restore(f_scsa) + x  # restore + residual (Eq. 11)


class CLIPCrossAttentionModule(torch.nn.Module):
    """CLIP-guided cross-attention για semantic decoder enhancement.

    Adapted από Papadeas et al. (CLIP Meets DINOv3, Eq. 11-17) στην αρχιτεκτονική
    SplitLawin+Lawin decoder. Εφαρμόζεται στα 256-channel decoder features ΠΡΙΝ
    το final 1×1 classification conv. Cross-attention μεταξύ visual queries και
    frozen CLIP text embeddings από class-specific prompts.

    Memory-efficient: φορτώνει CLIP text encoder ΜΟΝΟ at init για να υπολογίσει
    τα raw text embeddings, μετά τα cache-άρει ως buffer + discards το CLIP model.
    Στο training cost είναι ~0 (text embeddings constant ~6KB + lightweight attn).
    """
    def __init__(self, prompts, feat_dim: int = 256, text_dim: int = 512,
                 clip_model: str = "openai/clip-vit-base-patch16"):
        super().__init__()
        # Load CLIP text encoder ONCE, compute prompt embeddings, then DISCARD.
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
        except ImportError as e:
            raise ImportError(
                "transformers package required για CLIPCrossAttentionModule"
            ) from e

        tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        text_model = CLIPTextModel.from_pretrained(clip_model)
        text_model.eval()
        with torch.no_grad():
            tokens = tokenizer(list(prompts), padding=True, return_tensors="pt")
            text_outputs = text_model(**tokens)
            # pooler_output: [num_prompts, text_dim] (EOS-token representation)
            T_raw = text_outputs.pooler_output.detach().clone()
            # L2-normalize ώστε όλα τα prompts να έχουν unit magnitude — αποφεύγει
            # large-scale CLIP magnitudes που θα κυριαρχούσαν στο residual addition.
            T_raw = torch.nn.functional.normalize(T_raw, dim=-1)
        # Free CLIP (~63M params) — δεν χρειάζονται πια
        del tokenizer, text_model, text_outputs, tokens

        # Cache normalized CLIP embeddings ως buffer
        self.register_buffer("text_embeds", T_raw, persistent=True)
        self.num_classes = T_raw.shape[0]

        # Trainable text projection (Eq. 11): 512 → 256 with LayerNorm + ReLU
        self.text_proj = torch.nn.Sequential(
            torch.nn.Linear(text_dim, feat_dim),
            torch.nn.LayerNorm(feat_dim),
            torch.nn.ReLU(inplace=True),
        )

        # Cross-attention projections (Eq. 12-14)
        self.q_proj = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        self.k_proj = torch.nn.Linear(feat_dim, feat_dim)
        self.v_proj = torch.nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5

        # ZERO-INIT του v_proj (ControlNet-style): αρχικά attended=0 → output=x,
        # ώστε το CLIP module να αρχίζει σαν no-op και να εισάγεται σταδιακά
        # μέσω learning. Αποφεύγει class collapse από random init στα early epochs.
        torch.nn.init.zeros_(self.v_proj.weight)
        torch.nn.init.zeros_(self.v_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, C=256, H, W] visual features → enhanced via CLIP guidance.
        Residual connection (Eq. 17): out = x + attended."""
        B, C, H, W = x.shape

        # Project text embeddings (Eq. 11): [num_classes, feat_dim]
        # Cast σε x.dtype για AMP compatibility
        T_prime = self.text_proj(self.text_embeds.to(x.dtype))

        # Q from visual features (Eq. 12): [B, HW, C]
        q = self.q_proj(x).flatten(2).transpose(1, 2)

        # K, V from text features (Eq. 13-14): [1, num_classes, C]
        k = self.k_proj(T_prime).unsqueeze(0)
        v = self.v_proj(T_prime).unsqueeze(0)

        # Cross-attention (Eq. 15): [B, HW, num_classes]
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)

        # Attended features (Eq. 16): [B, HW, C] → [B, C, H, W]
        attended = torch.matmul(attn, v).transpose(1, 2).reshape(B, C, H, W)

        # Residual (Eq. 17)
        return x + attended


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
        # Optional ASVF input module (author's own module). Created AFTER super().__init__
        # ώστε να μην επηρεαστεί από το _init_weights pass του BaseLawin.
        if get_param(arch_params, "use_asvf", False):
            # Default indices για CIR+B+RE → expanded [NIR, G, R, B, RE]: NIR=0, R=2, RE=4
            self.asvf = ASVFInputModule(
                in_channels=_total_input_channels,
                nir_idx=get_param(arch_params, "asvf_nir_idx", 0),
                red_idx=get_param(arch_params, "asvf_red_idx", 2),
                main_dim=get_param(arch_params, "asvf_main_dim", 32),
                veg_dim=get_param(arch_params, "asvf_veg_dim", 16),
                alpha_init=get_param(arch_params, "asvf_alpha_init", 0.5),
                use_ndre=get_param(arch_params, "asvf_use_ndre", False),
                re_idx=get_param(arch_params, "asvf_re_idx", 4),
                beta_init=get_param(arch_params, "asvf_beta_init", 0.5),
            )
        else:
            self.asvf = None

        # Optional CLIP cross-attention στο decoder (Papadeas et al. inspired).
        # Φορτώνει frozen CLIP text encoder, encodes τα class prompts μία φορά,
        # μετά cache + discard. Inject στο LawinHead.clip_module.
        if get_param(arch_params, "use_clip", False):
            prompts = get_param(
                arch_params, "clip_prompts",
                ["background soil", "crop plant leaf", "weed plant"],
            )
            # Decoder embed_dim είναι 256 για B0 backbones (line ~26 του BaseLawin)
            feat_dim = 256 if 'B0' in get_param(arch_params, "backbone", 'MiT-B0') else 512
            self.decode_head.clip_module = CLIPCrossAttentionModule(
                prompts=prompts,
                feat_dim=feat_dim,
                clip_model=get_param(arch_params, "clip_model", "openai/clip-vit-base-patch16"),
            )

        # Optional Bottleneck-SCSA σε ένα ή ΠΕΡΙΣΣΟΤΕΡΑ encoder features.
        # SCSA mechanism by Si et al.; author's contribution = bottleneck integration.
        # Spatial+channel synergistic
        # attention για long-range dependencies.
        #   - use_scsa: True (backward compat) → defaults σε F4 μόνο
        #   - scsa_levels: [3, 4] → F3 + F4 (1-indexed, F1..F4)
        #   - scsa_levels: [2, 3, 4] → F2 + F3 + F4
        scsa_levels = get_param(arch_params, "scsa_levels", None)
        if scsa_levels is None and get_param(arch_params, "use_scsa", False):
            scsa_levels = [4]
        if scsa_levels:
            # 1-indexed (F1=1..F4=4) → 0-indexed (feat[0]..feat[3])
            self.scsa_indices = [lvl - 1 for lvl in scsa_levels]
            self.scsa_modules = torch.nn.ModuleList([
                BottleneckSCSAModule(self.backbone.channels[i])
                for i in self.scsa_indices
            ])
        else:
            self.scsa_indices = None
            self.scsa_modules = None

    def forward(self, x: Tensor) -> Tensor:
        if self.asvf is not None:
            x = self.asvf(x)
        main_channels = x[:, :self.main_channels, ::].contiguous()
        side_channels = x[:, self.main_channels:, ::].contiguous()
        first_feat_side = self.side_backbone(side_channels)
        first_feat_main = self.backbone.partial_forward(main_channels, slice(0, 1))
        first_feat = self.fusion((first_feat_main, first_feat_side))[0]
        feat = (first_feat,) + self.backbone.partial_forward(first_feat, slice(1, 4))
        # Optional Bottleneck-SCSA σε ένα ή περισσότερα encoder levels πριν decoder
        if self.scsa_modules is not None:
            feat = list(feat)
            for mod, idx in zip(self.scsa_modules, self.scsa_indices):
                feat[idx] = mod(feat[idx])
            feat = tuple(feat)
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
