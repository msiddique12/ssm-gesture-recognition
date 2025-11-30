import torch
import torch.nn as nn

try:
    import timm  # for Vision Transformer backbone
except ImportError:
    timm = None


class FrameEncoderViT(nn.Module):
    """
    Encodes individual frames using a pretrained Vision Transformer.
    Input:  [B*T, 3, H, W]
    Output: [B*T, D]   (D = embedding dim)
    """

    def __init__(self, model_name: str = "vit_small_patch16_224", pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for FrameEncoderViT. "
                "Install it with `pip install timm`."
            )

        # num_classes=0 makes timm return the final embedding instead of logits
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.vit.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, 3, H, W]
        return self.vit(x)  # [B*T, D]


class MambaBackbone(nn.Module):
    """
    Temporal backbone that *tries* to use Mamba if available.
    If mamba_ssm is not installed, falls back to a TransformerEncoder.

    Input / Output: [B, T, D]
    """

    def __init__(self, d_model: int, depth: int = 2, d_state: int = 64):
        super().__init__()
        self.use_mamba = False

        try:
            # If you install mamba-ssm, this path will be used.
            from mamba_ssm.models.mamba import Mamba  # type: ignore

            self.use_mamba = True
            self.layers = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                 for _ in range(depth)]
            )
        except Exception:
            # Fallback: standard Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=4 * d_model,
                batch_first=True,
            )
            self.layers = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if self.use_mamba:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # TransformerEncoder handles the whole stack
            return self.layers(x)


class MambaGestureRecognizer(nn.Module):
    """
    Full gesture recognizer:

        Frames [B, T, 3, H, W]
          -> ViT frame encoder [B, T, D]
          -> SSM/Transformer backbone [B, T, D]
          -> temporal pooling [B, D]
          -> classifier [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        seq_len: int = 16,
        img_size: int = 224,
        vit_name: str = "vit_small_patch16_224",
        ssm_depth: int = 2,
        ssm_state_dim: int = 64,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.img_size = img_size

        # 1) Per-frame ViT encoder
        self.frame_encoder = FrameEncoderViT(
            model_name=vit_name,
            pretrained=True,
        )
        d_model = self.frame_encoder.embed_dim  # feature dim D

        # 2) Temporal backbone: Mamba if available, else Transformer
        self.backbone = MambaBackbone(
            d_model=d_model,
            depth=ssm_depth,
            d_state=ssm_state_dim,
        )

        # 3) Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 3, H, W]
        returns: logits [B, num_classes]
        """
        B, T, C, H, W = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"

        # Merge batch and time to encode frames through ViT
        x = x.view(B * T, C, H, W)    # [B*T, 3, H, W]
        frame_feats = self.frame_encoder(x)  # [B*T, D]
        D = frame_feats.shape[-1]

        # Restore [B, T, D]
        frame_feats = frame_feats.view(B, T, D)

        # Temporal modeling with SSM / Transformer
        seq_feats = self.backbone(frame_feats)  # [B, T, D]

        # Simple temporal pooling: mean over time
        pooled = seq_feats.mean(dim=1)         # [B, D]

        # Class logits
        logits = self.head(pooled)             # [B, num_classes]
        return logits
