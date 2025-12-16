import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Try to import Mamba. If unavailable, use fallback (Transformer)
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print("[ssm_model] Using official mamba_ssm package ✅")
except ImportError:
    HAS_MAMBA = False
    print("[ssm_model] mamba_ssm not found. Will fallback to Transformer.")

class FrameEncoderViT(nn.Module):
    """
    Per-frame visual encoder using a Vision Transformer (ViT).
    """
    def __init__(self, vit_name: str = "vit_small_patch16_224", img_size: int = 160):
        super().__init__()
        self.img_size = img_size

        # timm ViT backbone, no classifier head
        # We allow img_size to be flexible (e.g. 160 instead of 224)
        self.vit = timm.create_model(
            vit_name,
            pretrained=True,
            num_classes=0,        # no classification head
            global_pool="avg",    # global average pool tokens
            img_size=img_size     # Force timm to accept our new resolution
        )

        # Infer embedding dimension
        self._embed_dim = self._infer_embed_dim()

    def _infer_embed_dim(self) -> int:
        self.vit.eval()
        with torch.no_grad():
            # Create dummy input with correct size
            dummy = torch.zeros(1, 3, self.img_size, self.img_size)
            out = self.vit(dummy)
        return out.shape[-1]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)  # [B*T, D]


class MambaBackbone(nn.Module):
    """
    Temporal backbone over frame embeddings.
    """
    def __init__(self, d_model: int, depth: int = 2, ssm_state_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.depth = depth

        if HAS_MAMBA:
            # Use official mamba_ssm package
            self.layers = nn.ModuleList(
                [Mamba(d_model=d_model, d_state=ssm_state_dim) for _ in range(depth)]
            )
            self.backbone_type = "official_mamba"
            print(f"[MambaBackbone] Active: Official Mamba SSM (depth={depth})")
        else:
            # Fallback to Transformer if SSM fails
            # We removed LightweightSSM to prevent accidental slow training
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model*4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.layers = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.backbone_type = "transformer"
            print(f"[MambaBackbone] Active: Transformer Fallback (depth={depth})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if self.backbone_type == "official_mamba":
            for layer in self.layers:
                x = layer(x)
            return x
        else:  # transformer
            return self.layers(x)


class MambaGestureRecognizer(nn.Module):
    """
    Full video gesture recognition model:

      Video: [B, T, 3, H, W]
        -> flatten time: [B*T, 3, H, W]
        -> ViT frame encoder: [B*T, D]
        -> reshape: [B, T, D]
        -> Mamba/SSM temporal backbone: [B, T, D]
        -> temporal pooling (mean): [B, D]
        -> linear classifier: [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        seq_len: int = 16,
        img_size: int = 160, # Optimized for 8GB VRAM
        vit_name: str = "vit_small_patch16_224",
        ssm_depth: int = 2,
        ssm_state_dim: int = 64,
        freeze_vit: bool = False, # Default to False for Full Fine-Tuning
    ):
        super().__init__()

        self.seq_len = seq_len
        self.img_size = img_size
        self.num_classes = num_classes

        # 1) Frame encoder (ViT)
        self.frame_encoder = FrameEncoderViT(vit_name=vit_name, img_size=img_size)
        d_model = self.frame_encoder.embed_dim
        
        print(f"[MambaGestureRecognizer] Initializing with:")
        print(f"  - num_classes: {num_classes}")
        print(f"  - seq_len: {seq_len}")
        print(f"  - img_size: {img_size}")
        print(f"  - vit_name: {vit_name}")
        print(f"  - d_model: {d_model}")
        print(f"  - ssm_depth: {ssm_depth}")
        print(f"  - ssm_state_dim: {ssm_state_dim}")

        # --- FULL FINE-TUNING STRATEGY ---
        print(f"[MambaGestureRecognizer] 🚀 FULL FINE-TUNING: Unfreezing ALL ViT layers.")
        
        # Force gradients ON for everything in the encoder
        for param in self.frame_encoder.parameters():
            param.requires_grad = True 

        # 2) Temporal backbone
        self.backbone = MambaBackbone(
            d_model=d_model,
            depth=ssm_depth,
            ssm_state_dim=ssm_state_dim,
        )

        # 3) Classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"[MambaGestureRecognizer] Model initialized successfully!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 3, H, W]
        returns logits: [B, num_classes]
        """
        B, T, C, H, W = x.shape
        assert C == 3, f"Expected 3 channels, got {C}"

        # Flatten time into batch: [B*T, 3, H, W]
        x_flat = x.reshape(B * T, C, H, W)

        # Encode frames via ViT: [B*T, D]
        frame_emb = self.frame_encoder(x_flat)  # [B*T, D]
        D = frame_emb.shape[-1]

        # Reshape back into videos: [B, T, D]
        seq = frame_emb.reshape(B, T, D)

        # Temporal Mamba/SSM backbone: [B, T, D]
        seq_out = self.backbone(seq)

        # Simple temporal pooling: mean over T -> [B, D]
        pooled = seq_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits