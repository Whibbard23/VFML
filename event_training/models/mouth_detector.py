# event_training/models/mouth_detector.py
"""
Mouth detector model wrapper.

- Uses torchvision ResNet18 as a backbone and the modern `weights=` API when available.
- Compatible with older torchvision by falling back to `weights=None`.
- Accepts `num_classes` (default 1). If the project's training code instantiates
  MouthDetector() without args, that still works.
- Forward accepts either:
    - [B, T, C, H, W] clips: frames are passed through the backbone individually
      and their features are averaged across time before the final head.
    - [B, C, H, W] single-frame inputs: processed directly.
- Produces logits (shape [B] for num_classes==1, or [B, num_classes] otherwise).
"""

from __future__ import annotations
import torch
import torch.nn as nn

# Build backbone using the modern weights enum when available; otherwise avoid deprecated API.
def _make_resnet18_backbone(pretrained_enum=True):
    try:
        # Preferred modern API
        from torchvision.models import resnet18, ResNet18_Weights
        if pretrained_enum:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        backbone = resnet18(weights=weights)
    except Exception:
        # Older torchvision: use weights=None to avoid deprecated 'pretrained' argument
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
    return backbone

class MouthDetector(nn.Module):
    def __init__(self, num_classes: int = 1, backbone_pretrained: bool = True):
        """
        Args:
            num_classes: number of output classes (1 for binary mouth-onset score).
            backbone_pretrained: whether to request pretrained weights via the modern enum.
                                If the enum is unavailable, code falls back to weights=None.
        """
        super().__init__()
        # Create a ResNet18 backbone and remove its final fc so we can use features.
        backbone = _make_resnet18_backbone(pretrained_enum=backbone_pretrained)
        # Save feature dimension before replacing fc
        feat_dim = backbone.fc.in_features if hasattr(backbone.fc, "in_features") else 512
        # Replace backbone.fc with identity so backbone returns features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

        # Simple head: linear classifier/regressor
        if num_classes == 1:
            self.head = nn.Linear(self.feat_dim, 1)
        else:
            self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Accepts:
            x: [B, T, C, H, W] or [B, C, H, W]

        Returns:
            logits: [B] if num_classes==1 else [B, num_classes]
        """
        # Single-frame input
        if x.dim() == 4:
            # [B, C, H, W]
            feats = self.backbone(x)  # [B, feat_dim]
            logits = self.head(feats)
            if logits.shape[-1] == 1:
                return logits.view(-1)
            return logits

        # Clip input: [B, T, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Process frames individually through backbone by flattening batch*time
            x_flat = x.view(B * T, C, H, W)
            feats_flat = self.backbone(x_flat)  # [B*T, feat_dim]
            feats = feats_flat.view(B, T, self.feat_dim)  # [B, T, feat_dim]
            # Temporal pooling: mean over frames
            pooled = feats.mean(dim=1)  # [B, feat_dim]
            logits = self.head(pooled)
            if logits.shape[-1] == 1:
                return logits.view(-1)
            return logits

        raise ValueError(f"Unsupported input tensor shape: {tuple(x.shape)}")
