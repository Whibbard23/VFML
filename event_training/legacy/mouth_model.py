# event_training/models/mouth_model.py
"""
ResNet18 early-fusion model that accepts 4-channel input (RGB + motion).
Uses torchvision weights enum when available to avoid deprecation warnings.
"""
import torch
import torch.nn as nn
import torchvision
from packaging import version

def _load_resnet18(pretrained: bool):
    """
    Return a torchvision resnet18 model instance.
    Uses the new `weights` enum when available (torchvision >= 0.13).
    """
    tv = torchvision
    tv_version = getattr(tv, "__version__", "0.0.0")
    try:
        # torchvision >= 0.13 exposes ResNet18_Weights
        if version.parse(tv_version) >= version.parse("0.13"):
            # import the weights enum
            try:
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                model = tv.models.resnet18(weights=weights)
                return model
            except Exception:
                # fallback to legacy call below
                pass
        # legacy fallback (older torchvision)
        model = tv.models.resnet18(pretrained=pretrained)
        return model
    except Exception:
        # final fallback: construct without pretrained weights
        return tv.models.resnet18(pretrained=False)

class ResNet18EarlyFusion(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # load base resnet18 using helper that avoids deprecation warnings
        base = _load_resnet18(pretrained)
        # adapt conv1 to 4 channels
        orig_w = base.conv1.weight.data.clone()  # (64,3,7,7)
        conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3, :, :] = orig_w
            conv1.weight[:, 3:4, :, :] = orig_w.mean(dim=1, keepdim=True)
        base.conv1 = conv1
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, 1)
        self.model = base

    def forward(self, x):
        logits = self.model(x).squeeze(1)
        return logits

if __name__ == "__main__":
    # smoke test
    m = ResNet18EarlyFusion(pretrained=False)
    x = torch.randn(2,4,224,224)
    y = m(x)
    print("Output shape:", y.shape)
