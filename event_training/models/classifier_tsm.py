import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TemporalShift(nn.Module):
    """
    Temporal Shift Module applied on feature maps.
    Input shape: (B, T, C, H, W)
    Shifts a fraction `shift_ratio` of channels forward/backward in time.
    """
    def __init__(self, n_segment, shift_ratio=0.25):
        super().__init__()
        self.n_segment = n_segment
        self.shift_ratio = shift_ratio

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        fold = int(C * self.shift_ratio)
        if fold == 0 or T == 1:
            return x
        # reshape to (B*T, C, H, W) then (B, T, C, H, W) already
        x = x.view(B, T, C, H, W)
        out = torch.zeros_like(x)
        # shift forward
        out[:, :-1, :fold, :, :] = x[:, 1:, :fold, :, :]
        # shift backward
        out[:, 1:, fold:2*fold, :, :] = x[:, :-1, fold:2*fold, :, :]
        # keep the rest
        out[:, :, 2*fold:, :, :] = x[:, :, 2*fold:, :, :]
        return out

class ResNetBackbone(nn.Module):
    """
    ResNet18 backbone that returns feature maps (no final fc).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet18(pretrained=pretrained)
        # keep layers up to layer4 (exclude avgpool and fc)
        self.stem = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool
        )
        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4
        # feature dimension after layer4
        self.out_channels = 512

    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # shape (B*T, F, H', W')

class ClassifierTSM(nn.Module):
    """
    Full classifier: per-frame ResNet backbone -> TSM on feature maps -> pooling -> head
    """
    def __init__(self, num_classes, n_segment=8, pretrained_backbone=True, shift_ratio=0.25, dropout=0.3):
        super().__init__()
        self.n_segment = n_segment
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        self.tsm = TemporalShift(n_segment=n_segment, shift_ratio=shift_ratio)
        feat_dim = self.backbone.out_channels  # 512 for ResNet18
        self.pool_spatial = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, clips):
        # clips: (B, T, C, H, W)
        B, T, C, H, W = clips.shape
        assert T == self.n_segment, f"Expected T={self.n_segment}, got {T}"
        # reshape to (B*T, C, H, W) to run backbone per-frame
        x = clips.view(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, F, H', W')
        Fdim = feats.shape[1]
        Hf, Wf = feats.shape[2], feats.shape[3]
        # reshape back to (B, T, F, H', W')
        feats = feats.view(B, T, Fdim, Hf, Wf)
        # apply TSM (in-place-like behavior avoided)
        feats = self.tsm(feats)
        # spatial pool -> (B, T, F)
        feats = feats.view(B * T, Fdim, Hf, Wf)
        feats = self.pool_spatial(feats).view(B, T, Fdim)
        # temporal aggregation (mean)
        clip_feat = feats.mean(dim=1)  # (B, F)
        clip_feat = self.dropout(clip_feat)
        logits = self.classifier(clip_feat)  # (B, num_classes)
        return logits
