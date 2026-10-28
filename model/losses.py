"""Loss functions for deviation network training."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DevNetZScoreLoss(nn.Module):
    """
    Original DevNet z-score-based deviation loss from KDD'19 paper.

    Algorithm:
    1. Create reference distribution: ref ~ N(0, 1) with size=5000
    2. Compute deviation score: dev = (pred - mean(ref)) / std(ref)
    3. Inlier loss: |dev| (push scores toward 0)
    4. Outlier loss: |max(margin - dev, 0)| (push scores above margin)

    Args:
        confidence_margin: Margin threshold (default: 5.0)
        ref_size: Size of reference distribution (default: 5000)
    """
    ref: torch.Tensor

    def __init__(self, confidence_margin: float=5.0, ref_size: int=5000):
        super().__init__()
        self.margin = confidence_margin
        self.register_buffer('ref', torch.randn(ref_size))

    def forward(self, y_pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze(1)
        ref_mean = self.ref.mean()
        ref_std = self.ref.std()
        deviation = (y_pred - ref_mean) / (ref_std + 1e-08)
        inlier_loss = torch.abs(deviation)
        outlier_loss = torch.abs(torch.clamp(self.margin - deviation, min=0.0))
        loss = (1 - labels) * inlier_loss + labels * outlier_loss
        return loss.mean()

class FeawadDevLoss(nn.Module):
    """
    FEAWAD Deviation Network loss function.

    For normal samples (y=0):
        L = |score| + reconstruction_error

    For anomaly samples (y=1):
        L = |max(margin - score, 0)| + |max(margin - reconstruction_error, 0)|

    Args:
        confidence_margin: Margin for hinge loss (default: 5.0)
    """

    def __init__(self, confidence_margin: float=5.0):
        super().__init__()
        self.margin = confidence_margin

    def forward(self, score: torch.Tensor, reconstruction_error: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        score = score.squeeze(1)
        inlier_loss = torch.abs(score) + reconstruction_error
        outlier_score_loss = torch.abs(torch.clamp(self.margin - score, min=0.0))
        outlier_recon_loss = torch.abs(torch.clamp(self.margin - reconstruction_error, min=0.0))
        outlier_loss = outlier_score_loss + outlier_recon_loss
        loss = (1 - labels) * inlier_loss + labels * outlier_loss
        return loss.mean()

class FeawadDevNetLoss(nn.Module):
    """
    Hybrid loss: FEAWAD reconstruction error + DevNet z-score deviation.

    For normal samples (y=0):
        L = |deviation(score)| + reconstruction_error

    For anomaly samples (y=1):
        L = |max(margin - deviation(score), 0)| + |max(margin - reconstruction_error, 0)|

    Args:
        confidence_margin: Margin threshold (default: 5.0)
        ref_size: Size of reference distribution for z-score (default: 5000)
    """
    ref: torch.Tensor

    def __init__(self, confidence_margin: float=5.0, ref_size: int=5000):
        super().__init__()
        self.margin = confidence_margin
        self.register_buffer('ref', torch.randn(ref_size))

    def forward(self, score: torch.Tensor, reconstruction_error: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if score.dim() > 1:
            score = score.squeeze(1)
        ref_mean = self.ref.mean()
        ref_std = self.ref.std()
        deviation = (score - ref_mean) / (ref_std + 1e-08)
        inlier_loss = torch.abs(deviation) + reconstruction_error
        outlier_score_loss = torch.abs(torch.clamp(self.margin - deviation, min=0.0))
        outlier_recon_loss = torch.abs(torch.clamp(self.margin - reconstruction_error, min=0.0))
        outlier_loss = outlier_score_loss + outlier_recon_loss
        loss = (1 - labels) * inlier_loss + labels * outlier_loss
        return loss.mean()
