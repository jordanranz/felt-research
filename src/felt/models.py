import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = {"audio": [0, 1, 2], "beat": [3, 4], "combined": [0, 1, 2, 3, 4]}
MODEL_VERSIONS = ("causal-tcn-v1", "warmup-sigmoid-v2", "warmup-linear-v2")


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.left_padding = 4 * dilation
        self.conv = nn.Conv1d(inputs, outputs, 5, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class EnvelopeModel(nn.Module):
    """29-frame receptive field; no normalization across time or future padding."""

    def __init__(self, mode="combined", hidden=32, version="causal-tcn-v1"):
        super().__init__()
        if version not in MODEL_VERSIONS:
            raise ValueError(f"Unsupported model version: {version}")
        self.version = version
        self.channels = CHANNELS[mode]
        self.network = nn.Sequential(
            CausalConv(len(self.channels), hidden, 1),
            nn.ReLU(),
            CausalConv(hidden, hidden, 2),
            nn.ReLU(),
            CausalConv(hidden, hidden, 4),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, 1),
            nn.Hardtanh(0, 1) if version == "warmup-linear-v2" else nn.Sigmoid(),
        )
        if version == "warmup-linear-v2":
            nn.init.constant_(self.network[-2].bias, 0.1)

    def forward(self, x):
        x = x[:, self.channels]
        if self.version != "causal-tcn-v1":
            # Assume silent prehistory at a fresh stream. Warm up hidden activations
            # with real zero input rather than padding each hidden layer with zeros.
            return self.network(F.pad(x, (28, 0))).squeeze(1)[:, 28:]
        return self.network(x).squeeze(1)


def loss(prediction, target, kind="active-weighted-mse-v1"):
    errors = (prediction - target) ** 2
    if kind == "active-weighted-mse-v1":
        weights = 1 + 4 * (target > 0.1).float()
        return torch.mean(weights * errors)
    if kind not in ("region-balanced-mse-v1", "region-startup-mse-v1"):
        raise ValueError(f"Unknown loss: {kind}")
    # Normalize each region separately so clip occupancy does not set its influence.
    regions = (
        (target > 0.1, 0.5),
        (target < 0.02, 0.35),
        ((target >= 0.02) & (target <= 0.1), 0.15),
    )
    value = sum(
        weight * (errors * mask).sum() / mask.sum().clamp_min(1) for mask, weight in regions
    )
    if kind == "region-startup-mse-v1":
        value = value + errors[:, :29].mean()
    return value
