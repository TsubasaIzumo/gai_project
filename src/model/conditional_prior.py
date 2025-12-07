import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class _FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        num_layers: int = 3,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_ch = min(max_channels, base_channels * (2 ** i))
            stride = 2 if i < num_layers - 1 else 1
            layers.append(_make_conv_block(channels, out_ch, stride=stride))
            channels = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriorNet(nn.Module):
    """
    条件先验网络 p_ψ(z | z_dirty)，输出对角高斯参数 μ_ψ, logσ²_ψ。
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        *,
        base_channels: int = 32,
        num_layers: int = 3,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = _FeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            max_channels=max_channels,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_dim = self.encoder.out_channels
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_dirty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(z_dirty)
        feat = self.pool(feat).flatten(1)
        mu = self.mu(feat)
        log_var = self.log_var(feat)
        log_var = torch.clamp(log_var, min=-8.0, max=8.0)
        return mu, log_var


class EncoderNet(nn.Module):
    """
    变分编码器 q_φ(z | x0, z_dirty)，输出对角高斯参数 μ_φ, logσ²_φ。
    """

    def __init__(
        self,
        in_channels_x: int,
        in_channels_dirty: int,
        latent_dim: int,
        *,
        base_channels: int = 32,
        num_layers: int = 3,
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        self.enc_x = _FeatureExtractor(
            in_channels=in_channels_x,
            base_channels=base_channels,
            num_layers=num_layers,
            max_channels=max_channels,
        )
        self.enc_dirty = _FeatureExtractor(
            in_channels=in_channels_dirty,
            base_channels=base_channels,
            num_layers=num_layers,
            max_channels=max_channels,
        )
        hidden_dim = self.enc_x.out_channels + self.enc_dirty.out_channels
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x0: torch.Tensor, z_dirty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_x = self.enc_x(x0)
        feat_zd = self.enc_dirty(z_dirty)

        feat_x = F.adaptive_avg_pool2d(feat_x, output_size=(1, 1)).flatten(1)
        feat_zd = F.adaptive_avg_pool2d(feat_zd, output_size=(1, 1)).flatten(1)

        feat = torch.cat([feat_x, feat_zd], dim=1)
        mu = self.mu(feat)
        log_var = self.log_var(feat)
        log_var = torch.clamp(log_var, min=-8.0, max=8.0)
        return mu, log_var


class LatentProjector(nn.Module):
    """
    将潜变量 z 映射到空间特征图，以便与 U-Net 输入通道拼接。
    """

    def __init__(self, latent_dim: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(latent_dim, out_channels)

    def forward(self, latent: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        h, w = spatial_size
        z_feat = self.linear(latent)
        z_feat = z_feat.view(latent.shape[0], self.out_channels, 1, 1)
        return z_feat.expand(-1, -1, h, w)


def kl_divergence_diag_gaussians(
    mu_q: torch.Tensor,
    log_var_q: torch.Tensor,
    mu_p: torch.Tensor,
    log_var_p: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算对角高斯之间的 KL(q || p)。
    返回 (逐样本 KL, 逐维 KL)。
    """
    var_q = torch.exp(log_var_q)
    var_p = torch.exp(log_var_p)
    diff = mu_p - mu_q

    kl_per_dim = 0.5 * (
        log_var_p - log_var_q + (var_q + diff.pow(2)) / (var_p + 1e-8) - 1.0
    )

    if free_bits > 0.0:
        min_val = free_bits
        kl_per_dim = torch.clamp(kl_per_dim, min=min_val)

    kl_per_sample = kl_per_dim.sum(dim=1)
    return kl_per_sample, kl_per_dim

