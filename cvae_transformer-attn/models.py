#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
升级版模型：ConvVAE + TransformerClassifier
新增：CBAM (通道+空间注意力) + 增强型 Transformer 自注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== CBAM 注意力模块 =====================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ===================== 升级版 ConvVAE =====================
class ConvVAE(nn.Module):
    """卷积变分自编码器 + CBAM 注意力"""
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器（添加 CBAM）
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAM(32),                     # ← 新增通道+空间注意力

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAM(64),                     # ← 新增通道+空间注意力

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # 解码器（可选择性添加注意力，这里保持轻量）
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAM(32),                     # ← 解码器也可加注意力（可选）
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


# ===================== 升级版 TransformerClassifier =====================
class TransformerClassifier(nn.Module):
    """基于 Transformer 的分类器（增强自注意力）"""
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim

        # 投影 + 位置编码
        self.seq_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, latent_dim))

        # 更强的 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)  # 层数从4→6（推荐）

        # 分类头（轻微加强）
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, z):
        z = z.unsqueeze(1)                    # (B, 1, D)
        z = self.seq_proj(z) + self.pos_embedding
        z = self.transformer(z)
        z = z.mean(dim=1)                     # 全局平均池化（可换成 CLS token）
        return self.classifier(z)