#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于卷积变分自编码器 + Transformer 分类器的图像分类算法实现
参考论文：《基于自编码器的图像分类算法设计与实现》

主要特点：
- 使用 Convolutional Variational AutoEncoder (ConvVAE) 进行特征提取
- 在潜空间上接 Transformer 分类器（符合论文第三章 3.4.3 Transformer分类器设计）
- 支持 Fashion-MNIST（便于调试）和 CUB-200-2011（细粒度分类）
- 包含详细中文注释，与论文结构和术语保持一致
- 模型自动保存，便于后续评估和部署
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# ===================== Windows 编码兼容 =====================
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("基于卷积变分自编码器与Transformer的图像分类算法实现")
print("参考论文：基于自编码器的图像分类算法设计与实现")
print("=" * 70)
print(f"使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}\n")


# ===================== 参数设置（可根据论文实验调整） =====================
BATCH_SIZE = 128
EPOCHS = 40
LR_VAE = 1e-3
LR_CLASSIFIER = 5e-4
LATENT_DIM = 128                    # 论文中建议的潜空间维度范围
KL_WEIGHT = 0.0001                  # KL散度权重（论文中提到可动态调整）
NUM_CLASSES = 10                    # Fashion-MNIST 为10类，CUB-200-2011为200类

# ===================== 数据加载（Fashion-MNIST） =====================
# 如需使用 CUB-200-2011，可自行替换数据集并调整图像大小为 224x224
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # 标准化有助于VAE训练稳定
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ===================== 1. 卷积变分自编码器 (ConvVAE) =====================
# 对应论文第三章 3.3 卷积变分自编码器设计与实现
class ConvVAE(nn.Module):
    """
    卷积变分自编码器 (Convolutional Variational AutoEncoder)
    - 编码器：提取图像的高级特征并输出分布参数 (mu, logvar)
    - 解码器：从潜在空间重构图像
    - 使用重参数化技巧保证可微分采样
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器：卷积特征提取 + 分布参数输出
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        # 输出分布参数（对应论文中VAE的 mu 和 logvar）
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # 解码器：从潜在向量重构图像
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()          # 输出范围 [0,1]
        )

    def encode(self, x):
        """编码器：输出分布参数"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧（对应论文公式 2-8 ~ 2-9）"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码器：从潜在向量重构图像"""
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z   # 返回 z 用于后续分类器


# ===================== 2. Transformer 分类器 =====================
# 对应论文第三章 3.4.3 Transformer分类器设计
class TransformerClassifier(nn.Module):
    """
    基于 Transformer 的分类器
    - 输入：VAE 输出的潜空间向量 z
    - 使用 Transformer Encoder 捕捉潜空间中的全局依赖关系
    - 最终输出类别概率
    """
    def __init__(self, latent_dim=LATENT_DIM, num_classes=10, num_heads=8, num_layers=4):
        super().__init__()
        self.latent_dim = latent_dim

        # 将单一潜向量扩展为序列形式（论文中提到的序列生成）
        self.seq_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, latent_dim))  # 可学习位置编码

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, z):
        # z shape: (batch_size, latent_dim) → 转为序列形式
        z = z.unsqueeze(1)                    # (B, 1, latent_dim)
        z = self.seq_proj(z) + self.pos_embedding
        z = self.transformer(z)
        z = z.mean(dim=1)                     # 全局平均池化
        return self.classifier(z)


# ===================== 3. 联合训练 =====================
if __name__ == "__main__":
    vae = ConvVAE(latent_dim=LATENT_DIM).to(device)
    classifier = TransformerClassifier(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

    optimizer_vae = optim.Adam(vae.parameters(), lr=LR_VAE)
    optimizer_clf = optim.Adam(classifier.parameters(), lr=LR_CLASSIFIER)

    print("开始联合训练 ConvVAE + Transformer 分类器...\n")

    for epoch in range(EPOCHS):
        vae.train()
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            # VAE 前向传播
            recon, mu, logvar, z = vae(images)
            vae_loss_val = vae_loss(recon, images, mu, logvar)

            # Transformer 分类器前向
            outputs = classifier(z)
            clf_loss = nn.CrossEntropyLoss()(outputs, labels)

            # 联合损失（可调节权重）
            loss = vae_loss_val * 0.1 + clf_loss   # 重构损失权重较低，重点优化分类

            # 反向传播
            optimizer_vae.zero_grad()
            optimizer_clf.zero_grad()
            loss.backward()
            optimizer_vae.step()
            optimizer_clf.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  Loss: {total_loss/len(train_loader):.4f}  Acc: {acc:.2f}%")

    # 保存模型
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'epoch': EPOCHS,
        'latent_dim': LATENT_DIM
    }, "cvae_transformer_fashionmnist.pth")

    print("\n模型训练完成并已保存！")

    # 测试集评估
    vae.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _, _, _, z = vae(images)
            outputs = classifier(z)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print(f"测试集分类准确率: {100 * correct / total:.2f}%")