#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST 变分自编码器 (VAE) —— PyTorch 版
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
LATENT_DIM = 64


transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform),
                         batch_size=BATCH_SIZE, shuffle=False)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*7*7, LATENT_DIM)
        self.fc_logvar = nn.Linear(64*7*7, LATENT_DIM)

        self.decoder_input = nn.Linear(LATENT_DIM, 64*7*7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
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
        return recon, mu, logvar

    def get_encoder(self):
        return self.encoder


def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ===================== 主程序 =====================
if __name__ == "__main__":
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 开始训练 VAE...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss/len(train_loader):.2f}")

    print("\n✅ VAE 训练完成！")
    
    # 可视化重建
    model.eval()
    with torch.no_grad():
        test_images = next(iter(test_loader))[0][:10].to(device)
        recon, _, _ = model(test_images)
    # ...（可视化代码与 ConvAE 类似，可自行复制上面的 plot_reconstructions 函数）

    print("VAE 已训练完成，可用于生成新样本！")