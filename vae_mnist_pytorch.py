#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST 变分自编码器 (VAE) —— PyTorch 版 + 模型保存（Windows 兼容版）
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import sys

# ===================== 解决 Windows 编码问题 =====================
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MNIST 变分自编码器 (VAE) - PyTorch 版")
print("=" * 60)
print(f"使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}\n")

# ===================== 参数设置 =====================
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
LATENT_DIM = 64

MODEL_PATH = "vae_mnist_pytorch.pth"

# ===================== 数据 =====================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ===================== VAE 模型 =====================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, LATENT_DIM)
        self.fc_logvar = nn.Linear(64 * 7 * 7, LATENT_DIM)

        self.decoder_input = nn.Linear(LATENT_DIM, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
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
        return recon, mu, logvar


# ===================== 损失函数 =====================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# ===================== 可视化 =====================
def plot_reconstructions(model, test_loader, n=10):
    model.eval()
    with torch.no_grad():
        images = next(iter(test_loader))[0][:n].to(device)
        recon, _, _ = model(images)
    
    images = images.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon[i][0], cmap='gray')
        plt.axis('off')
    plt.suptitle("上排：原始图像    下排：VAE 重建图像")
    plt.show()


def generate_samples(model, n=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, LATENT_DIM).to(device)
        samples = model.decode(z)
    
    samples = samples.cpu().numpy()
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(samples[i][0], cmap='gray')
        plt.axis('off')
    plt.suptitle("VAE 生成的新 MNIST 数字样本")
    plt.show()


# ===================== 主程序 =====================
if __name__ == "__main__":
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练 VAE...\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  Loss: {avg_loss:.4f}")

    print("\n训练完成！")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'latent_dim': LATENT_DIM
    }, MODEL_PATH)
    
    print(f"模型已保存到: {MODEL_PATH}\n")

    # 可视化
    print("正在生成可视化结果...")
    plot_reconstructions(model, test_loader)
    generate_samples(model, n=10)

    print("\nVAE 全部运行完成！")