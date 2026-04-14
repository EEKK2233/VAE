#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion-MNIST 变分自编码器 (VAE) —— PyTorch 版（带Loss曲线）
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

# 解决 Windows GBK 编码问题
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

# ==================== 修复Matplotlib中文乱码 ====================
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Fashion-MNIST 变分自编码器 (VAE) - PyTorch 版")
print("=" * 60)
print(f"使用设备: {device}")
print(f"PyTorch 版本: {torch.__version__}\n")

# ===================== 参数设置 =====================
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
LATENT_DIM = 64

MODEL_PATH = "vae_fashion_mnist_pytorch.pth"

# ===================== 数据加载 =====================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Fashion-MNIST 数据加载完成 → 训练集 {len(train_dataset)} 张 | 测试集 {len(test_dataset)} 张\n")


# ===================== VAE 模型 =====================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, LATENT_DIM)
        self.fc_logvar = nn.Linear(64 * 7 * 7, LATENT_DIM)

        self.decoder_input = nn.Linear(LATENT_DIM, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(self.decoder_input(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ===================== 损失函数（拆分） =====================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


# ===================== 曲线绘制 =====================
def plot_loss_curves(total_losses, recon_losses, kl_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(total_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Divergence")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE 训练损失曲线")
    plt.legend()
    plt.grid(True)
    plt.show()


# ===================== 可视化函数 =====================
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
    plt.suptitle("Fashion-MNIST VAE 重建结果\n第一行：原始图像    第二行：重建图像")
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
    plt.suptitle("VAE 生成的新 Fashion-MNIST 样本")
    plt.show()


# ===================== 主程序 =====================
if __name__ == "__main__":
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ⭐ 新增：记录loss
    total_losses = []
    recon_losses = []
    kl_losses = []

    print("开始训练 Fashion-MNIST VAE...\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)
            recon_loss, kl_loss = vae_loss(recon, data, mu, logvar)

            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        total_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] "
              f"Total: {avg_loss:.2f} | Recon: {avg_recon:.2f} | KL: {avg_kl:.2f}")

    print("\n训练完成！")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'latent_dim': LATENT_DIM
    }, MODEL_PATH)

    print(f"模型已保存到: {MODEL_PATH}\n")

    # ⭐ 新增：画曲线
    plot_loss_curves(total_losses, recon_losses, kl_losses)

    print("生成可视化结果...")
    plot_reconstructions(model, test_loader, n=10)
    generate_samples(model, n=10)

    print("\nFashion-MNIST VAE 全部运行完成！")