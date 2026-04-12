#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST 卷积自编码器 (ConvAE) —— PyTorch 版 + 模型保存
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# ===================== 参数设置 =====================
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
LATENT_DIM = 64

MODEL_PATH = "conv_autoencoder_mnist.pth"   # 模型保存路径

# ===================== 数据 =====================
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ===================== 模型定义 =====================
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, LATENT_DIM),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def get_encoder(self):
        return self.encoder


# ===================== 可视化函数 =====================
def plot_reconstructions(model, test_loader, n=10):
    model.eval()
    with torch.no_grad():
        images = next(iter(test_loader))[0][:n].to(device)
        recon, _ = model(images)
    
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
    plt.suptitle("上排：原始图像    下排：ConvAE 重建图像")
    plt.show()


def plot_latent_space(encoder, test_loader):
    encoder.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            z = encoder(data)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    z = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=8)
    plt.colorbar(scatter)
    plt.xlabel("潜空间维度 1")
    plt.ylabel("潜空间维度 2")
    plt.title("ConvAE 2D 潜空间分布")
    plt.grid(True, alpha=0.3)
    plt.show()


# ===================== 主程序 =====================
if __name__ == "__main__":
    model = ConvAutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ==================== 训练 ====================
    print("🚀 开始训练 ConvAE...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, _ = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.6f}")

    print("\n✅ 训练完成！")

    # ==================== 保存模型 ====================
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'latent_dim': LATENT_DIM
    }, MODEL_PATH)
    
    print(f"✅ 模型已保存到: {MODEL_PATH}")

    # ==================== 可视化 ====================
    print("\n📊 可视化重建结果...")
    plot_reconstructions(model, test_loader)
    
    print("\n📈 可视化 2D 潜空间...")
    plot_latent_space(model.get_encoder(), test_loader)

    print("\n🎉 ConvAE 训练与保存完成！")