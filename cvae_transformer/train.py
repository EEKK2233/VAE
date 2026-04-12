#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练代码：ConvVAE + Transformer 分类器（引用 models.py）
适用于 Fashion-MNIST 图像分类任务
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
from datetime import datetime
import os

# Windows 编码兼容
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

# ===================== 导入模型定义 =====================
from models import ConvVAE, TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# ===================== 参数设置 =====================
BATCH_SIZE = 2048
EPOCHS = 100
LR_VAE = 1e-3
LR_CLASSIFIER = 5e-4
LATENT_DIM = 128
NUM_CLASSES = 10

# ===================== 数据加载 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 模型初始化 =====================
vae = ConvVAE(latent_dim=LATENT_DIM).to(device)
classifier = TransformerClassifier(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

optimizer_vae = optim.Adam(vae.parameters(), lr=LR_VAE)
optimizer_clf = optim.Adam(classifier.parameters(), lr=LR_CLASSIFIER)

print("模型初始化完成，开始训练...\n")

# ===================== 训练循环 =====================
for epoch in range(EPOCHS):
    vae.train()
    classifier.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        # VAE 前向
        recon, mu, logvar, z = vae(images)
        vae_loss_val = nn.functional.mse_loss(recon, images, reduction='sum') + \
                       0.0001 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        # 分类器前向
        outputs = classifier(z)
        clf_loss = nn.CrossEntropyLoss()(outputs, labels)

        # 联合优化
        loss = vae_loss_val * 0.1 + clf_loss

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

print("\n训练完成！")

# ===================== 保存模型 =====================
timestamp = datetime.now().strftime("%Y%m%d%H%M")
filename = f"cvae_transformer_fashionmnist_{timestamp}.pth"
save_path = os.path.join("pth", filename)
torch.save({
    'vae_state_dict': vae.state_dict(),
    'classifier_state_dict': classifier.state_dict(),
    'epoch': EPOCHS,
    'latent_dim': LATENT_DIM
}, save_path)

print(f"模型已保存到: {save_path}")