#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUB-200-2011 数据集训练代码：ConvVAE + Transformer 分类器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import sys

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

from models_cub import ConvVAE_CUB, TransformerClassifier_CUB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# ===================== 参数设置 =====================
BATCH_SIZE = 32                     # CUB 数据集分辨率高，batch_size 需减小
EPOCHS = 50
LR_VAE = 5e-4
LR_CLASSIFIER = 1e-3
LATENT_DIM = 512
NUM_CLASSES = 200

# ===================== 数据增强 + 加载 =====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 注意：CUB-200-2011 需要自行下载并使用自定义 Dataset，这里假设你已准备好
# 如果你使用的是标准 torchvision 无法直接加载，可使用以下方式（推荐）
train_dataset = torchvision.datasets.ImageFolder(root='data/CUB_200_2011/train', transform=train_transform)
test_dataset = torchvision.datasets.ImageFolder(root='data/CUB_200_2011/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ===================== 模型 =====================
vae = ConvVAE_CUB(latent_dim=LATENT_DIM).to(device)
classifier = TransformerClassifier_CUB(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

optimizer_vae = optim.Adam(vae.parameters(), lr=LR_VAE)
optimizer_clf = optim.Adam(classifier.parameters(), lr=LR_CLASSIFIER)

print("开始在 CUB-200-2011 上训练 ConvVAE + Transformer 分类器...\n")

for epoch in range(EPOCHS):
    vae.train()
    classifier.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        recon, mu, logvar, z = vae(images)
        vae_loss_val = nn.functional.mse_loss(recon, images, reduction='sum') + \
                       0.00005 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        outputs = classifier(z)
        clf_loss = nn.CrossEntropyLoss()(outputs, labels)

        loss = vae_loss_val * 0.05 + clf_loss     # 重构损失权重较低，重点优化分类

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
}, "cvae_transformer_cub200.pth")

print("\nCUB-200-2011 模型训练完成并已保存！")