#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版训练脚本 - ConvVAE (带CBAM) + TransformerClassifier
已优化：BCE Loss + 降低VAE权重 + 后期冻结编码器 + 加强数据增强
适配 RTX 4060 8GB
"""

import sys
import io

# ===================== 解决 Windows 中文乱码 =====================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
from datetime import datetime
import torch.multiprocessing

# ===================== Windows 多进程修复 =====================
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | GPU: RTX 4060 8GB\n")

    # ===================== 参数设置 =====================
    BATCH_SIZE_VAE = 1024
    BATCH_SIZE_FINETUNE = 768

    EPOCHS_VAE = 50                # 增加到50
    EPOCHS_FINETUNE = 70           # 增加到70

    LR_VAE_STAGE1 = 1e-3
    LR_VAE_STAGE2 = 5e-5
    LR_CLASSIFIER = 1.2e-3         # 提高分类器学习率

    LATENT_DIM = 128
    NUM_CLASSES = 10
    WEIGHT_DECAY = 1e-4

    # β-VAE 参数
    BETA_START = 0.0
    BETA_END = 0.0005              # 降低最终β
    BETA_WARMUP_EPOCHS = 20

    # 创建目录
    os.makedirs("pth", exist_ok=True)
    os.makedirs("recon_samples", exist_ok=True)

    # ===================== 数据增强（加强） =====================
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )

    val_size = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    vae_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_VAE, shuffle=True, 
                                  num_workers=0, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_FINETUNE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_FINETUNE, shuffle=False, 
                            num_workers=0, pin_memory=True)

    # ===================== 模型 =====================
    from models import ConvVAE, TransformerClassifier

    vae = ConvVAE(latent_dim=LATENT_DIM).to(device)
    classifier = TransformerClassifier(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)

    optimizer_vae = optim.AdamW(vae.parameters(), lr=LR_VAE_STAGE1, weight_decay=WEIGHT_DECAY)
    optimizer_clf = optim.AdamW(classifier.parameters(), lr=LR_CLASSIFIER, weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler()

    print("模型初始化完成，开始两阶段训练...\n")

    # ===================== β 退火 =====================
    def get_beta(epoch):
        if epoch < BETA_WARMUP_EPOCHS:
            return BETA_START + (BETA_END - BETA_START) * (epoch / BETA_WARMUP_EPOCHS)
        return BETA_END

    # ===================== 保存重建图像 =====================
    def save_reconstruction(epoch, stage=""):
        vae.eval()
        with torch.no_grad():
            images, _ = next(iter(val_loader))
            images = images[:8].to(device)
            recon, _, _, _ = vae(images)
            comparison = torch.cat([images, recon])
            save_image(comparison, f"recon_samples/{stage}epoch_{epoch:03d}.png", nrow=8, normalize=True)
        vae.train()

    # ===================== 阶段1：VAE 预训练 =====================
    print("=== 阶段1：VAE 预训练（使用 BCE Loss） ===")
    for epoch in range(EPOCHS_VAE):
        vae.train()
        total_loss = 0.0
        beta = get_beta(epoch)

        for images, _ in tqdm(vae_train_loader, desc=f"VAE Epoch {epoch+1}/{EPOCHS_VAE}"):
            images = images.to(device)

            with torch.amp.autocast(device_type='cuda'):
                recon, mu, logvar, _ = vae(images)
                recon_loss = nn.functional.binary_cross_entropy(recon, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss

            optimizer_vae.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            scaler.step(optimizer_vae)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(vae_train_loader)
        print(f"VAE Epoch [{epoch+1:2d}/{EPOCHS_VAE}]  Loss: {avg_loss:.2f}  β={beta:.5f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS_VAE - 1:
            save_reconstruction(epoch + 1, stage="vae_")

    print("VAE 预训练完成！开始阶段2联合微调...\n")

    # ===================== 阶段2：联合微调 =====================
    print("=== 阶段2：VAE + Classifier 联合微调 ===")
    best_val_acc = 0.0
    patience = 12
    counter = 0

    for epoch in range(EPOCHS_FINETUNE):
        vae.train()
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{EPOCHS_FINETUNE}"):
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type='cuda'):
                recon, mu, logvar, z = vae(images)
                outputs = classifier(z)

                recon_loss = nn.functional.binary_cross_entropy(recon, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                clf_loss = nn.CrossEntropyLoss()(outputs, labels)

                # 关键改进：VAE 权重动态降低 + 后期冻结编码器
                vae_weight = 0.005 if epoch >= 40 else 0.012
                beta = 0.0003
                loss = vae_weight * (recon_loss + beta * kl_loss) + clf_loss

            optimizer_vae.zero_grad()
            optimizer_clf.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            scaler.step(optimizer_vae)
            scaler.step(optimizer_clf)
            scaler.update()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total

        # 验证
        vae.eval()
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, _, z = vae(images)
                outputs = classifier(z)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Finetune Epoch [{epoch+1:2d}/{EPOCHS_FINETUNE}]  Loss: {avg_loss:.4f}  "
              f"TrainAcc: {acc:.2f}%  ValAcc: {val_acc:.2f}%")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS_FINETUNE - 1:
            save_reconstruction(epoch + 1, stage="finetune_")

        # 后期冻结 VAE 编码器
        if epoch == 39:
            print("→ 从 Epoch 40 开始冻结 VAE 编码器，只微调 decoder + classifier")
            for param in vae.encoder.parameters():
                param.requires_grad = False

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            save_path = f"pth/cvae_transformer_fashionmnist_best_{timestamp}.pth"
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'latent_dim': LATENT_DIM
            }, save_path)
            print(f"→ 新最佳模型已保存！Val Acc: {val_acc:.2f}%")
        else:
            counter += 1
            if counter >= patience:
                print("Early Stopping 触发！")
                break

    print("\n训练全部完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型保存在 pth/ 目录下")