#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUB-200-2011 测试代码 - 高级指标评估
包含 Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, 混淆矩阵等
"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tqdm import tqdm

# Windows 编码兼容
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# ===================== 参数设置 =====================
BATCH_SIZE = 32
MODEL_PATH = "cvae_transformer_cub200.pth"   # 请修改为你的实际模型路径
NUM_CLASSES = 200
IMAGE_SIZE = 224

# ===================== 数据加载 =====================
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 注意：请将你的 CUB-200-2011 测试集整理成 ImageFolder 格式
test_dataset = torchvision.datasets.ImageFolder(
    root='../data/CUB_200_2011/test',
    transform=test_transform,
    download=True
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 如果你有类别名称列表，可以在这里定义（可选）
# class_names = [...]  # 200个鸟类名称，可自行加载

# ===================== 导入模型定义 =====================
from models_cub import ConvVAE_CUB, TransformerClassifier_CUB

# ===================== 加载模型 =====================
print("正在加载 CUB-200-2011 模型...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

vae = ConvVAE_CUB(latent_dim=512).to(device)
classifier = TransformerClassifier_CUB(latent_dim=512, num_classes=NUM_CLASSES).to(device)

vae.load_state_dict(checkpoint['vae_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

vae.eval()
classifier.eval()

print("模型加载成功！开始评估...\n")

# ===================== 测试推理 =====================
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="测试中"):
        images = images.to(device)
        _, _, _, z = vae(images)
        outputs = classifier(z)
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ===================== 基础指标 =====================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print("=" * 80)
print("               CUB-200-2011 测试集高级评估结果")
print("=" * 80)
print(f"准确率 (Accuracy)     : {accuracy*100:6.2f}%")
print(f"精确率 (Precision)    : {precision*100:6.2f}%")
print(f"召回率 (Recall)       : {recall*100:6.2f}%")
print(f"F1 分数 (Macro)       : {f1*100:6.2f}%")
print("=" * 80)

# 详细分类报告（每类指标）
print("\n每类详细分类报告（前20类示例）：")
print(classification_report(all_labels, all_preds, digits=4)[:2000])  # 只打印部分，避免过长

# ===================== ROC 曲线 & AUC =====================
print("\n正在绘制 ROC 曲线...")
all_labels_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))

plt.figure(figsize=(14, 10))
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Class {i} (AUC={roc_auc:.3f})' if i < 10 else "")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CUB-200-2011 ConvVAE+Transformer 多类 ROC 曲线 (One-vs-Rest)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"平均 ROC-AUC: {np.mean(aucs):.4f}")

# ===================== PR 曲线 & mAP =====================
print("\n正在绘制 PR 曲线...")
plt.figure(figsize=(14, 10))
aps = []

for i in range(NUM_CLASSES):
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
    ap = average_precision_score(all_labels_bin[:, i], all_probs[:, i])
    aps.append(ap)
    if i < 10:  # 只绘制前10类，避免图像过乱
        plt.plot(recall_curve, precision_curve, lw=2, label=f'Class {i} (AP={ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('CUB-200-2011 多类 PR 曲线')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"平均 PR-AUC (mAP): {np.mean(aps):.4f}")

# ===================== 混淆矩阵 =====================
print("\n正在绘制混淆矩阵...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')   # annot=False 避免200类太密集
plt.title('CUB-200-2011 ConvVAE + Transformer 分类器混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.show()

print("\nCUB-200-2011 所有评估指标计算完成！")