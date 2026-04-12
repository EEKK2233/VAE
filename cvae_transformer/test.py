#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级测试脚本 - 使用 models.py 中的模型定义
计算 Accuracy, Precision, Recall, F1, ROC, PR 曲线等指标
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Windows 编码兼容
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 导入模型定义 =====================
from models import ConvVAE, TransformerClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# ===================== 参数 =====================
BATCH_SIZE = 128
MODEL_PATH = "pth/cvae_transformer_fashionmnist_202604121758.pth"   # 
NUM_CLASSES = 10

# ===================== 数据加载 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ===================== 加载模型 =====================
print("正在加载模型...")

checkpoint = torch.load(MODEL_PATH, map_location=device)

vae = ConvVAE(latent_dim=128).to(device)
classifier = TransformerClassifier(latent_dim=128, num_classes=NUM_CLASSES).to(device)

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
    for images, labels in test_loader:
        images = images.to(device)
        _, _, _, z = vae(images)           # 获取潜空间特征
        outputs = classifier(z)
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)


# ===================== 计算指标 =====================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

print("=" * 70)
print("                  测试集高级评估结果")
print("=" * 70)
print(f"准确率 (Accuracy)     : {accuracy*100:6.2f}%")
print(f"精确率 (Precision)    : {precision*100:6.2f}%")
print(f"召回率 (Recall)       : {recall*100:6.2f}%")
print(f"F1 分数               : {f1*100:6.2f}%")
print("=" * 70)

print("\n每类详细报告：")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


# ===================== ROC & PR 曲线 =====================
all_labels_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))

# ROC 曲线
plt.figure(figsize=(12, 10))
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲线 (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# PR 曲线
plt.figure(figsize=(12, 10))
for i in range(NUM_CLASSES):
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels_bin[:, i], all_probs[:, i])
    ap = average_precision_score(all_labels_bin[:, i], all_probs[:, i])
    plt.plot(recall_curve, precision_curve, lw=2, label=f'{class_names[i]} (AP={ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR 曲线')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n所有评估指标计算完成！")