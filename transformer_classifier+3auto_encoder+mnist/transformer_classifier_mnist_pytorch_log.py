import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging

# 自动添加 当前目录(./) 和 上级目录(../) 到Python搜索路径
sys.path.append(os.path.abspath("."))   # 兼容 ./
sys.path.append(os.path.abspath(".."))  # 兼容 ../

# ==================== 日志配置 ====================
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

# ==================== 中文显示 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 导入编码器 ====================
from three_kinds_autoencoder.autoencoder_mnist_pytorch import AutoEncoder
from three_kinds_autoencoder.conv_autoencoder_mnist_pytorch import ConvAutoEncoder
from three_kinds_autoencoder.vae_mnist_pytorch import VAE

# ==================== 全局配置 ====================
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f"使用设备: {device}")
logging.info(f"PyTorch 版本: {torch.__version__}")

# ==================== 超参数 ====================
ENCODER_TYPE = "conv_ae"   # 'mlp_ae' / 'conv_ae' / 'vae'
LATENT_DIM = 64
SEQ_LEN = 16
D_MODEL = LATENT_DIM
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.3
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 28

# ==================== 数据加载 ====================
if ENCODER_TYPE == "mlp_ae":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

logging.info(f"数据加载完成 → 训练集 {len(train_dataset)} 张 | 测试集 {len(test_dataset)} 张")

# ==================== 1. 序列生成 ====================
class SequenceGenerator(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, latent_dim * seq_len)

    def forward(self, z):
        out = self.fc(z)
        return out.view(out.shape[0], self.seq_len, -1)

# ==================== 2. 位置编码 ====================
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_emb

# ==================== 3. Transformer 分类器 ====================
class TransformerClassifier(nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "mlp_ae":
            self.encoder = AutoEncoder()
        elif encoder_type == "conv_ae":
            self.encoder = ConvAutoEncoder()
        elif encoder_type == "vae":
            self.encoder = VAE()
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")

        self.encoder.to(device)

        self.seq_gen = SequenceGenerator(LATENT_DIM, SEQ_LEN)
        self.pos_emb = PositionalEmbedding(SEQ_LEN, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, NUM_CLASSES)
        )

    def forward(self, x):
        if self.encoder_type in ["mlp_ae", "conv_ae"]:
            _, latent = self.encoder(x)
        else:
            _, mu, _ = self.encoder(x)
            latent = mu

        seq = self.seq_gen(latent)
        seq = self.pos_emb(seq)
        feat = self.transformer(seq)
        feat = torch.mean(feat, dim=1)
        return self.classifier(feat)

# ==================== 评估 ====================
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    return total_loss/len(loader), 100*correct/len(loader.dataset)

# ==================== 训练 ====================
if __name__ == "__main__":
    model = TransformerClassifier(ENCODER_TYPE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    logging.info(f"开始训练 {ENCODER_TYPE} + Transformer 分类器")
    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{ENCODER_TYPE}_transformer_classifier.pth")

        logging.info(
            "Epoch: %d, Train Loss: %.4f, Test Loss: %.4f, Test Acc: %.2f%%, Best: %.2f%%",
            epoch+1, train_loss, test_loss, test_acc, best_acc
        )

    logging.info(f"训练完成！最佳准确率: {best_acc:.2f}%")

    # ==================== 可视化 ====================
    model.load_state_dict(torch.load(f"{ENCODER_TYPE}_transformer_classifier.pth"))
    model.eval()

    with torch.no_grad():
        x, y = next(iter(test_loader))
        x, y = x[:10].to(device), y[:10].cpu().numpy()
        pred = model(x).argmax(1).cpu().numpy()

        if ENCODER_TYPE == "mlp_ae":
            x = x.cpu().numpy().reshape(-1, 28, 28)
        else:
            x = x.cpu().numpy().squeeze(1)

    plt.figure(figsize=(20, 4))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(x[i], cmap='gray')
        plt.title(f"真实:{y[i]}\n预测:{pred[i]}")
        plt.axis('off')

    plt.suptitle(f"{ENCODER_TYPE} + Transformer分类器结果")
    plt.show()

    logging.info("程序运行完毕！")