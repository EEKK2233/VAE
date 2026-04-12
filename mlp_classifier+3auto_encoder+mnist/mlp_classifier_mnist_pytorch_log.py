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
# 导入日志模块（完全模仿你的自编码器代码）
import logging

# 自动添加 当前目录(./) 和 上级目录(../) 到Python搜索路径
sys.path.append(os.path.abspath("."))   # 兼容 ./
sys.path.append(os.path.abspath(".."))  # 兼容 ../

sys.stdout.reconfigure(encoding='utf-8')

# ==================== 日志配置（与自编码器代码完全一致） ====================
logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

# ==================== 修复Matplotlib中文乱码 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 导入编码器 ====================
from autoencoder_mnist_pytorch import AutoEncoder
from conv_autoencoder_mnist_pytorch import ConvAutoEncoder
from vae_mnist_pytorch import VAE

# ==================== 全局设置 ====================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 日志输出设备信息
logging.info(f"使用设备: {device}")
logging.info(f"PyTorch 版本: {torch.__version__}")

# ==================== 参数设置 ====================
# 编码器类型选择: 'mlp_ae' / 'conv_ae' / 'vae'
ENCODER_TYPE = "conv_ae"
IMG_SIZE = 28
NUM_FEATURES = IMG_SIZE * IMG_SIZE
LATENT_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# ==================== 数据加载 ====================
if ENCODER_TYPE == "mlp_ae":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
else:
    transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 日志输出数据加载信息
logging.info(f"数据加载完成 → 训练集 {len(train_dataset)} 张 | 测试集 {len(test_dataset)} 张")

# ==================== 分类模型 ====================
class EncoderClassifier(nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == "mlp_ae":
            self.encoder = AutoEncoder().to(device)
        elif encoder_type == "conv_ae":
            self.encoder = ConvAutoEncoder().to(device)
        elif encoder_type == "vae":
            self.encoder = VAE().to(device)
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        )

    def forward(self, x):
        if self.encoder_type == "mlp_ae":
            _, latent = self.encoder(x)
        elif self.encoder_type == "conv_ae":
            _, latent = self.encoder(x)
        elif self.encoder_type == "vae":
            _, mu, _ = self.encoder(x)
            latent = mu
        
        logits = self.classifier(latent)
        return logits

# ==================== 评估函数 ====================
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ==================== 训练主逻辑 ====================
if __name__ == "__main__":
    model = EncoderClassifier(ENCODER_TYPE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练开始日志
    logging.info(f"开始训练 {ENCODER_TYPE} + MLP 分类器...")
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f"{ENCODER_TYPE}_classifier_best.pth")
        
        # 统一格式的训练日志（与自编码器风格一致）
        logging.info(
            "Epoch: %d, Train Loss: %.6f, Test Loss: %.6f, Test Acc: %.2f%%, Best Acc: %.2f%%",
            epoch+1, avg_train_loss, test_loss, test_acc, best_accuracy
        )

    # 训练完成日志
    logging.info("训练完成！")
    logging.info(f"最佳测试准确率: {best_accuracy:.2f}%")

    # 加载模型
    model.load_state_dict(torch.load(f"{ENCODER_TYPE}_classifier_best.pth"))
    model.eval()

    # 可视化
    with torch.no_grad():
        data, labels = next(iter(test_loader))
        data = data[:10].to(device)
        labels = labels[:10].cpu().numpy()
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
    if ENCODER_TYPE == "mlp_ae":
        data = data.cpu().numpy().reshape(-1, IMG_SIZE, IMG_SIZE)
    else:
        data = data.cpu().numpy().squeeze(1)
    
    plt.figure(figsize=(20, 4))
    for i in range(10):
        ax = plt.subplot(1, 10, i+1)
        plt.imshow(data[i], cmap="gray")
        plt.title(f"真实:{labels[i]}\n预测:{preds[i]}", fontsize=10)
        plt.axis("off")
    plt.suptitle(f"{ENCODER_TYPE} + MLP分类器预测结果", fontsize=14)
    plt.show()
    
    logging.info(f"{ENCODER_TYPE} 分类器运行完毕！最佳模型已保存为 {ENCODER_TYPE}_classifier_best.pth")