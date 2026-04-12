import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ==================== 修复Matplotlib中文乱码（仅添加这部分配置） ====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows黑体（Mac/Linux自行替换）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==================== 核心：直接导入已有编码器文件 ====================
# 假设三个编码器文件与当前文件在同一目录下
from autoencoder_mnist_pytorch import AutoEncoder  # 导入普通AE
from conv_autoencoder_mnist_pytorch import ConvAutoEncoder  # 导入卷积AE
from vae_mnist_pytorch import VAE  # 导入VAE

# ==================== 全局设置 ====================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")
print(f"✅ PyTorch 版本: {torch.__version__}")

# ==================== 参数设置（与已有编码器对齐） ====================
# 编码器类型选择: 'mlp_ae' / 'conv_ae' / 'vae'
ENCODER_TYPE = "vae"  
IMG_SIZE = 28
NUM_FEATURES = IMG_SIZE * IMG_SIZE   # 784 (仅mlp_ae使用)
LATENT_DIM = 64                      # 与已有编码器保持一致
HIDDEN_DIM = 128                     # 分类头隐藏层维度
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 10                     # MNIST分类类别数

# ==================== 1. 数据加载（严格对齐已有编码器的transform） ====================
if ENCODER_TYPE == "mlp_ae":
    # 复用autoencoder_mnist_pytorch.py的transform（展平向量）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
else:
    # 复用conv_ae/vae的transform（2D图像）
    transform = transforms.Compose([transforms.ToTensor()])

# 加载MNIST数据集（与已有编码器逻辑一致）
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"数据加载完成 → 训练集 {len(train_dataset)} 张 | 测试集 {len(test_dataset)} 张")

# ==================== 2. 分类器封装（仅封装+新增分类头，不修改原有编码器） ====================
class EncoderClassifier(nn.Module):
    def __init__(self, encoder_type):
        super().__init__()
        # 初始化已有编码器
        if encoder_type == "mlp_ae":
            self.encoder = AutoEncoder().to(device)
        elif encoder_type == "conv_ae":
            self.encoder = ConvAutoEncoder().to(device)
        elif encoder_type == "vae":
            self.encoder = VAE().to(device)
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # MLP分类头（新增，不影响原有编码器）
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES)
        )
        self.encoder_type = encoder_type

    def forward(self, x):
        # 适配不同编码器的forward逻辑
        if self.encoder_type == "mlp_ae":
            # AutoEncoder forward返回 (recon, latent)
            _, latent = self.encoder(x)
        elif self.encoder_type == "conv_ae":
            # ConvAutoEncoder forward返回 (recon, latent)
            _, latent = self.encoder(x)
        elif self.encoder_type == "vae":
            # VAE forward返回 (recon, mu, logvar)，取mu作为特征
            _, mu, _ = self.encoder(x)
            latent = mu
        
        # 分类头预测
        logits = self.classifier(latent)
        return logits

# ==================== 3. 评估函数 ====================
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

# ==================== 4. 训练主逻辑 ====================
if __name__ == "__main__":
    # 1. 初始化分类器（仅传入编码器类型，复用已有编码器）
    model = EncoderClassifier(ENCODER_TYPE).to(device)
    
    # 2. 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()  # 分类任务专用损失
    
    # 3. 训练
    print(f"\n🚀 开始训练 {ENCODER_TYPE} + MLP分类器...")
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # 反向传播+优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 评估
        avg_train_loss = train_loss / len(train_loader)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f"{ENCODER_TYPE}_classifier_best.pth")
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  训练损失: {avg_train_loss:.6f} | 测试损失: {test_loss:.6f}")
        print(f"  测试准确率: {test_acc:.2f}% | 最佳准确率: {best_accuracy:.2f}%")
    
    # ==================== 5. 最终评估和可视化 ====================
    print("\n✅ 训练完成！")
    print(f"🏆 最佳测试准确率: {best_accuracy:.2f}%")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f"{ENCODER_TYPE}_classifier_best.pth"))
    model.eval()
    
    # 可视化前10个测试样本
    with torch.no_grad():
        data, labels = next(iter(test_loader))
        data = data[:10].to(device)
        labels = labels[:10].cpu().numpy()
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
    # 适配不同编码器的图像格式
    if ENCODER_TYPE == "mlp_ae":
        data = data.cpu().numpy().reshape(-1, IMG_SIZE, IMG_SIZE)
    else:
        data = data.cpu().numpy().squeeze(1)
    
    # 绘图
    plt.figure(figsize=(20, 4))
    for i in range(10):
        ax = plt.subplot(1, 10, i+1)
        plt.imshow(data[i], cmap="gray")
        plt.title(f"真实:{labels[i]}\n预测:{preds[i]}", fontsize=10)
        plt.axis("off")
    plt.suptitle(f"{ENCODER_TYPE} + MLP分类器预测结果", fontsize=14)
    plt.show()
    
    print(f"\n🎉 {ENCODER_TYPE} 分类器运行完毕！最佳模型已保存为 {ENCODER_TYPE}_classifier_best.pth")