import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ==================== 全局设置 ====================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")
print(f"✅ PyTorch 版本: {torch.__version__}")


# ==================== 参数设置 ====================
IMG_SIZE = 28
NUM_FEATURES = IMG_SIZE * IMG_SIZE   # 784
LATENT_DIM = 64                      # 潜在空间维度（可修改）
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001


# ==================== 1. 数据加载 ====================
transform = transforms.Compose([
    transforms.ToTensor(),           # 转为 [0,1] 的 Tensor
    transforms.Lambda(lambda x: x.view(-1))  # 展平为 784 维
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"数据加载完成 → 训练集 {len(train_dataset)} 张 | 测试集 {len(test_dataset)} 张")


# ==================== 2. 定义模型 ====================
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(NUM_FEATURES, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_FEATURES),
            nn.Sigmoid()          # 输出像素值在 [0,1] 之间
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent   # 返回重建结果和潜变量（用于可视化）

    def get_encoder(self):
        """返回单独的编码器（用于 2D 潜空间可视化）"""
        return self.encoder


# ==================== 3. 可视化函数 ====================
def plot_reconstructions(model, test_loader, n=10):
    """可视化原始图像 vs 重建图像"""
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))[0][:n].to(device)
        reconstructed, _ = model(data)
    
    data = data.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原始
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        plt.axis("off")
        # 重建
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        plt.axis("off")
    plt.suptitle("上排：原始图像    下排：重建图像")
    plt.show()


def plot_latent_space(encoder, test_loader):
    """2D 潜空间散点图"""
    encoder.eval()
    all_latent = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            latent = encoder(data)
            all_latent.append(latent.cpu().numpy())
            all_labels.append(labels.numpy())
    
    z = np.concatenate(all_latent, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap="tab10", alpha=0.7, s=8)
    plt.colorbar(scatter, label="数字类别")
    plt.xlabel("潜空间维度 1")
    plt.ylabel("潜空间维度 2")
    plt.title("MNIST 在 2D 潜空间中的分布（PyTorch 版）")
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 4. 训练 ====================
if __name__ == "__main__":
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("🚀 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.6f}")
    
    print("\n✅ 训练完成！")
    
    # ==================== 5. 可视化 ====================
    print("\n📊 可视化重建结果...")
    plot_reconstructions(model, test_loader)
    
    print("\n📈 可视化 2D 潜空间...")
    encoder = model.get_encoder()
    plot_latent_space(encoder, test_loader)
    
    # 可选：保存模型
    # torch.save(model.state_dict(), "autoencoder_mnist_pytorch.pth")
    # print("模型已保存！")
    
    print("\n🎉 PyTorch 版 MNIST 自动编码器运行完毕！")