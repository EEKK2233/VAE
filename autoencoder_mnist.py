import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, datasets, callbacks

# ==================== 全局设置 ====================
tf.random.set_seed(42)          # 固定随机种子，保证每次运行结果一致
np.random.seed(42)

print("✅ TensorFlow 版本:", tf.__version__)
print("✅ GPU 可用情况:", "Yes" if tf.config.list_physical_devices('GPU') else "No")


# ==================== 1. 参数设置 ====================
IMG_SIZE = 28
NUM_FEATURES = IMG_SIZE * IMG_SIZE   # 784
LATENT_DIM = 64                      # 潜在空间维度（可自行修改）
HIDDEN_DIM = 128

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001


# ==================== 2. 数据加载 ====================
def load_mnist():
    """加载 MNIST 数据并归一化"""
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    # 归一化到 [0,1] 并展平
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train.reshape(-1, NUM_FEATURES)
    x_test = x_test.reshape(-1, NUM_FEATURES)
    
    return (x_train, y_train), (x_test, y_test)


# ==================== 3. 构建模型 ====================
def build_autoencoder():
    """构建编码器 + 解码器"""
    # ----- 编码器 -----
    encoder_input = layers.Input(shape=(NUM_FEATURES,), name="encoder_input")
    x = layers.Dense(HIDDEN_DIM, activation="relu")(encoder_input)
    latent = layers.Dense(LATENT_DIM, activation="relu", name="latent")(x)
    encoder = Model(encoder_input, latent, name="encoder")
    
    # ----- 解码器 -----
    decoder_input = layers.Input(shape=(LATENT_DIM,), name="latent_input")
    x = layers.Dense(HIDDEN_DIM, activation="relu")(decoder_input)
    decoder_output = layers.Dense(NUM_FEATURES, activation="sigmoid")(x)
    decoder = Model(decoder_input, decoder_output, name="decoder")
    
    # ----- 完整自编码器 -----
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = Model(encoder_input, autoencoder_output, name="autoencoder")
    
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"                     # 重建任务使用均方误差
    )
    
    encoder.summary()
    autoencoder.summary()
    return encoder, decoder, autoencoder


# ==================== 4. 可视化函数 ====================
def plot_reconstructions(model, x_test, n=10):
    """可视化原始图像 vs 重建图像"""
    decoded_imgs = model.predict(x_test[:n])
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 原始图像
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        plt.axis("off")
        
        # 重建图像
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        plt.axis("off")
    plt.suptitle("原始图像（上） vs 重建图像（下）")
    plt.show()


def plot_latent_space(encoder, x_test, y_test):
    """在 2D 潜空间中绘制散点图（颜色代表数字类别）"""
    z = encoder.predict(x_test, batch_size=BATCH_SIZE)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y_test, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="数字类别")
    plt.xlabel("潜空间维度 1")
    plt.ylabel("潜空间维度 2")
    plt.title("MNIST 在 2D 潜空间中的分布")
    plt.grid(True, alpha=0.3)
    plt.show()


# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    print("\n🚀 开始加载数据...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")
    
    print("\n🏗️  正在构建模型...")
    encoder, decoder, autoencoder = build_autoencoder()
    
    # EarlyStopping + 模型检查点（防止过拟合）
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    print("\n🔥 开始训练...")
    history = autoencoder.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n✅ 训练完成！")
    
    # ==================== 6. 结果可视化 ====================
    print("\n📊 可视化重建结果...")
    plot_reconstructions(autoencoder, x_test)
    
    print("\n📈 可视化 2D 潜空间...")
    plot_latent_space(encoder, x_test, y_test)
    
    # 可选：保存模型
    # encoder.save("encoder_mnist_tf2.h5")
    # autoencoder.save("autoencoder_mnist_tf2.h5")
    # print("模型已保存！")
    
    print("\n🎉 全部完成！欢迎修改 LATENT_DIM 或隐藏层大小进行实验～")