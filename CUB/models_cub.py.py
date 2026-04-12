# models_cub.py
import torch
import torch.nn as nn
import torchvision.models as models

class ConvVAE_CUB(nn.Module):
    """针对 CUB-200-2011 的卷积变分自编码器（使用 ResNet50 作为 backbone）"""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim

        # 使用预训练 ResNet50 作为特征提取器（论文中推荐的迁移学习方式）
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 移除最后的全连接层
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # 输出 (batch, 2048, 1, 1)

        # 分布参数头
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

        # 解码器（从 latent_dim 重构 224×224 图像）
        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 7, 7)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  3,   kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)           # (B, 2048, 1, 1)
        h = h.view(h.size(0), -1)     # (B, 2048)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class TransformerClassifier_CUB(nn.Module):
    """Transformer 分类器（适用于 200 类细粒度任务）"""
    def __init__(self, latent_dim=512, num_classes=200):
        super().__init__()

        self.seq_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)   # 增加层数适应细粒度任务

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, z):
        z = z.unsqueeze(1)
        z = self.seq_proj(z) + self.pos_embedding
        z = self.transformer(z)
        z = z.mean(dim=1)
        return self.classifier(z)