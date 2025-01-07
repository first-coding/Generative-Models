import torch

# 超参数设置
class Config:
    def __init__(self):
        # 数据集参数
        self.dataset = 'CIFAR10'
        self.image_size = 64  # CIFAR-10 的图像尺寸为 32x32，但我们将其缩放为 64x64
        self.channels = 3  # CIFAR-10 是 RGB 图像

        # 训练参数
        self.batch_size = 128  # 批次大小
        self.num_epochs = 50  # 训练轮数
        self.learning_rate = 0.0002  # 学习率
        self.beta1 = 0.5  # Adam 优化器的 beta1 参数
        self.beta2 = 0.999  # Adam 优化器的 beta2 参数

        # 模型参数
        self.noise_size = 100  # 噪声向量的大小
        self.feature = 64  # 生成器中特征图数量的基数
        self.ndf = 64  # 判别器中特征图数量的基数

        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
