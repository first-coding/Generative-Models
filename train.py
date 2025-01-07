from random import shuffle
from cv2 import transform
from sympy import false
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
from models.DCGANModel import Generator,Discriminator
from config.config import Config
from torch.utils.data import DataLoader  
from tqdm import tqdm  # 导入tqdm模块
import torchvision.utils as vutils
import matplotlib.pyplot as plt  # 导入matplotlib
import os
config = Config()

transform = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])

dataset = torchvision.datasets.CIFAR10(root='./data',download=False,transform=transform)
dataloader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True)

netG = Generator(config.noise_size, config.feature, config.channels).to(config.device)
netD = Discriminator(config.channels, config.ndf).to(config.device)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64,config.noise_size,1,1,device=config.device)

optimizerD = optim.Adam(netD.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

losses_D = [] 
losses_G = []  


for epoch in range(config.num_epochs):
    # 使用 tqdm 显示每个 epoch 的进度条
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{config.num_epochs}', ncols=100) as pbar:
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_images = data[0].to(config.device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size,), 1, dtype=torch.float, device=config.device)
            output = netD(real_images).view(-1)
            errD_real = criterion(output, labels)
            errD_real.backward()

            noise = torch.randn(batch_size, config.noise_size, 1, 1, device=config.device)
            fake_images = netG(noise)
            labels.fill_(0)
            output = netD(fake_images.detach()).view(-1)
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            optimizerD.step()

            netG.zero_grad()
            labels.fill_(1)  
            output = netD(fake_images).view(-1)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()

            # 更新损失记录
            if i % 50 == 0:
                losses_D.append(errD_real.item() + errD_fake.item())
                losses_G.append(errG.item())
                pbar.set_postfix(Loss_D=errD_real.item() + errD_fake.item(), Loss_G=errG.item())
            
            pbar.update(1)  # 更新进度条

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        # 在保存图像之前，确保 output 文件夹存在
        if not os.path.exists('output'):
            os.makedirs('./output')
        vutils.save_image(fake, f'./output/fake_samples_epoch_{epoch}.png', normalize=True, nrow=8)

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_D, label="D Loss")
plt.plot(losses_G, label="G Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('output/loss_curve.png') 
plt.show()