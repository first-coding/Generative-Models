import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,noise_size=100,feature=64,n_channels=3):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(noise_size,feature*8,4,1,0,bias=False),
            nn.BatchNorm2d(feature*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature*8,feature*4,4,2,1,bias=False),
            nn.BatchNorm2d(feature*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature*4,feature*2,4,2,1,bias=False),
            nn.BatchNorm2d(feature*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature*2,feature,4,2,1,bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature,n_channels,4,2,1,bias=False),
            nn.Tanh(),
        )

    def forward(self,input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)