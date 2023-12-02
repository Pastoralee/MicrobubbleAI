import torch
import torch.nn as nn
import math
from util import pad_to

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convB = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())

    def forward(self, inputs):
        x = self.convB(inputs)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convE = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())

        self.poolE = nn.Sequential(nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size,
                              bias = False, stride = 2, padding = math.ceil((1 - 2 + (kernel_size-1))/2)), #stride = 2 agis comme MaxPool2d en mieux, cf: https://arxiv.org/pdf/1412.6806.pdf
        nn.BatchNorm2d(out_channels),
        nn.ReLU())

    def forward(self, inputs):
        x = self.convE(inputs)
        p = self.poolE(x)
        return x, p

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.upD = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)

        self.convD = nn.Sequential(nn.Conv2d(in_channels=out_channels+out_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())

    def forward(self, inputs, skip):
        x = self.upD(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.convD(x)
        return x


class UnetBulle(nn.Module):
    def __init__(self, conv_dim = 3):
        super(UnetBulle, self).__init__()
        self.e1 = Encoder(1, 64, conv_dim)
        self.e2 = Encoder(64, 128, conv_dim)
        self.e3 = Encoder(128, 256, conv_dim)
        self.e4 = Encoder(256, 512, conv_dim)

        self.bottleneck = Bottleneck(512, 1024, conv_dim)

        self.d1 = Decoder(1024, 512, conv_dim)
        self.d2 = Decoder(512, 256, conv_dim)
        self.d3 = Decoder(256, 128, conv_dim)
        self.d4 = Decoder(128, 64, conv_dim)

        self.classifier = nn.Sequential(nn.Conv2d(64, out_channels = 1, kernel_size = 1, padding = 0),
            nn.Sigmoid())

        self.fc = nn.Sequential(nn.Flatten(),
            nn.Linear(96*144, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh())

    def forward(self, x):
        up1, x = self.e1(x)
        up2, x = self.e2(x)
        up3, x = self.e3(x)
        up4, x = self.e4(x)
        
        up4_pad, pads_up4 = pad_to(up4, 2)
        up3_pad, pads_up3 = pad_to(up3, 4)
        up2_pad, pads_up2 = pad_to(up2, 8)
        up1_pad, pads_up1 = pad_to(up1, 16)

        x = self.bottleneck(x)

        x = self.d1(x, up4_pad)
        x = self.d2(x, up3_pad)
        x = self.d3(x, up2_pad)
        x = self.d4(x, up1_pad)
        
        x = self.classifier(x)

        numBubbles = self.fc(x)
        return numBubbles