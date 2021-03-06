import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as f



class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g




class Residual_Block1(nn.Module):
    def __init__(self, Channel):
        super(Residual_Block1, self).__init__()
        self.bn1 = nn.BatchNorm2d(Channel)
        self.bn2 = nn.BatchNorm2d(Channel)
        self.conv1 = nn.Conv2d(Channel, Channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(Channel, Channel, 3, 1, 1)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.prelu(self.bn1(x1))
        x3 = self.conv2(x2)
        x4 = self.bn2(x3)
        x5 = torch.add(x, x4)
        return x5


class Scale_1_Encoder(nn.Module):
    def __init__(self, Mid_Channel):
        super(Scale_1_Encoder, self).__init__()

        # scale 2
        self.block1 = Residual_Block1(Mid_Channel)
        self.block2 = Residual_Block1(Mid_Channel)
        self.block3 = Residual_Block1(Mid_Channel)
        self.block4 = Residual_Block1(Mid_Channel)
        self.block5 = Residual_Block1(Mid_Channel)
        self.block6 = Residual_Block1(Mid_Channel)
        self.block7 = Residual_Block1(Mid_Channel)
        self.conv1 = nn.Conv2d(3,64,5,1,2)
        self.conv2 = nn.Conv2d(64,8,3,1,1)
        self.pooling1 = nn.Conv2d(64,64,4,2,1)
        self.pooling2 = nn.Conv2d(64,64,4,2,1)
        self.pooling3 = nn.Conv2d(64,64,4,2,1)
        self.prelu1 = nn.PReLU()

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pooling1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pooling2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pooling3(x)
        x = self.block7(x)

        x1 = f.sigmoid(self.conv2(x))
        x1 = x1 * 63.0
        x = RoundNoGradient.apply(x1)   # every feature map 0/1
        return x,x1





class Scale_1_Decoder(nn.Module):
    def __init__(self,Mid_Channel):
        super(Scale_1_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(8,64,3,1,1)
        self.block1 = Residual_Block1(Mid_Channel)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block2 = Residual_Block1(Mid_Channel)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block3 = Residual_Block1(Mid_Channel)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.block4 = Residual_Block1(Mid_Channel)
        self.block5 = Residual_Block1(Mid_Channel)
        self.block6 = Residual_Block1(Mid_Channel)
        self.conv7 = nn.Conv2d(16, 3, 5, 1, 2)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
        self.up3 = nn.PixelShuffle(2)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()
    def forward(self, x):
        x = x/63.0
        xp = self.prelu1(self.conv1(x))

        x = self.block1(xp)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.prelu2(self.up1(x))
        x = self.prelu3(self.conv3(x))
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv4(x)
        x = self.prelu4(self.up2(x))
        x = self.prelu5(self.conv5(x))
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv6(x)
        x = self.prelu6(self.up3(x))
        x = self.conv7(x)
        return x

class Scale_1_AutoEncoder(nn.Module):
    def __init__(self,Mid_Channel):
        super(Scale_1_AutoEncoder,self).__init__()
        self.encoder = Scale_1_Encoder(Mid_Channel)    # how to use the new mode
        self.decoder = Scale_1_Decoder(Mid_Channel)


    def forward(self,x):
        x,x1 = self.encoder(x)
        out = self.decoder(x)
        return out,x1
