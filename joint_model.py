import torch
import torch.nn as nn
import torch.nn.functional as f
import math
import pixelcnn


# 以下 RoundNoGradient,Residual_Block1,Scale_1_Encoder,Scale_1_Decoder为deep coder的网络模块
# 其中 RoundNoGradient在前向传播中进行量化，反向传播中跳过了量化步骤的梯度计算
# Residual_Block1是deep coder的基础网络模块
# Scale_1_Encoder为encoder
# Scale_1_Decoder为decoder
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
        self.conv1 = nn.Conv2d(Channel, Channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(Channel, Channel, 3, 1, 1)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.prelu(x1)
        x3 = self.conv2(x2)
        y = torch.add(x, x3)
        return y

class Scale_1_Encoder(nn.Module):
    def __init__(self, Res_Channel, Mid_Channel):
        super(Scale_1_Encoder, self).__init__()

        # scale 2
        self.block1 = Residual_Block1(Res_Channel)
        self.block2 = Residual_Block1(Res_Channel)
        self.block3 = Residual_Block1(Res_Channel)
        self.block4 = Residual_Block1(Res_Channel)
        self.block5 = Residual_Block1(Res_Channel)
        self.block6 = Residual_Block1(Res_Channel)
        self.block7 = Residual_Block1(Res_Channel)
        self.conv1 = nn.Conv2d(3,64,5,1,2)

        self.conv2 = nn.Conv2d(64,Mid_Channel,3,1,1) #!!!

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

        x1 = torch.sigmoid(self.conv2(x))
        x1 = x1 * 63.0
        x = RoundNoGradient.apply(x1)   # every feature map 0/1
        return x, x1

class Scale_1_Decoder(nn.Module):
    def __init__(self,Res_Channel,Mid_Channel):
        super(Scale_1_Decoder, self).__init__()

        self.conv1 = nn.Conv2d(Mid_Channel,64,3,1,1) #!!!

        self.block1 = Residual_Block1(Res_Channel)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block2 = Residual_Block1(Res_Channel)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 64, 3, 1, 1)
        self.block3 = Residual_Block1(Res_Channel)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.block4 = Residual_Block1(Res_Channel)
        self.block5 = Residual_Block1(Res_Channel)
        self.block6 = Residual_Block1(Res_Channel)
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

  return x

  
# 以下 conv3x3,BasicBlock,FBResNet为vision task engine的分类模块
# 该部分主要采用了Resnet的源码
# 具体的网络组成由CodedVision_net类中的命令self.classifier = FBResNet(BasicBlock, [2, 2, 2, 2], num_classes=21)中的参数来定义
# 这里[2,2,2,2]定义了一个18层的Resnet,num_classes设为了21，以适应训练和测试使用的Pascal数据集
  
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FBResNet(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        # Special attributs
        self.input_space = None
        self.input_size = (256, 256, 3)
        self.mean = None
        self.std = None
        super(FBResNet, self).__init__()
        # Modules
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool  = nn.Conv2d(64, 64, 3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        #x = features
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        input = input/63
        x = self.features(input)
        x = self.logits(x)
        return x

#以下是coded vision网络的定义
#输入图像先经过encoder进行编码
#编码后分为两个分支，decoder将其解码，classifier则直接在码流上进行分类		
		
class CodedVision_net(nn.Module):
    def __init__(self, channel, midchannel):
        super(CodedVision_net,self).__init__()

        self.encoder = Scale_1_Encoder(channel, midchannel)
        self.decoder = Scale_1_Decoder(Res_Channel=channel, Mid_Channel=midchannel)
        self.classifier = FBResNet(BasicBlock, [2, 2, 2, 2], num_classes=21)

    def forward(self, x):
        x, x_entropy = self.encoder(x)
        x1 = self.classifier(x)
        x2 = self.decoder(x)
        return x1, x2
