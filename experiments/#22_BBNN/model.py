import math
import random
import torch
import torch.nn as nn
import torchaudio

class BBNN(nn.Module):
    def __init__(self, args):
        super(BBNN, self).__init__()
        C = 13
        self.conv = nn.Conv2d(1, C, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(C)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((1,4))

        self.dense_blocks = dense_block(C)
        self.transition_blocks = transition_block(C*13)

        self.bn1 = nn.BatchNorm2d(C)
        # relu
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(C, 28)
        self.cce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, labels=None, train=True):
        B = x.size()[0]
        x = x.unsqueeze(dim=1)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.dense_blocks(x)
        x = self.transition_blocks(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.global_avg_pool(x)

        x = x.squeeze()
        if B == 1:
            x = x.unsqueeze(dim=0)
        x = self.classifier(x)
        
        if train:
            loss = self.cce(x, labels)
            return loss
        else:
            loss = self.cce(x, labels)
            prediction = self.softmax(x)
            return prediction, loss
        
        
    

class base_conv_block(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(base_conv_block, self).__init__()

        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding='same')

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class multi_scale_block(nn.Module):
    def __init__(self, inplanes, planes):
        super(multi_scale_block, self).__init__()

        self.branch1x1 = base_conv_block(inplanes, planes, 1)

        self.branch3x3_1 = base_conv_block(inplanes, planes, 1)
        self.branch3x3_2 = base_conv_block(planes, planes, 3)

        self.branch5x5_1 = base_conv_block(inplanes, planes, 1)
        self.branch5x5_2 = base_conv_block(planes, planes, 5)

        self.max_pool = nn.MaxPool2d((3,3),(1,1), padding=1)
        self.branch1x1_max = base_conv_block(inplanes, planes, 1)

    def forward(self,x):
        b1 = self.branch1x1(x)

        b2 = self.branch3x3_1(x)
        b2 = self.branch3x3_2(b2)

        b3 = self.branch5x5_1(x)
        b3 = self.branch5x5_2(b3)

        b4 = self.max_pool(x)
        b4 = self.branch1x1_max(b4)

        out = torch.cat((b1,b2,b3,b4),dim=1)

        return out

# 3 layer dense block
class dense_block(nn.Module):
    def __init__(self, inplanes):
        super(dense_block, self).__init__()

        self.dense1 = multi_scale_block(inplanes, inplanes)
        self.dense2 = multi_scale_block(inplanes*5, inplanes)
        self.dense3 = multi_scale_block(inplanes*9, inplanes)
        ## 32 -> 160(5) -> 288(9) -> 416(13)

    def forward(self, x):
        out = self.dense1(x)
        x = torch.cat((x,out), dim=1)
        
        out = self.dense2(x)
        x = torch.cat((x,out), dim=1)
        
        out = self.dense3(x)
        x = torch.cat((x,out), dim=1)
        
        return x

class transition_block(nn.Module):
    def __init__(self, inplanes):
        super(transition_block, self).__init__()

        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        # 416 -> 32
        self.conv = nn.Conv2d(inplanes, inplanes//13, kernel_size=1)
        self.avg_pool = nn.AvgPool2d((2,2))

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    args = { }
    model = BBNN(args).cuda()
    summary(model, input_size=(48,647))
