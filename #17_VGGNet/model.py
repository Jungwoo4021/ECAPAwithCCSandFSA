import random

import torch.nn as nn
import torch.nn.functional as F


####################
## Main framework ##
####################
class VGGNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectrogram_norm = nn.InstanceNorm1d(48)
        
        self.block1 = VGGBlock(1, 64)
        self.block2 = VGGBlock(64, 128)
        self.block3 = VGGBlock(128, 256, flag_3block=True)
        self.block4 = VGGBlock(256, 512, flag_3block=True)
        self.block5 = VGGBlock(512, 512, flag_3block=True)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(3072, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, args['num_class'])

        self.cce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None):
        # spec_norm
        x = self.spectrogram_norm(x).unsqueeze(1)

        # conv
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # fc
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        # loss
        if labels is not None:
            loss = self.cce(x, labels)
            return loss
        else:
            prediction = self.softmax(x)
            return prediction

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, flag_3block=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = None
        if flag_3block:
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.conv3 is not None:
            x = self.conv3(x)
            x = self.relu(x)
        x = self.maxpool(x)

        return x
