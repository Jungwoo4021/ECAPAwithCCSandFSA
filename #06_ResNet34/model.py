import random

import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch


####################
## Main framework ##
####################
class ResNet18(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.torchfbank = nn.Sequential(          
            torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=2048, win_length=512, hop_length=256, \
                                                 window_fn=torch.hann_window, n_mels=48),
            )
        self.spectrogram_norm = nn.InstanceNorm1d(48)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.layer1 = self._make_layer(64, 64, num_block=3, stride=2)
        self.layer2 = self._make_layer(64, 128, num_block=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_block=6, stride=2)
        self.layer4 = self._make_layer(256, 512, num_block=3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 10)

        self.cce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()
            #x = x - torch.mean(x, dim=-1, keepdim=True) # CMV

            # spec_norm
            x = self.spectrogram_norm(x).unsqueeze(1)

        # fst conv
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        # resblock
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # pooling
        x = self.avg_pool(x)
        
        # fc
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # loss
        if labels is not None:
            loss = self.cce(x, labels)
            return loss
        else:
            prediction = self.softmax(x)
            return prediction

    def _make_layer(self, in_channels, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(2, in_channels, out_channels, stride, downsample))
        for i in range(1, num_block):
            layers.append(BasicBlock(2, out_channels, out_channels))

        return nn.Sequential(*layers)



#################
## Sub modules ##
#################
class BasicBlock(nn.Module):
    def __init__(self, conv_dim, in_channels, out_channels, stride=1, downsample=None, reduction=8):
        super().__init__()
        if conv_dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif conv_dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn(out_channels)

        self.conv2 = conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = bn(out_channels)

        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class PreEmphasis(torch.nn.Module):
    
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)