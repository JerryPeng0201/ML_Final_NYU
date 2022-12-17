import torch
import torchvision
from torch import nn
from collections import OrderedDict


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_stride = 4
        self.out_layers = [1, 2, 3]
        self.resnet = torchvision.models.resnet18(pretrained=True)
        children = list(self.resnet.children())
        self.layer0 = nn.Sequential(*children[:4])  # layer0: conv + bn + relu + pool
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]

    def forward(self, x):
        feat = OrderedDict()
        x = self.layer0(x)  # out_stride: 4
        feat1 = self.layer1(x)  # out_stride: 4
        feat2 = self.layer2(feat1)  # out_stride: 8
        feat3 = self.layer3(feat2)  # out_stride: 16
        feat4 = self.layer4(feat3)  # out_stride: 32
        feat['map3'] = feat3
        feat['map4'] = feat4
        return feat


class Regressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_dim, 196, 7, padding=3),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 64, 5, padding=2),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 1, 1),
            #nn.LeakyReLU(),
            #nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )
        self.pool = 'mean'

    def forward(self, im):
        num_sample = im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))

            output = torch.mean(output, dim=(0),keepdim=True)
            return output
        else:
            for i in range(0,num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0),keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0,keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output),dim=0)
            return Output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
