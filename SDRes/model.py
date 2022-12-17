import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

class StochasticDepth(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        p: float = 0.5,
        projection: torch.nn.Module = None,
    ):
        super().__init__()
        if not 0 < p < 1:
            raise ValueError(
                "Stochastic Depth p has to be between 0 and 1 but got {}".format(p)
            )
        self.module: torch.nn.Module = module
        self.p: float = p
        self.projection: torch.nn.Module = projection
        self._sampler = torch.Tensor(1)

    def forward(self, inputs):
        if self.training and self._sampler.uniform_():
            if self.projection is not None:
                return self.projection(inputs)
            return inputs
        return self.p * self.module(inputs)

class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()

        self.resnet = torchvision.models.resnet152(pretrained=True)
        #torch.load(settings.MODEL_FILE, map_location={'cuda:0': 'cpu'})
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

        #model = torchvision.models.resnet18()

        self.resnet.layer1 = StochasticDepth(self.resnet.layer1)
        self.resnet.layer2 = StochasticDepth(
            self.resnet.layer2,
            projection=torch.nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
        )
        self.resnet.layer3 = StochasticDepth(
            self.resnet.layer3,
            projection=torch.nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
        )
        self.resnet.layer4 = StochasticDepth(
            self.resnet.layer4,
            projection=torch.nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
        )





    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat


class CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample =  im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0),keepdim=True)  
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0,keepdim=True)
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
                    Output = torch.cat((Output,output),dim=0)
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
            
            