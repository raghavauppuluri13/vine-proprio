from torch import nn
import torch.nn.functional as F
from util import init_weights
import torchvision.models as models
import torch

from torchvision.models import resnet50

class ProprioNet(nn.Module):
    def __init__(self, in_dim, state_dim):
        super(ProprioNet, self).__init__()
        self.state_dim = state_dim
        assert in_dim[1] % 4 == 0 and in_dim[0] % 4 == 0
        self.resnet15 = self._create_resnet15()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_layer = nn.Linear(256, state_dim)

        # Freeze the parameters of the ResNet model
        for param in self.resnet15.parameters():
            param.requires_grad = False

        self.apply(self.init_weights)
        self.double()

    def _create_resnet15(self):
        resnet = resnet50(pretrained=True)
        # modify resnet to have 15 layers
        resnet = nn.Sequential(*list(resnet.children())[:5])  # assuming 3 is the number of layers you want
        return resnet

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
        elif type(m) == nn.BatchNorm2d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.resnet15(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        x = torch.tanh(x)  # tanh activation limits output to [-1, 1]
        x = x * (torch.pi / 2)  # scale output to range [-pi/2, pi/2]
        return x

