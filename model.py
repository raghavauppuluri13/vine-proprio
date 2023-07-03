from torch import nn
import torch.nn.functional as F
from util import init_weights
import torchvision.models as models
import torch

class ProprioNet(nn.Module):

    def __init__(self, in_dim, state_dim):
        super(ProprioNet, self).__init__()
        self.state_dim = state_dim
        assert in_dim[1] % 4 == 0 and in_dim[0] % 4 == 0
        #self.resnet50 = models.resnet34(pretrained=True)
        self.conv1 = nn.Conv2d(3,64,kernel_size=4, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64,128,kernel_size=4, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(128,256,kernel_size=4, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.conv4 = nn.Conv2d(256,512,kernel_size=4, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(num_features=512)
        self.last_layer = nn.Conv2d(512,1,kernel_size=3, padding=1, stride=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x) # bsx512x31x31

        x = self.last_layer(x) # 16x16x1
        x = nn.functional.sigmoid(x)
        x = torch.flatten(x,start_dim=1)
        return x
