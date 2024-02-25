import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet50 as regular_resnet50


class Basic_SSL_Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Basic_SSL_Model, self).__init__()
        self.f = []
        for name, module in regular_resnet50().named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        # return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        return feature, F.normalize(out, dim=-1)
