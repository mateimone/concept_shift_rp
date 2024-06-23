import torch
from torch import nn


class RiverNDMLP(nn.Module):
    def __init__(self, n_features):
        super(RiverNDMLP, self).__init__()
        self.fc1 = nn.Linear(n_features, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.drop(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = torch.sigmoid(self.fc2(x))
        return x