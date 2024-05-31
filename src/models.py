#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_geometric.transforms as T


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


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    """
    Second CNN is on run command
    First CNN is in terminal
    terminal 1 has baseline
    terminal 2 has federated
    """

    # def __init__(self, args):  # 80% accuracy on test set
    #     super(CNNCifar, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=1, padding=0)
    #     self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
    #     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #     self.dropout1 = nn.Dropout(0.4)
    #
    #     self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
    #     self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
    #     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    #     self.dropout2 = nn.Dropout(0.4)
    #
    #     self.flatten = nn.Flatten()
    #     self.fc1 = nn.Linear(128 * 11 * 11, 1024)
    #     self.fc2 = nn.Linear(1024, 1024)
    #     self.fc3 = nn.Linear(1024, 10)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.pool1(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.pool1(x)
    #     x = self.dropout1(x)
    #
    #     x = F.relu(self.conv3(x))
    #     x = self.pool1(x)
    #     x = F.relu(self.conv4(x))
    #     x = self.pool2(x)
    #     x = self.dropout2(x)
    #
    #     x = self.flatten(x)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.log_softmax(self.fc3(x), dim=1)
    #     return x
    def __init__(self):  # 75% accuracy on test set
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 128, 2)
        # self.pool3 = nn.MaxPool2d(2, 2)
        # self.conv4 = nn.Conv2d(128, 128, 2)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = F.relu(self.conv4(x))

        # x = x.view(-1, 64 * 4 * 4)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fc3(x)

        return x


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)  # for node classification where you only have 1 graph

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class CNNFlower(nn.Module):
    def __init__(self, args):
        super(CNNFlower, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 256, 3)

    def forward(self, x):
        x = F.relu()

    # def __init__(self, args):
    #     super(CNNCifar, self).__init__()
    #     # Assuming img_rows, img_cols, and channels are defined elsewhere in the code
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # padding to maintain size
    #     self.dropout1 = nn.Dropout(0.2)
    #     self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # padding to maintain size
    #     self.dropout2 = nn.Dropout(0.2)
    #     self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # padding to maintain size
    #     self.dropout3 = nn.Dropout(0.2)
    #     self.flatten = nn.Flatten()
    #     self.fc1 = nn.Linear(64 * 32 * 32, 128)  # adjust based on your image dimensions
    #     self.fc2 = nn.Linear(128, args.num_classes)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.dropout1(x)
    #     x = F.relu(self.conv2(x))
    #     x = self.dropout2(x)
    #     x = F.relu(self.conv3(x))
    #     x = self.dropout3(x)
    #     x = self.flatten(x)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return F.softmax(x, dim=1)


class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(modelC, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
