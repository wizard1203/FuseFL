#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

from .basics import View


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
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
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self, hidden_features, num_of_classes):
        super(CNNCifar, self).__init__()
        self.hidden_features = hidden_features
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_features, 3),
            nn.BatchNorm2d(self.hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.hidden_features, self.hidden_features, 3),
            nn.BatchNorm2d(self.hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.hidden_features, self.hidden_features, 3),
            nn.BatchNorm2d(self.hidden_features),
            nn.ReLU())
        # self.conv1 = nn.Conv2d(3, self.hidden_features, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(self.hidden_features, self.hidden_features, 3)
        # self.conv3 = nn.Conv2d(self.hidden_features, self.hidden_features, 3)

        self.fc1 = nn.Linear(self.hidden_features * 4 * 4, num_of_classes)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(self.conv3(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(-1, self.hidden_features * 4 * 4)
        x = self.fc1(x)
        return x



def make_CNNCifar_seqs(in_features, hidden_features, out_features, init_classifier):
    layers = []
    layers.append(nn.Sequential(
        nn.Conv2d(in_features, hidden_features, 3),
        nn.BatchNorm2d(hidden_features),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)))
    layers.append(nn.Sequential(
        nn.Conv2d(hidden_features, hidden_features, 3),
        nn.BatchNorm2d(hidden_features),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)))
    layers.append(nn.Sequential(
        nn.Conv2d(hidden_features, hidden_features, 3),
        nn.BatchNorm2d(hidden_features),
        nn.ReLU(),
        View([hidden_features * 4 * 4])))
    # conv1 = nn.Conv2d(3, hidden_features, 3)
    # pool = nn.MaxPool2d(2, 2)
    # conv2 = nn.Conv2d(hidden_features, hidden_features, 3)
    # conv3 = nn.Conv2d(hidden_features, hidden_features, 3)
    # layers
    # fc1 = nn.Linear(hidden_features * 4 * 4, num_of_classes)

    if init_classifier:
        classifier = torch.nn.Linear(hidden_features * 4 * 4, out_features)
        layers.append(classifier)

    return layers


def make_CNNCifar_Head_seqs(in_features, hidden_features, out_features, init_classifier, split_layer_index):
    origin_res_layer_index = 0
    layers = []
    if origin_res_layer_index > split_layer_index:
        layers.append(nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)))
    origin_res_layer_index += 1
    if origin_res_layer_index > split_layer_index:
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)))
    origin_res_layer_index += 1
    if origin_res_layer_index > split_layer_index:
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(),
            View([hidden_features * 4 * 4])))
    origin_res_layer_index += 1

    if init_classifier:
        if origin_res_layer_index > split_layer_index:
            classifier = torch.nn.Linear(hidden_features, out_features)
            layers.append(classifier)
        origin_res_layer_index += 1
    return layers





class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x


class CNNCifar2(nn.Module):  # 重新搭建CNN
    def __init__(self):
        super(CNNCifar2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
