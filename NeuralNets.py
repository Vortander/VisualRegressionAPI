# coding: utf-8

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn

class SatNet(nn.Module):

    def __init__(self, feature_vector_size, output):
        super(SatNet, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.hidden_layer = int(np.mean([feature_vector_size, output]))

        self.fc1 = nn.Linear(self.feature_vector_size, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.output)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StreetNet(nn.Module):

    def __init__(self, feature_vector_size, output):
        super(StreetNet, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.hidden_layer = int(np.mean([feature_vector_size, output]))

        self.fc1 = nn.Linear(self.feature_vector_size, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.output)


    def forward(self, x):
        x = x.view(-1, self.feature_vector_size)  #Concat mode
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class StreetSatNet(nn.Module):

    def __init__(self, feature_vector_size, output):
        super(StreetSatNet, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.hidden_layer = int(np.mean([feature_vector_size, output]))

        self.fc1 = nn.Linear(self.feature_vector_size, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.output)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1) #Concat mode
        x = x.view(-1, self.feature_vector_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
