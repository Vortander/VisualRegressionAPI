# coding: utf-8

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn

class SatNet(nn.Module):

    def __init__(self, feature_vector_size, output):
        super(Net, self).__init__()
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
        super(Net, self).__init__()
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