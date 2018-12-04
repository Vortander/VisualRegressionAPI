# coding: utf-8

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn


class SatNet(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=None):
        super(SatNet, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.hidden_layer = int(np.mean([feature_vector_size, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.output)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout != None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return x

class SatNet2(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=None):
        super(SatNet2, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SatNet3(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=[None, None, None]):
        super(SatNet3, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.third_hidden_layer = int(np.mean([self.second_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, self.third_hidden_layer)
        self.fc4 = nn.Linear(self.third_hidden_layer, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.dropout[0] != None:
            x = F.dropout(x, p=self.dropout[0], training=self.training)
        x = F.relu(self.fc2(x))

        if self.dropout[1] != None:
            x = F.dropout(x, p=self.dropout[1], training=self.training)
        x = F.relu(self.fc3(x))

        if self.dropout[2] != None:
            x = F.dropout(x, p=self.dropout[1], training=self.training)
        x = F.relu(self.fc4(x))

        return x

class SatNet4(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=None):
        super(SatNet4, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.third_hidden_layer = int(np.mean([self.second_hidden_layer, output]))
        self.fourth_hidden_layer = int(np.mean([self.third_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, self.third_hidden_layer)
        self.fc4 = nn.Linear(self.third_hidden_layer, self.fourth_hidden_layer)
        self.fc5 = nn.Linear(self.fourth_hidden_layer, self.output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
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

class StreetNet2(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=None):
        super(StreetNet2, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, output)

    def forward(self, x):
        x = x.view(-1, self.feature_vector_size)  #Concat mode
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class StreetNet3(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=None):
        super(StreetNet3, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.third_hidden_layer = int(np.mean([self.second_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, self.third_hidden_layer)
        self.fc4 = nn.Linear(self.third_hidden_layer, output)

    def forward(self, x):
        x = x.view(-1, self.feature_vector_size)  #Concat mode
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
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

class StreetSatNet2(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=[None, None]):
        super(StreetSatNet2, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, output)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1) #Concat mode
        x = x.view(-1, self.feature_vector_size)  #Concat mode

        x = F.relu(self.fc1(x))
        if self.dropout[0] != None:
            x = F.dropout(x, p=self.dropout[0], training=self.training)

        x = F.relu(self.fc2(x))
        if self.dropout[1] != None:
            x = F.dropout(x, p=self.dropout[1], training=self.training)

        x = F.relu(self.fc3(x))

        return x

class StreetSatNet3(nn.Module):

    def __init__(self, feature_vector_size, output, dropout=[None, None, None]):
        super(StreetSatNet3, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output
        self.first_hidden_layer = int(np.mean([feature_vector_size, output]))
        self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]))
        self.third_hidden_layer = int(np.mean([self.second_hidden_layer, output]))
        self.dropout = dropout

        self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
        self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
        self.fc3 = nn.Linear(self.second_hidden_layer, self.third_hidden_layer)
        self.fc4 = nn.Linear(self.third_hidden_layer, output)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1) #Concat mode
        x = x.view(-1, self.feature_vector_size)  #Concat mode

        x = F.relu(self.fc1(x))
        if self.dropout[0] != None:
            x = F.dropout(x, p=self.dropout[0], training=self.training)

        x = F.relu(self.fc2(x))
        if self.dropout[0] != None:
            x = F.dropout(x, p=self.dropout[0], training=self.training)

        x = F.relu(self.fc3(x))
        if self.dropout[0] != None:
            x = F.dropout(x, p=self.dropout[0], training=self.training)

        x = F.relu(self.fc4(x))
        return x
