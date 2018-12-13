# coding: utf-8

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn

from torchvision import models

class SatLinear(nn.Module):
    def __init__(self, feature_vector_size, output):
        super(SatLinear, self).__init__()
        self.feature_vector_size = feature_vector_size
        self.output = output

        self.fc1 = nn.Linear(self.feature_vector_size, self.output)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x

class SatNet(nn.Module):
    def __init__(self, fc_dropout=[None, None, None]):
        super(SatNet, self).__init__()
        self.densenet = models.densenet161(pretrained=False)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.fc_dropout = fc_dropout

        self.fc1 = nn.Linear(2208, 1104)
        self.fc2 = nn.Linear(1104, 552)
        self.fc3 = nn.Linear(552, 276)
        self.fc4 = nn.Linear(276, 1)
        
    def forward(self, x):
        #print('input', x.size())
        x = self.densenet(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        if self.fc_dropout[0] != None:
            x = F.dropout(x, p=self.fc_dropout[0], training=self.training)

        x = F.relu(self.fc2(x))
        if self.fc_dropout[1] != None:
            x = F.dropout(x, p=self.fc_dropout[1], training=self.training)

        x = F.relu(self.fc3(x))
        if self.fc_dropout[2] != None:
            x = F.dropout(x, p=self.fc_dropout[2], training=self.training)

        x = F.relu(self.fc4(x))

        return x


class OldNet(nn.Module):
	def __init__(self, n_hidden, n_output):
		super(OldNet, self).__init__()
		self.resnet = models.resnet101(pretrained=True)
		for param in self.resnet.parameters():
			param.requires_grad = False

		self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

		self.fc1 = nn.Linear(2048 * 1 * 1, 120)
		self.fc2 = nn.Linear(120, 84)

		self.predict = torch.nn.Linear(336, 1)
		#self.predict = torch.nn.Linear(8192, 1)


	#Output shapes
	# ('input', (1L, 1L, 28L, 28L))
	# ('conv1 out', (1L, 6L, 12L, 12L))
	# ('conv2 out', (1L, 16L, 4L, 4L))
	# ('view out ', (1L, 256L))
	# ('fc1 out ', (1L, 120L))
	# ('fc2 out ', (1L, 84L))
	# ('hidden out ', (1L, 10L))
	# ('predict out ', (1L, 1L))

	def forward_once(self, x):
		#print('input', x.size())
		x = self.resnet(x)
		#print('resnet out ', x.size())
		x = x.view(-1, 2048 * 1 * 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		#print('view out ', x.size())

		return x

	def forward(self, input1, input2, input3, input4):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		output3 = self.forward_once(input3)
		output4 = self.forward_once(input4)

		p = torch.cat((output1, output2, output3, output4),1)
		p = self.predict(p)
		return p

class SatNet_1(nn.Module):

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
            x = F.dropout(x, p=self.dropout[2], training=self.training)

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

    def __init__(self, feature_vector_size, output, dropout=[None, None, None]):
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
        if self.dropout[0] != None:
            x = F.dropout(x, p=self.dropout[0], training=self.training)

        x = F.relu(self.fc2(x))
        if self.dropout[1] != None:
            x = F.dropout(x, p=self.dropout[1], training=self.training)

        x = F.relu(self.fc3(x))
        if self.dropout[2] != None:
            x = F.dropout(x, p=self.dropout[2], training=self.training)

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
        self.first_hidden_layer = int(np.mean([self.feature_vector_size, output]))
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
        if self.dropout[1] != None:
            x = F.dropout(x, p=self.dropout[1], training=self.training)

        x = F.relu(self.fc3(x))
        if self.dropout[2] != None:
            x = F.dropout(x, p=self.dropout[2], training=self.training)

        x = F.relu(self.fc4(x))
        return x
