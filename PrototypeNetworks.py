# coding: utf-8

import sys, os

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
from torchvision import transforms, utils
from torchvision import models

from torchvision.models.resnet import model_urls

model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')
model_urls['resnet34'] = model_urls['resnet34'].replace('https://', 'http://')
model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
model_urls['resnet101'] = model_urls['resnet101'].replace('https://', 'http://')
model_urls['resnet152'] = model_urls['resnet152'].replace('https://', 'http://')

architectures = {'ResNet':['18','34','50','101','152'], 'SqueezeNet':['1.0', '1.1'], 'Densenet':['121','169','201','161'], 'Inception':['v3']}


class Net(nn.Module):
    def __init__(self, architecture, archversion, load_weights=True, frozen=True):
        super(Net, self).__init__()
        self.architecture = architecture
        self.version = archversion
        self.load_weights = load_weights
        self.frozen = frozen

        #Select architecture (July 2018 versions) and transfer learning from imagenet (pytorch default)
        if architecture == "ResNet":
            if archversion == '18':
                self.model = models.resnet18(pretrained=load_weights)
            if archversion == '34':
                self.model = models.resnet34(pretrained=load_weights)
            if archversion == '50':
                self.model = models.resnet50(pretrained=load_weights)
            if archversion == '101':
                self.model = models.resnet101(pretrained=load_weights)
            if archversion == '152':
                self.model = models.resnet152(pretrained=load_weights)

        elif architecture == "SqueezeNet":
            if archversion == '1.0':
                self.model = models.squeezenet1_0(pretrained=load_weights)
            if archversion == '1.1':
                self.model = models.squeezenet1_1(pretrained=load_weights)
        
        elif architecture == "Densenet":
            if archversion == '121':
                self.model = models.densenet_121(pretrained=load_weights)
            if archversion == '169':
                self.model = models.densenet_169(pretrained=load_weights)
            if archversion == '161':
                self.model = models.densenet_161(pretrained=load_weights)
            if archversion == '201':
                self.model = models.densenet_201(pretrained=load_weights)

        elif architecture == "Inception":
            if archversion == 'v3':
                self.model = models.inception_v3(pretrained=load_weights)

        else:
            self.model = False

        #Freeze layers for fine tunning if needed
        if frozen == True:
            for param in self.model.parameters():
                param.requires_grad = False

        
	def forward_once(self, x):
		#print('input', x.size())
		x = self.model(x)
		#print('resnet out ', x.size())
		#x = x.view(-1, 2048 * 1 * 1)
		#x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		#print('view out ', x.size())
		return x
    
    def siamese_forward(self, input_vector=list()):
        lenght = len(inputvector)
        all_outputs = list()
        if lenght > 0:
            for input_image in input_vector:
                output = self.forward_once(input_image)
                all_outputs.append(output)
                
            p = torch.cat(all_outputs,1)
            #p = self.predict(p)
            return p

        else:
            return False

    


    