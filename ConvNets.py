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

architectures = {'alexnet', 'resnet18', 'resnet50', 'densenet161'}

class Net(nn.Module):
    def __init__(self, architecture='alexnet', dataset='ImageNet', load_weights=True, frozen=True, remove_last_layer=False):
        super(Net, self).__init__()
        self.architecture = architecture
        self.dataset = dataset
        self.load_weights = load_weights
        self.frozen = frozen
        self.remove_last_layer = remove_last_layer

        if dataset == 'Places':
            # load the pre-trained weights as in places365/run_placesCNN_basic.py
            model_file = '%s_places365.pth.tar' % self.architecture
            if not os.access(model_file, os.W_OK):
                weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                os.system('wget ' + weight_url)

            self.model = models.__dict__[self.architecture](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

            self.model.load_state_dict(state_dict)

        elif dataset == 'ImageNet':
            self.model = models.__dict__[self.architecture](pretrained=load_weights)

        else:
            self.model = False

        #Freeze layers for fine tunning if needed
        if self.frozen == True:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.remove_last_layer == True:
            self.model = nn.Sequential(*list(self.model.children())[:-1])


	def forward_once(self, x):
		#print('input', x.size())
		x = self.model(x)
		#print('resnet out ', x.size())
		#x = x.view(-1, 2048 * 1 * 1)
		#x = F.relu(self.fc1(x))
		#x = F.relu(self.fc2(x))
		print('model out', x.size())
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




