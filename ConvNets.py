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

class Net(nn.Module):
    def __init__(self, architecture='alexnet', dataset='ImageNet', load_weights=True, frozen=True, eval_mode=False):
        super(Net, self).__init__()

        self.architecture = architecture
        self.dataset = dataset
        self.load_weights = load_weights
        self.frozen = frozen
        self.remove_last_layer = remove_last_layer
        self.eval_mode = eval_mode

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

        if self.eval_mode == True:
            self.model.eval()

    def get_model_layer(self, layer_index='default'):
        if self.architecture == 'alexnet':
            if layer_index == 'default':
                layer = self.model.classifier[-2]
            else:
                layer = self.model.classifier[-int(layer)]

        if 'resnet' in self.architecture:
            if layer_index == 'default':
                layer = self.model._modules.get('avgpool')
            else:
                layer = self.model._modules[layer]

        if 'densenet' in self.architecture:
            if layer_index == 'default':
                layer = self.model._modules.get('features')
            else:
                layer = self.model._modules[layer]

        return layer

    def get_feature_vector(self, x, method='layer', feature_size=4096):
        if method == 'layer' and 'densenet':
            layer = self.get_model_layer()
            vector = torch.zeros(feature_size)

            def copy_data(m, i, o):
                if 'densenet' in self.architecture:
                    o = F.relu(o, inplace=True)
                    o = F.avg_pool2d(o, kernel_size=7).view(o.size(0), -1)

                vector.copy_(o.data)

            h = layer.register_forward_hook(copy_data)

            self.model(x)
            h.remove()

            return vector.numpy()

        elif method == 'model':
            if self.architecture == 'alexnet':
                #TODO: model method not working for alexnet
                model = self.model.features
                model_pop = nn.Sequential(*list(self.model.classifier.children())[:-1])
                f = model(x)
                f = f.view(f.size(0), -1)
                f = model_pop(f)

                output = f.cpu().data.numpy()[0]

            elif 'resnet' in self.architecture:
                model_pop = nn.Sequential(*list(self.model.children())[:-1])
                output = model_pop(x).view(1,feature_size).cpu().data.numpy()[0]

            elif 'densenet' in self.architecture:
                model_pop = nn.Sequential(*list(self.model.children())[:-1])
                f = model_pop(x)
                f = F.relu(f, inplace=True)
                f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)

                output = f.cpu().data.numpy()[0]

            return output

    def forward(self, x):
        print('input', x.size())
        x = self.model(x)
        # #x = x.view(-1, 2048 * 1 * 1)
        # #x = F.relu(self.fc1(x))
        # #x = F.relu(self.fc2(x))
        print('model out', x.size())
        return x

    def siamese_forward(self, input_vector=list()):
        lenght = len(input_vector)
        all_outputs = list()
        if lenght > 0:
            for input_image in input_vector:
                output = self.forward(input_image)
                all_outputs.append(output)

            p = torch.cat(all_outputs,1)
            #p = self.predict(p)
            return p

        else:
            return False

    def get_vector(self, image):
        my_embedding = torch.zeros()







