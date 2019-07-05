# coding: utf-8

import sys, os, re

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
        self.eval_mode = eval_mode

        if dataset == 'Places':
            # By https://github.com/CSAILVision/places365
            # If you use this dataset, please cite the authors: http://places2.csail.mit.edu/PAMI_places.pdf
            # @article{zhou2017places,
                #    title={Places: A 10 million Image Database for Scene Recognition},
                #    author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
                #    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
                #    year={2017},
                #    publisher={IEEE}
                #  }
            # load the pre-trained weights as in places365/run_placesCNN_basic.py
            model_file = '%s_places365.pth.tar' % self.architecture
            if not os.access(model_file, os.W_OK):
                weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
                os.system('wget ' + weight_url)

            self.model = models.__dict__[self.architecture](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

            if 'densenet' in self.architecture:
                #Code inside if statement from https://github.com/KaiyangZhou/deep-person-reid/issues/23
                # Please check the link for more information
                # This pattern is used to find old key versions of DenseNet models such as 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
                pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
                state_dict = checkpoint['state_dict']
                for key in list(state_dict.keys()):
                    res = pattern.match(key)
                    if res:
                        new_key = res.group(1) + res.group(2)
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]

                state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

            self.model.load_state_dict(state_dict)

        elif dataset == 'ImageNet':
            print(models.__dict__[self.architecture])

            self.model = models.__dict__[self.architecture](pretrained=load_weights)

        else:
            self.model = False

        #Freeze layers for fine tunning if needed
        if self.frozen == True:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.eval_mode == True:
            self.model.eval()

    def get_model_layer(self, block=None, target_layer=None, by_name=None):
        if self.architecture == 'alexnet':
            if block == 'features' and target_layer != None:
                layer = self.model.features[target_layer]
            elif block == 'classifier' and target_layer != None:
                layer = self.model.classifier[target_layer]

        if 'resnet' in self.architecture:
            if target_layer != None:
                layer = self.model._modules.get(by_name)
            #else:
            #    layer = self.model._modules[target_layer]

        if 'densenet' in self.architecture:
            if block == 'features' and target_layer != None:
                layer = self.model.features[target_layer]
            elif block == 'classifier' and target_layer != None:
                layer = self.model.classifier

            #else:
            #    layer = self.model._modules[target_layer]

        return layer

    def get_feature_vector(self, x, method='layer', target_layer=None):
        if method == 'layer':
            layer = self.get_model_layer(block=target_layer[0], target_layer=target_layer[1]-1, by_name=target_layer[3])
            print(layer)
            print(target_layer)
            vector = torch.zeros([1, target_layer[2]])

            def copy_data(m, i, o):
                #If densenet norm5 layer
                if 'densenet' in self.architecture and target_layer[1] == 12:
                    o = F.relu(o, inplace=True)
                    o = F.adaptive_avg_pool2d(o, (1, 1)).view(o.size(0), -1)
                    print(o)

                vector.copy_(o.data)

            h = layer.register_forward_hook(copy_data)

            self.model(x)
            h.remove()

            return vector.numpy()[0]

        elif method == 'model':
            if self.architecture == 'alexnet':
                features = [f for f in self.model.features]
                classifier = [c for c in self.model.classifier]
                if target_layer[0] == 'features':
                    model_features = nn.Sequential(*list(features[0:target_layer[1]]))
                    output = model_features(x)
                    output = output.view(output.size(0), 256 * 6 * 6)

                elif target_layer[0] == 'classifier':
                    model_features = nn.Sequential(*list(features[0:13]))
                    model_classifier = nn.Sequential(*list(classifier[0:target_layer[1]]))
                    output = model_features(x)
                    output = output.view(output.size(0), 256 * 6 * 6)
                    output = model_classifier(output)

                output = output.cpu().data.numpy()[0]

            elif 'resnet18' in self.architecture:
                if target_layer[0] == 'features':
                    model_features = nn.Sequential(*list(self.model.children())[0:target_layer[1]])
                    output = model_features(x)
                    output = output.view(output.size(0), target_layer[2])
                elif target_layer[0] == 'classifier':
                    model_features = nn.Sequential(*list(self.model.children())[0:9])
                    model_classifier = nn.Sequential(*list(self.model.children())[9:10])
                    output = model_features(x)
                    output = output.view(output.size(0), 512 * 1 * 1)
                    output = model_classifier(output)

                output = output.cpu().data.numpy()[0]

            elif 'resnet50' in self.architecture:
                if target_layer[0] == 'features':
                    model_features = nn.Sequential(*list(self.model.children())[0:target_layer[1]])
                    output = model_features(x)
                    output = output.view(output.size(0), target_layer[2])
                elif target_layer[0] == 'classifier':
                    model_features = nn.Sequential(*list(self.model.children())[0:9])
                    model_classifier = nn.Sequential(*list(self.model.children())[9:10])
                    output = model_features(x)
                    output = output.view(output.size(0), 2048 * 1 * 1)
                    output = model_classifier(output)

                output = output.cpu().data.numpy()[0]

            elif 'resnet101' in self.architecture:
                if target_layer[0] == 'features':
                    model_features = nn.Sequential(*list(self.model.children())[0:target_layer[1]])
                    output = model_features(x)
                    output = output.view(output.size(0), target_layer[2])
                elif target_layer[0] == 'classifier':
                    model_features = nn.Sequential(*list(self.model.children())[0:9])
                    model_classifier = nn.Sequential(*list(self.model.children())[9:10])
                    output = model_features(x)
                    output = output.view(output.size(0), 2048 * 1 * 1)
                    output = model_classifier(output)

                output = output.cpu().data.numpy()[0]

            elif 'densenet' in self.architecture:
                features = [f for f in self.model.features]
                classifier = self.model.classifier

                if target_layer[0] == 'features':
                    model_features = nn.Sequential(*list(features[0:target_layer[1]]))
                    features = model_features(x)

                    output = F.relu(features, inplace=True)
                    output = F.adaptive_avg_pool2d(output, (1, 1)).view(features.size(0), -1)

                elif target_layer[0] == 'classifier':
                    model_features = nn.Sequential(*list(features[0:12]))
                    model_classifier = nn.Sequential(classifier)
                    features = model_features(x)
                    output = F.relu(features, inplace=True)
                    output = F.adaptive_avg_pool2d(output, (1, 1)).view(features.size(0), -1)
                    output = model_classifier(output)

                output = output.cpu().data.numpy()[0]

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







