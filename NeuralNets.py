# coding: utf-8

import sys, os, re
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn

from torchvision import models



#Baseline
class LinearRegression(nn.Module):
	def __init__(self, input_vector):
		super(LinearRegression, self).__init__()
		self.input_vector = input_vector
		self.linear = nn.Linear(self.input_vector, 1)

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

#9248 (HOG_8212: 9248 features)
class RegressionNet_FirstTry(nn.Module):
	def __init__(self, input_vector):
		super(RegressionNet_FirstTry, self).__init__()
		self.input_vector = input_vector
		self.fc1 = nn.Linear(self.input_vector, 4624)
		self.fc2 = nn.Linear(4624, 2312)
		self.fc3 = nn.Linear(2312, 1156)
		self.fc4 = nn.Linear(1156, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))

		return x

#9248 (HOG_8212: 9248 features)
#Smaller
#class RegressionNet_WithoutRelu(nn.Module): Same network from SatNet-DenseNet
class RegressionNet(nn.Module):
	def __init__(self, input_vector):
		super(RegressionNet, self).__init__()
		self.input_vector = input_vector
		self.fc1 = nn.Linear(self.input_vector, 1104)
		self.fc2 = nn.Linear(1104, 552)
		self.fc3 = nn.Linear(552, 276)
		self.fc4 = nn.Linear(276, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))

		return x

class StreetNet3_ForHOG(nn.Module):
	def __init__(self, feature_vector_size, output, dropout=[None, None, None, None]):
		super(StreetNet3_ForHOG, self).__init__()
		self.feature_vector_size = feature_vector_size
		self.output = output
		self.dropout = dropout

		self.fc1 = nn.Linear(self.feature_vector_size, 4416)
		self.fc2 = nn.Linear(4416, 2208)
		self.fc3 = nn.Linear(2208, 1104)
		self.fc4 = nn.Linear(1104, self.output)

	def forward(self, x):
		x = x.view(-1, self.feature_vector_size)  #Concat mode

		if self.dropout[0] != None:
			x = F.dropout(x, p=self.dropout[0], training=self.training)
			#print("Dropout Input activated")

		x = self.fc1(x)
		#x = self.norm1(x)
		x = F.relu(x)
		if self.dropout[1] != None:
			x = F.dropout(x, p=self.dropout[1], training=self.training)
			#print("Dropout 2 activated")

		x = self.fc2(x)
		#x = self.norm2(x)
		x = F.relu(x)
		if self.dropout[2] != None:
			x = F.dropout(x, p=self.dropout[2], training=self.training)
			#print("Dropout 3 activated")

		x = self.fc3(x)
		#x = self.norm3(x)
		x = F.relu(x)
		if self.dropout[3] != None:
			x = F.dropout(x, p=self.dropout[3], training=self.training)
			#print("Dropout 4 activated")

		#x = self.fc4(x)
		x = F.relu(self.fc4(x))
		return x


#Street_RAM:
#   Model using methodology of "Learning Deep Features for Discriminative Localization" (Zhou et al., 2016),
#   and "Diabetic Retinopathy Detection via Deep Convolutional Networks for Discriminative Localization and Visual Explanation" (Wang; Yang, 2017)
#   Difference: Tested with DenseNet-161.
class Street_RAM(nn.Module):
	def __init__(self, pretrained=False, places=True):
		super(Street_RAM, self).__init__()
		self.pretrained = pretrained
		self.places = places
		if places == False:
			self.densenet = models.densenet161(pretrained=self.pretrained)
		elif places == True:
			#Loading and setting DenseNet161 with Places weights.
			model_file = 'densenet161_places365.pth.tar'
			if not os.access(model_file, os.W_OK):
				weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
				os.system('wget ' + weight_url)

			self.densenet = models.densenet161(num_classes=365)
			checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
			print("Places365 weights loaded!")

			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
			state_dict = checkpoint['state_dict']
			for key in list(state_dict.keys()):
				res = pattern.match(key)
				if res:
					new_key = res.group(1) + res.group(2)
					state_dict[new_key] = state_dict[key]
					del state_dict[key]

			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			self.densenet.load_state_dict(state_dict)

		self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
		self.gap = nn.AdaptiveMaxPool2d((1,1))

		self.fc = nn.Linear(2208, 1)

	#As presented in Widersnet.py places365
	def forward_once(self, x):
		x = self.densenet(x)
		x = F.relu(x, inplace=True)
		x = self.gap(x)
		x = x.view(x.size(0), -1)
		return x

	def siamese_forward(self, input1, input2, input3, input4):
		output1 = self.forward_once(input1)
		del input1
		output2 = self.forward_once(input2)
		del input2
		output3 = self.forward_once(input3)
		del input3
		output4 = self.forward_once(input4)
		del input4

		x = torch.cat((output1, output2, output3, output4),1)
		x = F.relu(self.fc(x))

		return x

	def forward(self, _input):
		x = self.forward_once(_input)
		del _input
		x = F.relu(self.fc(x))

		return x

#StreetNet (Sat-3 for street-view, ResNet18 version)
class Street_ResNet18(nn.Module):
	def __init__(self, fc_dropout=[None, None, None]):
		super(Street_ResNet18, self).__init__()
		self.resnet = models.resnet18(pretrained=False)
		self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
		self.fc_dropout = fc_dropout

		self.fc1 = nn.Linear(512, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 1)

	def forward(self, x):
		#print('input', x.size())
		x = self.resnet(x)
		x = F.relu(x, inplace=True)
		x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

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

	def get_model_layer(self, block=None, target_layer=None, by_name=None):
		if block == 'resnet' and target_layer != None:
			layer = self.densenet
		elif block == 'fc4' and target_layer != None:
			layer = self.fc4

		return layer

	def get_feature_vector(self, x, target_layer=None):
		layer = self.get_model_layer(block=target_layer[0], target_layer=target_layer[1]-1, by_name=target_layer[3])
		vector = torch.zeros([1, target_layer[2]])

		def copy_data(m, i, o):
			#If densenet
			if target_layer[0] == 'resnet':
				o = F.relu(o, inplace=True)
				o = F.adaptive_avg_pool2d(o, (1, 1)).view(o.size(0), -1)
			vector.copy_(o.data)

		h = layer.register_forward_hook(copy_data)

		self(x)
		h.remove()

		return vector.numpy()[0]

#StreetNet (Sat-3 for street-view, DenseNet161 version)
class Street_DenseNet161(nn.Module):
	def __init__(self, fc_dropout=[None, None, None], pretrained=False, places=False):
		super(Street_DenseNet161, self).__init__()
		self.pretrained = pretrained
		self.places = places
		if places == False:
			self.densenet = models.densenet161(pretrained=self.pretrained)
		elif places == True:
			#Loading and setting DenseNet161 with Places weights.
			model_file = 'densenet161_places365.pth.tar'
			if not os.access(model_file, os.W_OK):
				weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
				os.system('wget ' + weight_url)

			self.densenet = models.densenet161(num_classes=365)
			checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
			print("Places365 weights loaded!")

			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
			state_dict = checkpoint['state_dict']
			for key in list(state_dict.keys()):
				res = pattern.match(key)
				if res:
					new_key = res.group(1) + res.group(2)
					state_dict[new_key] = state_dict[key]
					del state_dict[key]

			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			self.densenet.load_state_dict(state_dict)

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

	def get_model_layer(self, block=None, target_layer=None, by_name=None):
		if block == 'densenet' and target_layer != None:
			layer = self.densenet
		elif block == 'fc4' and target_layer != None:
			layer = self.fc4

		return layer

	def get_feature_vector(self, x, target_layer=None):
		layer = self.get_model_layer(block=target_layer[0], target_layer=target_layer[1]-1, by_name=target_layer[3])
		vector = torch.zeros([1, target_layer[2]])

		def copy_data(m, i, o):
			#If densenet
			if target_layer[0] == 'densenet':
				o = F.relu(o, inplace=True)
				o = F.adaptive_avg_pool2d(o, (1, 1)).view(o.size(0), -1)
			vector.copy_(o.data)

		h = layer.register_forward_hook(copy_data)

		self(x)
		h.remove()

		return vector.numpy()[0]

#StreetNet (StreetNet for street-view, DenseNet161-Places start version)
class AllStreet_DenseNet161(nn.Module):
	def __init__(self, fc_dropout=[None, None, None], pretrained=False, places=True):
		super(AllStreet_DenseNet161, self).__init__()
		self.pretrained = pretrained
		self.places = places
		if places == False:
			self.densenet = models.densenet161(pretrained=self.pretrained)
		elif places == True:
			#Loading and setting DenseNet161 with Places weights.
			model_file = 'densenet161_places365.pth.tar'
			if not os.access(model_file, os.W_OK):
				weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
				os.system('wget ' + weight_url)

			self.densenet = models.densenet161(num_classes=365)
			checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
			print("Places365 weights loaded!")

			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
			state_dict = checkpoint['state_dict']
			for key in list(state_dict.keys()):
				res = pattern.match(key)
				if res:
					new_key = res.group(1) + res.group(2)
					state_dict[new_key] = state_dict[key]
					del state_dict[key]

			state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
			self.densenet.load_state_dict(state_dict)

		self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
		self.fc_dropout = fc_dropout

		self.fc1 = nn.Linear(8832, 4416)
		self.fc2 = nn.Linear(4416, 2208)
		self.fc3 = nn.Linear(2208, 1104)
		self.fc4 = nn.Linear(1104, 1)

	def forward_once(self, x):
		x = self.densenet(x)
		x = F.relu(x, inplace=True)
		x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

		return x

	def forward(self, input1, input2, input3, input4):
		output1 = self.forward_once(input1)
		del input1
		output2 = self.forward_once(input2)
		del input2
		output3 = self.forward_once(input3)
		del input3
		output4 = self.forward_once(input4)
		del input4

		x = torch.cat((output1, output2, output3, output4),1)

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

	def get_model_layer(self, block=None, target_layer=None, by_name=None):
		if block == 'densenet' and target_layer != None:
			layer = self.densenet
		elif block == 'fc4' and target_layer != None:
			layer = self.fc4

		return layer

	def get_feature_vector(self, x, target_layer=None):
		layer = self.get_model_layer(block=target_layer[0], target_layer=target_layer[1]-1, by_name=target_layer[3])
		vector = torch.zeros([1, target_layer[2]])

		def copy_data(m, i, o):
			#If densenet
			if target_layer[0] == 'densenet':
				o = F.relu(o, inplace=True)
				o = F.adaptive_avg_pool2d(o, (1, 1)).view(o.size(0), -1)
			vector.copy_(o.data)

		h = layer.register_forward_hook(copy_data)

		self(x)
		h.remove()

		return vector.numpy()[0]



#Sat-3 (IJCNN-2019)
class SatNetNoBN(nn.Module):
	def __init__(self, fc_dropout=[None, None, None]):
		super(SatNetNoBN, self).__init__()
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

	def get_model_layer(self, block=None, target_layer=None, by_name=None):
		if block == 'densenet' and target_layer != None:
			layer = self.densenet
		elif block == 'fc4' and target_layer != None:
			layer = self.fc4

		return layer

	def get_feature_vector(self, x, target_layer=None, to_tensor=False):
		layer = self.get_model_layer(block=target_layer[0], target_layer=target_layer[1]-1, by_name=target_layer[3])
		vector = torch.zeros([1, target_layer[2]])

		def copy_data(m, i, o):
			#If densenet
			if target_layer[0] == 'densenet':
				o = F.relu(o, inplace=True)
				o = F.adaptive_avg_pool2d(o, (1, 1)).view(o.size(0), -1)
			vector.copy_(o.data)

		h = layer.register_forward_hook(copy_data)

		self(x)
		h.remove()

		return vector if to_tensor == True else vector.numpy()[0]

#Street-3 (IJCNN-2019)
class StreetNet3(nn.Module):
	def __init__(self, feature_vector_size, output, dropout=[None, None, None, None]):
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

		if self.dropout[0] != None:
			x = F.dropout(x, p=self.dropout[0], training=self.training)
			#print("Dropout Input activated")

		x = self.fc1(x)
		#x = self.norm1(x)
		x = F.relu(x)
		if self.dropout[1] != None:
			x = F.dropout(x, p=self.dropout[1], training=self.training)
			#print("Dropout 2 activated")

		x = self.fc2(x)
		#x = self.norm2(x)
		x = F.relu(x)
		if self.dropout[2] != None:
			x = F.dropout(x, p=self.dropout[2], training=self.training)
			#print("Dropout 3 activated")

		x = self.fc3(x)
		#x = self.norm3(x)
		x = F.relu(x)
		if self.dropout[3] != None:
			x = F.dropout(x, p=self.dropout[3], training=self.training)
			#print("Dropout 4 activated")

		#x = self.fc4(x)
		x = F.relu(self.fc4(x))
		return x





class DenseNet161_Places(nn.Module):
	def __init__(self, frozen=True, eval_mode=True ):
		super(DenseNet161_Places, self).__init__()

		self.frozen = frozen
		self.eval_mode = eval_mode

		#Loading and setting DenseNet161 with Places weights.
		model_file = 'densenet161_places365.pth.tar'
		if not os.access(model_file, os.W_OK):
			weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
			os.system('wget ' + weight_url)

		self.densenet = models.densenet161(num_classes=365)
		checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
		state_dict = checkpoint['state_dict']
		for key in list(state_dict.keys()):
			res = pattern.match(key)
			if res:
				new_key = res.group(1) + res.group(2)
				state_dict[new_key] = state_dict[key]
				del state_dict[key]

		state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
		self.densenet.load_state_dict(state_dict)

		#Freeze layers for fine tunning if needed
		if self.frozen == True:
			for param in self.densenet.parameters():
				param.requires_grad = False

		if self.eval_mode == True:
			self.densenet.eval()

		features = [f for f in self.densenet.features]
		self.densenet_features = nn.Sequential(*list(features[0:len(features)]))


	def forward(self, x):
		features = self.densenet_features(x)
		output = F.relu(features, inplace=True)
		output = F.adaptive_avg_pool2d(output, (1, 1)).view(features.size(0), -1)

		return output


class StreetNet3_ForAlex(nn.Module):
	def __init__(self, feature_vector_size, output, dropout=[None, None, None, None]):
		super(StreetNet3_ForAlex, self).__init__()
		self.feature_vector_size = feature_vector_size
		self.output = output
		self.first_hidden_layer = int(np.mean([feature_vector_size, output]) / 4.0)
		self.second_hidden_layer = int(np.mean([self.first_hidden_layer, output]) / 4.0)
		self.third_hidden_layer = int(np.mean([self.second_hidden_layer, output]) / 4.0)
		self.dropout = dropout

		self.fc1 = nn.Linear(self.feature_vector_size, self.first_hidden_layer)
		self.fc2 = nn.Linear(self.first_hidden_layer, self.second_hidden_layer)
		self.fc3 = nn.Linear(self.second_hidden_layer, self.third_hidden_layer)
		self.fc4 = nn.Linear(self.third_hidden_layer, output)

	def forward(self, x):
		x = x.view(-1, self.feature_vector_size)  #Concat mode
		if self.dropout[0] != None:
			x = F.dropout(x, p=self.dropout[0], training=self.training)
			#print("Dropout Input activated")

		x = self.fc1(x)
		#x = self.norm1(x)
		x = F.relu(x)
		if self.dropout[1] != None:
			x = F.dropout(x, p=self.dropout[1], training=self.training)
			#print("Dropout 2 activated")

		x = self.fc2(x)
		#x = self.norm2(x)
		x = F.relu(x)
		if self.dropout[2] != None:
			x = F.dropout(x, p=self.dropout[2], training=self.training)
			#print("Dropout 3 activated")

		x = self.fc3(x)
		#x = self.norm3(x)
		x = F.relu(x)
		if self.dropout[3] != None:
			x = F.dropout(x, p=self.dropout[3], training=self.training)
			#print("Dropout 4 activated")

		#x = self.fc4(x)
		x = F.relu(self.fc4(x))
		return x

#StreetSat1 (IJCNN-2019)
class OldNet3(nn.Module):
	def __init__(self, feature_vector_size, output, dropout=[None, None]):
		super(OldNet3, self).__init__()
		self.feature_vector_size = feature_vector_size
		self.output = output
		self.dropout = dropout

		self.fc1 = nn.Linear(self.feature_vector_size, 1000)
		self.norm1 = nn.BatchNorm1d( 1000 )
		self.fc2 = nn.Linear(1000, 1)
		#self.norm2 = nn.BatchNorm1d( self.second_hidden_layer )

		self.predict = torch.nn.Linear( 1 * 5, 1)

	def forward_once(self, x):
		x = x.view(-1, self.feature_vector_size * 1)
		if self.dropout[0] != None:
			x = F.dropout(x, p=self.dropout[0], training=self.training)

		x = self.fc1(x)
		x = self.norm1(x)
		x = F.relu(x)
		if self.dropout[1] != None:
			x = F.dropout(x, p=self.dropout[1], training=self.training)

		x = F.relu(self.fc2(x))

		return x

	def forward(self, input1, input2, input3, input4, input5):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		output3 = self.forward_once(input3)
		output4 = self.forward_once(input4)
		output5 = self.forward_once(input5)

		p = torch.cat((output1, output2, output3, output4, output5), 1)
		p = p.view(-1, 1 * 5)
		#all_p = p
		p = F.relu(self.predict(p))
		#out_p = p
		#print(all_p[0], out_p[0])

		return p


#IJCN-2018
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

