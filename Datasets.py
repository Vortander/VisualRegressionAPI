# coding: utf-8

import cv2
import os, sys, copy
import numpy as np
import h5py
from scipy.misc import imread

from PIL import Image

import random
from random import shuffle

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

import torchvision.transforms as transforms


#Get pointlist size
def get_pointlist_size(citypointlist):
	fr = open(citypointlist, 'r')
	lines = fr.readlines()
	fr.close()

	return len(lines)

# Get point list from citypoints
def get_pointlist(citypointlist, randomize=False):

	fr = open(citypointlist, 'r')
	lines = fr.readlines()
	fr.close()

	points = []
	for line in lines:
		line = line.replace('\n', '')
		_id, cell, lat, lon, attr = line.split(';')

		points.append([_id, cell, lat, lon, attr])

	if randomize == True:
		random.shuffle(points)

	return points

# Get random points for validation
def get_validation_pointlist(citypointlist, validation_group_size=10, save=False, path_file=None):

	points = get_pointlist(citypointlist)

	validationpoints = []
	random.shuffle(points)
	for turn in range(0, validation_group_size):
		sample = points.pop()
		validationpoints.append(sample)

	if save!=False:
		torch.save(validationpoints, path_file)
	return validationpoints

# Get random pointlist batches, remove validation points if exists
def generate_random_pointlists(citypointlist=None, pointsarray=None, group_size=32, validationpoints=None, max_elements=None, save=False, path_file=None):
	training_points = []
	points = []
	if pointsarray != None:
		all_points = pointsarray
		points = copy.deepcopy(all_points)
	else:
		all_points = get_pointlist(citypointlist, randomize=True)

	if max_elements != None:
		all_points = all_points[0:max_elements]

	if validationpoints != None:
		points = copy.deepcopy(all_points)

		count = 0
		for v in validationpoints:
			points.remove(v)
			sys.stdout.write('[%d/%d] \r' % ( count, len(validationpoints) ))
			sys.stdout.flush()
			count+=1

		training_points = copy.deepcopy(points)
	else:
		points = copy.deepcopy(all_points)
		training_points = copy.deepcopy(all_points)

	groups = int(len(points)/float(group_size))
	batch_groups = []
	random.shuffle(points)
	for turn in range(0, groups):
		#random.shuffle(points)
		group = []
		for g in range(0, group_size):
			sample = points.pop()
			group.append(sample)

		batch_groups.append(group)

	if save == True:
		torch.save(training_points, path_file)

	del all_points
	del points

	return batch_groups, training_points

#Get point from citypoints by ID
#TODO: Set specific city sector to diferentiate between cities.
def get_point_by_id(citypointlist, id):
	all_points = get_pointlist(citypointlist, randomize=False)

	batch_groups = []
	for point in all_points:
		if str(id) in point:
			batch_groups.append(point)

	return batch_groups

# Street-level images class
class StreetImages(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], resize=True, imgsize=(224,224), ext='.jpg', normalize=None, transform=None):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.resize = resize
		self.imgsize = imgsize
		self.ext = ext
		self.normalize = normalize
		self.transform = transform

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):

		image_block = []

		point = self.pointlist[idx]

		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		for c in self.camera_views:
			image_name = str(lat) + '_' + str(lon) + '_' + c + self.ext
			img = cv2.imread(os.path.join(self.source_path, image_name))

			pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')

			# if self.normalize is not None:
			# 	pixels = self.normalize(pixels)

			if self.transform is not None:
				pixels = self.transform(pixels)

			image_block.append(pixels)

		transformed_images = torch.from_numpy(np.array(image_block, dtype=np.uint8))

		sample = {'image': torch.stack([ image.permute(2, 0, 1) for image in transformed_images ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell }

		return sample

# Street-level images class
class StreetImagesPIL(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], resize=True, imgsize=(224,224), ext='.jpg', normalize=None, transform=None ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.resize = resize
		self.imgsize = imgsize
		self.camera_views = camera_views
		self.ext = ext
		self.normalize = normalize
		self.transform = transform

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):

		image_block = []

		point = self.pointlist[idx]

		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		for c in self.camera_views:
			image_name = str(lat) + '_' + str(lon) + '_' + c + self.ext

			#img = cv2.imread(os.path.join(self.source_path, image_name))
			# Open as PIL
			#TODO: Make a better error handling
			try:
				img = Image.open(os.path.join(self.source_path, image_name)).convert('RGB')
			except:
				print("Exception in StreetImagesPIL DataLoader: ", lat, lon, c)
				return False

			#pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')
			#Aply transforms Scale insted cv2.resize
			#scaler = transforms.Scale(self.imgsize)
			scaler = transforms.Resize(self.imgsize)
			to_tensor = transforms.ToTensor()

			pixels = to_tensor(scaler(img))

			if self.normalize is not None:
				pixels = self.normalize(pixels)

			if self.transform is not None:
				pixels = self.transform(pixels)


			#print(pixels)

			image_block.append(pixels)

		#transformed_images = torch.from_numpy(np.array(image_block, dtype=np.uint8))
		transformed_images = image_block

		sample = {'image': torch.stack([ image for image in transformed_images ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell }

		return sample

class StreetFeatures(Dataset):
	def __init__( self, pointlist, source_path, camera_views=['0','90','180','270'], ext={} ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.ext = ext

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):
		feature_block = []
		point = self.pointlist[idx]
		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		if type(self.source_path) is dict:
			key, sector = cell.split("-")
			source_path = self.source_path[key]
		else:
			source_path = self.source_path

		if type(self.ext) is dict:
			key, sector = cell.split("-")
			ext = self.ext[key]
		else:
			ext = self.ext

		for c in self.camera_views:
			feature_name = str(lat) + '_' + str(lon) + '_' + c + ext
			feature_array = torch.load(os.path.join(source_path, feature_name))
			feature = torch.from_numpy(feature_array['features'])
			feature_block.append(feature)

		sample = {'image': torch.stack([ feature for feature in feature_block ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell, 'lat_lon': str( str(lat)+ "_" + str(lon) ), 'full_content': str(str(_id) + ";" + str(cell) + ";" + str(lat) + ";" + str(lon) + ";" + str(attr)) }

		return sample

class StreetSatImages(Dataset):
	def __init__( self, pointlist, source_path={}, camera_views={}, resize=True, imgsize=(224,224), ext={}, normalize=None ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.resize = resize
		self.imgsize = imgsize
		self.ext = ext
		self.normalize = normalize

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):

		image_sat_block = []
		image_street_block = []

		point = self.pointlist[idx]
		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		source_path_sat = self.source_path['Sat']
		source_path_street = self.source_path['Street']
		ext_sat = self.ext['Sat']
		ext_street = self.ext['Street']

		for c in self.camera_views['Sat']:
			image_name = str(lat) + '_' + str(lon) + '_' + c + ext_sat
			img = cv2.imread(os.path.join(source_path_sat, image_name))
			pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')

			if self.normalize is not None:
				pixels = self.normalize(pixels)

			image_sat_block.append(pixels)

		for c in self.camera_views['Street']:
			image_name = str(lat) + '_' + str(lon) + '_' + c + ext_street
			img = cv2.imread(os.path.join(source_path_street, image_name))
			pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')

			if self.normalize is not None:
				pixels = self.normalize(pixels)

			image_street_block.append(pixels)

		transformed_sat_images = torch.from_numpy(np.array(image_sat_block, dtype=np.uint8))
		transformed_street_images = torch.from_numpy(np.array(image_street_block, dtype=np.uint8))

		sample = {'street_image': torch.stack([ image.permute(2, 0, 1) for image in transformed_street_images ]), 'sat_image': torch.stack([ image.permute(2, 0, 1) for image in transformed_sat_images ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell }

		return sample


class StreetSatFeatures(Dataset):
	def __init__( self, pointlist, source_path={}, camera_views={}, ext={}, multicity=False ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.ext = ext
		self.multicity = multicity

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):
		feature_sat_block = []
		feature_street_block = []

		point = self.pointlist[idx]
		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		if self.multicity == True:
			key, sector = cell.split("-")
			source_path_sat = self.source_path['Sat'][key]
			source_path_street = self.source_path['Street'][key]
			ext_sat = self.ext['Sat'][key]
			ext_street = self.ext['Street'][key]
		else:
			source_path_sat = self.source_path['Sat']
			source_path_street = self.source_path['Street']
			ext_sat = self.ext['Sat']
			ext_street = self.ext['Street']

		for c in self.camera_views['Sat']:
			feature_name_sat = str(lat) + '_' + str(lon) + '_' + c + ext_sat
			feature_array_sat = torch.load(os.path.join(source_path_sat, feature_name_sat))
			feature_sat = torch.from_numpy(feature_array_sat['features'])
			feature_sat_block.append(feature_sat)

		for c in self.camera_views['Street']:
			feature_name_street = str(lat) + '_' + str(lon) + '_' + c + ext_street
			feature_array_street = torch.load(os.path.join(source_path_street, feature_name_street))
			feature_street = torch.from_numpy(feature_array_street['features'])
			feature_street_block.append(feature_street)

		sample = {'street_image': torch.stack([ feature for feature in feature_street_block ]), 'sat_image': torch.stack([ feature for feature in feature_sat_block ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell, 'lat_lon': str( str(lat)+ "_" + str(lon) ), 'full_content': str(str(_id) + ";" + str(cell) + ";" + str(lat) + ";" + str(lon) + ";" + str(attr)) }

		return sample

