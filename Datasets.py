# coding: utf-8

import cv2
import os, sys
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

# Get point list from citypoints
def get_pointlist(ciytpointlist, randomize=False):

	fr = open(ciytpointlist, 'r')
	lines = fr.readlines()
	fr.close()

	points = []
	for line in lines:
		line = line.replace('\n', '')
		_id, cell, lat, lon, attr = line.split(';')

		points.append([_id, cell, lat, lon, attr])

	if randomize == True:
		random.shuffle(points)

	#print("Total points: ", len(points))
	return points

# Get random points for validation
def get_validation_pointlist(ciytpointlist, validation_group_size=10):

	points = get_pointlist(ciytpointlist)

	validationpoints = []
	for turn in range(0, validation_group_size):
		random.shuffle(points)
		sample = points.pop()
		validationpoints.append(sample)

	#print("Validation points: ", validationpoints)

	return validationpoints

# Get random pointlist batches, remove validation points if exists
def generate_random_pointlists(ciytpointlist, group_size=32, validationpoints=None):

	all_points = get_pointlist(ciytpointlist, randomize=True)

	if validationpoints != None:
		#print("Removing validation points...")
		points = [p for p in all_points if p[0] not in [v[0] for v in validationpoints]]
		#print("Validation points removed: ", validationpoints)
	else:
		points = all_points

	groups = int(len(points)/float(group_size))

	#print("Generating random batch groups...")
	#print("Total points: ", len(points))
	#print("Group size: ", group_size)
	#print("Groups total_points/group_size: ", groups)

	batch_groups = []
	for turn in range(0, groups):
		random.shuffle(points)
		group = []
		for g in range(0, group_size):
			sample = points.pop()
			group.append(sample)

		batch_groups.append(group)

	#print("Total batch groups created: ", len(batch_groups))
	return batch_groups

#Get point from citypoints by ID
def get_point_by_id(ciytpointlist, id):
	all_points = get_pointlist(ciytpointlist, randomize=False)

	batch_groups = []
	for point in all_points:
		if str(id) in point:
			batch_groups.append(point)

	return batch_groups


# Street-level images class
class StreetImages(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], resize=True, imgsize=(227,227), ext='.jpg' ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.resize = resize
		self.imgsize = imgsize
		self.ext = ext

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

			image_block.append(pixels)

		transformed_images = torch.from_numpy(np.array(image_block, dtype=np.uint8))

		sample = {'image': torch.stack([ image.permute(2, 0, 1) for image in transformed_images ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell }

		return sample

# Street-level images class
class StreetImagesPIL(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], resize=True, imgsize=(227,227), ext='.jpg', normalize=None ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.resize = resize
		self.imgsize = imgsize
		self.camera_views = camera_views
		self.ext = ext
		self.normalize = normalize

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
				return False

			#pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')
			#Aply transforms Scale insted cv2.resize
			scaler = transforms.Scale(self.imgsize)
			to_tensor = transforms.ToTensor()

			pixels = to_tensor(scaler(img))

			if self.normalize is not None:
				pixels = self.normalize(pixels)

			#print(pixels)

			image_block.append(pixels)

		#transformed_images = torch.from_numpy(np.array(image_block, dtype=np.uint8))
		transformed_images = image_block

		sample = {'image': torch.stack([ image for image in transformed_images ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell }

		return sample
