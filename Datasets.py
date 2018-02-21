# coding: utf-8

import cv2
import os, sys
import numpy as np
import h5py
from scipy.misc import imread

import random
from random import shuffle

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# Plot distribution of citypointslist
def plot_pointlist_distribution(ciytpointlist):
	fr = open(ciytpointlist, 'r')
	points = fr.readlines()

	print(len(points))

	attributes = []
	for p in points:
		p = p.replace('\n', '')
		lat, lon, attr = p.split(';')

		attributes.append(attr)

	ord_attributes = sorted(attributes)
	
	plt.plot(ord_attributes, 'o')
	plt.show()



def generate_random_pointlists(ciytpointlist, group_size=32):
	fr = open(ciytpointlist, 'r')
	lines = fr.readlines()
	fr.close()

	points = []
	for line in lines:
		line = line.replace('\n', '')
		_id, cell, lat, lon, attr = line.split(';')

		points.append([_id, cell, lat, lon, attr])

	groups = int(len(points)/float(group_size))

	print("Generating random batch groups...")
	print("Total points: ", len(points))
	print("Group size: ", group_size)
	print("Groups total_points/group_size: ", groups)

	batch_groups = []
	for turn in range(0, groups):
		random.shuffle(points)
		group = []
		for g in range(0, group_size):
			sample = points.pop()
			group.append(sample)

		batch_groups.append(group)

	print("Total batch groups created: ", len(batch_groups))
	return batch_groups


def get_pointlist(ciytpointlist, randomize=True):
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

	print("Total points: ", len(points))
	return points


def imshow(img):
	print(img.shape)
	npimg = img.numpy()
	plt.axis("off")
	plt.imshow(cv2.cvtColor(np.transpose(npimg, (1, 2, 0)), cv2.COLOR_BGR2RGB))
	plt.show()


# class StreetImages(Dataset):
# 	def __init__( self, pointlist, source_path, camera_views=['0','90','180','270'], resize=True, imgsize=(227,227), augmentation=False):
		
# 		self.data = {'images': [], 'labels': []}
		
# 		for point in pointlist:

# 			image_block = []

# 			_id = point[0]
# 			cell = point[1]
# 			lat = point[2]
# 			lon = point[3]
# 			attr = point[4]

# 			for c in camera_views:

# 				image_name = str(lat) + '_' + str(lon) + '_' + c + '.jpg'
				
# 				img = cv2.imread(os.path.join(source_path, image_name))
# 				pixels = np.array(cv2.resize(img, imgsize), dtype='uint8')
# 				image_block.append(pixels)


# 			self.data['images'].append(image_block)
# 			self.data['labels'].append(float(attr))


# 		self.data['images'] = torch.from_numpy(np.array(self.data['images'], dtype=np.uint8))
# 		self.data['labels'] = torch.from_numpy(np.array(self.data['labels'], dtype=np.float32))

# 	def __len__(self):
# 		return len(self.data['images'])

# 	def __getitem__(self, idx):
# 		return self.data['images'][idx][0].permute(2, 0, 1), self.data['images'][idx][1].permute(2, 0, 1), self.data['images'][idx][2].permute(2, 0, 1), self.data['images'][idx][3].permute(2, 0, 1), self.data['labels'][idx]


class StreetImages(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], resize=True, imgsize=(227,227) ):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.resize = resize
		self.imgsize = imgsize

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):	

		image_block = []
		
		point = self.pointlist[idx]
		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		for c in self.camera_views:
			image_name = str(lat) + '_' + str(lon) + '_' + c + '.jpg'
			img = cv2.imread(os.path.join(self.source_path, image_name))
			pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')
			image_block.append(pixels)

		transformed_images = torch.from_numpy(np.array(image_block, dtype=np.uint8))
		
		sample = {'image': torch.stack([ image.permute(2, 0, 1) for image in transformed_images ]), 'label': float(attr), 'id': _id,  'cell': cell } 

		return sample









# class StreetImages(Dataset):
# 	def __init__(self, path, imagefiles, h5_file_list, imagefiles_index, maximum):
	
# 		self.data = {'images': [], 'labels': []}
	
# 		for file in h5_file_list:
# 			random_index = randint(0, len(imagefiles_index[file])-1)
# 			pointindex = imagefiles_index[file][random_index]
# 			del imagefiles_index[file][random_index]

# 			if len(imagefiles_index[file]) == 0:
# 				imagefiles.remove(file)

# 			with h5py.File(path + file, "r") as hf:
# 				#self.data['images'].append = hf["images"][:, :]
# 				#self.data['labels'] = hf["labels"][:]
# 				self.data['images'].append(hf['images'][pointindex])
# 				self.data['labels'].append(hf['labels'][pointindex])

# 		flabels = np.array(self.data['labels'], dtype=np.float32)
# 		self.labels = [float(i)/float(maximum) for i in flabels]
		
# 		self.images = torch.from_numpy(np.array(self.data['images'], dtype=np.uint8))
# 		self.labels = torch.from_numpy(np.array(self.labels, dtype=np.float32))

# 	def __len__(self):
# 		return len(self.data['images'])

# 	def __getitem__(self, idx):
# 		return self.images[idx][0].permute(2, 0, 1), self.images[idx][1].permute(2, 0, 1), self.images[idx][2].permute(2, 0, 1), self.images[idx][3].permute(2, 0, 1), self.labels[idx]