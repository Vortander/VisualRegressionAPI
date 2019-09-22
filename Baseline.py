#coding: utf-8

import cv2
import skimage.feature as feature
from PIL import Image

#from https://github.com/tuttieee/lear-gist-python
import gist

import numpy as np
import os, glob, sys

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms


#image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm=None, visualize=False, visualise=None, transform_sqrt=False, feature_vector=True, multichannel=None
class StreetImagesHOG(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], ext='.jpg', resize=True, imgsize=(224,224), hog_orientations=9, hog_pixels_per_cell=(8,8), hog_cells_per_block=(3,3), hog_block_norm='L2-Hys', hog_visualize=True, hog_transform_sqrt=False, hog_feature_vector=True, hog_gray=False, hog_multichannel=True):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.resize = resize
		self.imgsize = imgsize
		self.ext = ext

		self.orientations = hog_orientations
		self.pixels_per_cell = hog_pixels_per_cell
		self.cells_per_block = hog_cells_per_block
		self.block_norm = hog_block_norm
		self.visualize = hog_visualize
		self.transform_sqrt = hog_transform_sqrt
		self.feature_vector = hog_feature_vector
		self.gray = hog_gray
		self.multichannel = hog_multichannel

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):

		hog_feature_block = []
		hog_image_block = []
		image_cam = []

		point = self.pointlist[idx]

		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		for c in self.camera_views:
			image_name = str(lat) + '_' + str(lon) + '_' + c + self.ext
			#try:
			img = Image.open(os.path.join(self.source_path, image_name)).convert('RGB')
			#except:
			#    print("Exception in StreetImagesPIL DataLoader: ",_id, lat, lon, c)
			#    return False
			# try:
			#     img = cv2.imread(os.path.join(self.source_path, image_name))
			#     print(img)
			# except:
			#     print("Exception in StreetImagesPIL DataLoader: ",_id, lat, lon, c)
			#     return False
			scaler = transforms.Resize(self.imgsize)
			pixels = scaler(img)

			#img = cv2.imread(os.path.join(self.source_path, image_name))

			#pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')

			# if self.gray and self.multichannel == False:
			#     pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)

			H, hog_image = feature.hog(pixels, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, block_norm=self.block_norm, visualize=self.visualize, transform_sqrt=self.transform_sqrt, feature_vector=self.feature_vector, multichannel=self.multichannel)

			hog_feature_block.append(H)
			hog_image_block.append(hog_image)
			image_cam.append(c)

		transformed_hog = torch.from_numpy(np.array(hog_feature_block, dtype=np.float))

		sample = {'hog': torch.stack([ h for h in transformed_hog ]), 'hog_image': np.stack([him for him in hog_image_block]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell, 'cam': np.stack(np.array([int(c) for c in image_cam])) }
		#sample = {'hog': np.stack([ h for h in hog_feature_block ]), 'label': np.array([float(attr)]), 'id': _id,  'cell': cell }

		return sample


class StreetImagesGIST(Dataset):
	def __init__( self, pointlist, source_path, camera_views = ['0','90','180','270'], ext='.jpg', resize=True, imgsize=(224,224), nblocks=4, orientations_per_scale=(8, 8, 4)):
		self.pointlist = pointlist
		self.source_path = source_path
		self.camera_views = camera_views
		self.resize = resize
		self.imgsize = imgsize
		self.ext = ext

		self.nblocks = nblocks
		self.orientations_per_scale = orientations_per_scale

	def __len__(self):
		return len(self.pointlist)

	def __getitem__(self, idx):

		gist_feature_block = []

		point = self.pointlist[idx]

		_id, cell, lat, lon, attr = point[0], point[1], point[2], point[3], point[4]

		for c in self.camera_views:
			image_name = str(lat) + '_' + str(lon) + '_' + c + self.ext
			img = cv2.imread(os.path.join(self.source_path, image_name))

			pixels = np.array(cv2.resize(img, self.imgsize), dtype='uint8')
			desc = gist.extract(pixels, nblocks=self.nblocks, orientations_per_scale=self.orientations_per_scale)

			gist_feature_block.append(desc)

		transformed_gist = torch.from_numpy(np.array(gist_feature_block, dtype=np.float))

		sample = {'gist': torch.stack([ h for h in transformed_gist ]), 'label': torch.from_numpy(np.array([float(attr)])), 'id': _id,  'cell': cell }

		return sample
