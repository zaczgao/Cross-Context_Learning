#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


class TinyImageNet(VisionDataset):
	"""`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

		Args:
			root (string): Root directory of the dataset.
			split (string, optional): The dataset split, supports ``train``, or ``val``.
			transform (callable, optional): A function/transform that  takes in an PIL image
			   and returns a transformed version. E.g, ``transforms.RandomCrop``
			target_transform (callable, optional): A function/transform that takes in the
			   target and transforms it.
			download (bool, optional): If true, downloads the dataset from the internet and
			   puts it in root directory. If dataset is already downloaded, it is not
			   downloaded again.
	"""
	url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
	filename = 'tiny-imagenet-200.zip'
	md5 = '90528d7ca1a48142e341f4ef8d21d0de'

	def __init__(self, datasetDir, split='train', transform=None, target_transform=None, in_memory=False):
		super().__init__(os.path.join(datasetDir, ".."), transform=transform, target_transform=target_transform)

		self.dataset_path = datasetDir
		self.loader = default_loader
		self.split = verify_str_arg(split, "split", ("train", "val",))
		self.in_memory = in_memory

		classes, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))
		class_to_name = find_classes_names(os.path.join(self.dataset_path, 'words.txt'), class_to_idx)

		self.classes = classes
		self.class_to_idx = class_to_idx
		self.data = make_dataset(self.dataset_path, self.split, class_to_idx)
		self.samples = self.data
		self.targets = [s[1] for s in self.samples]

		# read all images into torch tensor in memory to minimize disk IO overhead
		if self.in_memory:
			self.images = [self.loader(img_path) for img_path, target in self.samples]

	def __getitem__(self, index):
		img_path, target = self.samples[index]
		if self.in_memory:
			image = self.images[index]
		else:
			image = self.loader(img_path)

		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return image, target, index

	def __len__(self):
		return len(self.samples)


def find_classes(class_file):
	with open(class_file) as r:
		classes = list(map(lambda s: s.strip(), r.readlines()))

	classes.sort()
	class_to_idx = {classes[i]: i for i in range(len(classes))}

	return classes, class_to_idx


def find_classes_names(class_file, class_to_idx):
	with open(class_file) as r:
		class_list = list(map(lambda s: s.strip(), r.readlines()))

	classes = dict()
	for line in class_list:
		[idx, target] = line.split("\t")
		classes[idx] = target

	class_to_name = dict()
	for idx in class_to_idx.keys():
		class_to_name[idx] = classes[idx]

	# with open("./class_to_name.txt", "w") as f:
	# 	for target in class_to_name.values():
	# 		f.write(target + "\n")
	return class_to_name


def make_dataset(datasetDir, dirname, class_to_idx):
	images = []
	dir_path = os.path.join(datasetDir, dirname)

	if dirname == 'train':
		for fname in sorted(os.listdir(dir_path)):
			cls_fpath = os.path.join(dir_path, fname)
			if os.path.isdir(cls_fpath):
				cls_imgs_path = os.path.join(cls_fpath, 'images')
				for imgname in sorted(os.listdir(cls_imgs_path)):
					path = os.path.join(cls_imgs_path, imgname)
					item = (path, class_to_idx[fname])
					images.append(item)
	else:
		imgs_path = os.path.join(dir_path, 'images')
		imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

		with open(imgs_annotations) as r:
			data_info = map(lambda s: s.split('\t'), r.readlines())

		cls_map = {line_data[0]: line_data[1] for line_data in data_info}

		for imgname in sorted(os.listdir(imgs_path)):
			path = os.path.join(imgs_path, imgname)
			item = (path, class_to_idx[cls_map[imgname]])
			images.append(item)

	return images


if __name__ == '__main__':
	from configs.moco.tinyimagenetcfg import TinyImageNetConfig

	cfg = TinyImageNetConfig("pretrain")
	train_dataset = TinyImageNet(cfg.DATA["root"], split="train", in_memory=True)
	test_dataset = TinyImageNet(cfg.DATA["root"], split="val", in_memory=True)
	print(len(train_dataset), train_dataset[0], train_dataset[-1])
	print(len(test_dataset), test_dataset[0], test_dataset[-1])
