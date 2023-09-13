#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


if __name__ == '__main__':
	from configs.moco.cifar10cfg import Cifar10Config

	cfg = Cifar10Config("pretrain")
	transform = transforms.Compose([
		transforms.ToTensor()
	])
	dataset = datasets.CIFAR10(cfg.DATA_ROOT, train=True, transform=transform, download=True)
	loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

	for i, (images, target) in enumerate(loader):
		print(target)
