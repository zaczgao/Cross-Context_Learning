#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import random
import numpy as np
from numpy.random import random_sample
import cv2
from PIL import ImageFilter, Image

import torchvision.transforms as transforms

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


class GaussianBlur(object):
	"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

	def __init__(self, sigma=[.1, 2.]):
		self.sigma = sigma

	def __call__(self, x):
		sigma = random.uniform(self.sigma[0], self.sigma[1])
		x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
		return x


if __name__ == "__main__":
	pass

