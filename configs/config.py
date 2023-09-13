#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

__author__ = "GZ"

import os
import multiprocessing

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class BaseConfig(object):
	"""Base configuration class. For custom configurations, create a
	sub-class that inherits from this one and override properties
	that need to be changed.
	"""
	# Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
	# Useful if your code needs to do things differently depending on which
	# experiment is running.
	NAME = None  # Override in sub-classes

	MODEL = dict(
		type = "MoCo",
		feat_dim = 128,
		queue_len = 65536,
		momentum = 0.999,
		temp = 0.2,
		mlp = True,
		arch="resnet18"
	)

	OPTIM = dict(
		type="sgd",
		num_epoch=200,
		warm_up=0,
		lr_base=0.03,
		lr=0.,
		momentum=0.9,
		weight_decay=0.0001,
		schedule_type="cos",  # use cosine lr schedule
		schedule=[120, 160]  # learning rate schedule (when to drop lr by 10x)
	)

	DATA = dict(
		type = "tinyimagenet",
		root = os.path.join("../data", "tiny-imagenet-200"),
		batch_size = 256,
		workers = min(8, multiprocessing.cpu_count() - 2),
		num_classes = 200,
		transform = dict(
			type="ssl_transforms",
			img_size=224,
			mean=[0.4802, 0.4481, 0.3975],
			std=[0.2302, 0.2265, 0.2262],
		)
	)

	EVAL = dict(
		enabled=False,
		print_steps=20,  # print frequency
		save_freq=100,  # Number of epochs between checkpoints
		val_epoch=10,
		knn_k=20,
		knn_t=0.07
	)

	def __init__(self):
		"""Set values of computed attributes."""
		pass

	def display(self):
		"""Display Configuration values."""
		print("\nConfigurations:")
		for a in dir(self):
			if not a.startswith("__") and not callable(getattr(self, a)):
				print("{:30} {}".format(a, getattr(self, a)))
		print("\n")
