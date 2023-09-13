#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import multiprocessing

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from configs.config import BaseConfig


class ImageNetConfig(BaseConfig):
	"""Configuration for training on ImageNet.
	Derives from the base Config class and overrides values specific
	to the ImageNet dataset.
	"""
	NAME = "ImageNet"

	MODEL = dict(
		type="CGH",
		feat_dim=512,
		hid_dim=4096,
		hyper_dim=512,
		queue_len=65536*2,
		momentum=0.999,
		temp=0.2,
		temp_t=0.08,
		temp_s=0.08,
		layers=["layer3", "layer4"],
		mlp=True,
		norm_center=False,
		arch="resnet50"
	)

	OPTIM = dict(
		type="sgd",
		num_epoch=200,
		warm_up=5,
		lr_base=0.05,
		lr=0.,
		momentum=0.9,
		weight_decay=0.0001,
		schedule_type="cos",  # use cosine lr schedule
		schedule=[120, 160]  # learning rate schedule (when to drop lr by 10x)
	)

	DATA = dict(
		type="imagenet",
		root="/import/nobackup_mmv_ioannisp/shared/datasets/ImageNet",
		root_val="/import/nobackup_mmv_ioannisp/zg002/data/ImageNet/val",
		batch_size=256,
		workers=min(16, multiprocessing.cpu_count() - 2),
		num_classes=1000,
		transform=dict(
			type="ressl_transforms",
			multi=False,
			img_size=224,
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		)
	)

	LOSS = dict(
		type="cghloss",
		warm_up_loss=5,
		weight=dict(
			cls=1.0
		)
	)

	EVAL = dict(
		enabled=True,
		print_steps=200,  # print frequency
		save_freq=100,  # Number of epochs between checkpoints
		val_epoch=10,
		knn_k=20,
		knn_t=0.07
	)

	def __init__(self, phase, debug=False):
		"""Set values of computed attributes."""
		super().__init__()

		self.DEBUG = debug
		if debug:
			self.DATA["batch_size"] = 16

		self.OPTIM["lr"] = self.OPTIM["lr_base"] * self.DATA["batch_size"] / 256

		self.PHASE = phase
		if phase == "lincls":
			self.DATA["batch_size"] = 256
			self.OPTIM["type"] = "sgd"
			self.OPTIM["num_epoch"] = 100
			self.OPTIM["warm_up"] = 0
			self.OPTIM["lr_base"] = 0
			self.OPTIM["lr"] = 0.3
			self.OPTIM["momentum"] = 0.9
			self.OPTIM["weight_decay"] = 0.
			self.OPTIM["schedule_type"] = "cos"
			self.OPTIM["schedule"] = [60, 80]


if __name__ == '__main__':
	pass
