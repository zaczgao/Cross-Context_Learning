#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

import torch
import torch.nn as nn

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from models.base_model import BaseModel
from models.network import buildEncoder, desc_dim
from models.criterion import calcFinalLoss, ClassifierLoss
from optimizers import get_optimizer


def Classifier(dataset, arch, num_class):
	model, _ = buildEncoder(dataset, arch, num_class, 1)

	# freeze all layers but the last fc
	for name, param in model.named_parameters():
		if name not in ['fc.weight', 'fc.bias']:
			param.requires_grad = False
	# init the fc layer
	model.fc.weight.data.normal_(mean=0.0, std=0.01)
	model.fc.bias.data.zero_()

	return model


class ClassifierModel(BaseModel):
	def __init__(self, args, cfg, world_size):
		super().__init__(args, cfg, world_size)

		self.loss_names = ["cls"]
		self.model_names = ["net"]

		# define networks
		self.net = Classifier(cfg.DATA["type"], cfg.MODEL["arch"], cfg.DATA["num_classes"])
		self.init_net(syncBN=False)

		if self.isTrain:
			self.criterion = ClassifierLoss().to(self.device)

			parameters = list(filter(lambda p: p.requires_grad, self.net.parameters()))
			assert len(parameters) == 2  # fc.weight, fc.bias
			net = self.net.module.fc if hasattr(self.net, 'module') else self.net.fc
			self.optimizer = get_optimizer(cfg.OPTIM["type"], net.parameters(), lr=cfg.OPTIM["lr"],
			                               momentum=cfg.OPTIM["momentum"], weight_decay=cfg.OPTIM["weight_decay"],
			                               nesterov=cfg.OPTIM.get("nesterov", False))
			self.optimizers.append(self.optimizer)

	def set_input(self, input, target, index=None):
		self.img = input.cuda(non_blocking=True)
		self.target = target.cuda(non_blocking=True)

	def forward(self):
		# Casts operations to mixed precision
		if self.args.half_precision:
			with torch.cuda.amp.autocast():
				output = self.net(self.img)
		else:
			output = self.net(self.img)

		return output

	def backward(self, output, target):
		if self.args.half_precision:
			with torch.cuda.amp.autocast():
				loss_pack = self.criterion(output, target)
				loss = calcFinalLoss(loss_pack, self.loss_weight)
			# Scales the loss, and calls backward() to create scaled gradients
			self.scaler.scale(loss).backward()
		else:
			loss_pack = self.criterion(output, target)
			loss = calcFinalLoss(loss_pack, self.loss_weight)
			loss.backward()

		return loss, loss_pack

	def optimize_parameters(self):
		output = self.forward()

		self.optimizer.zero_grad()
		loss, loss_pack = self.backward(output, self.target)

		if self.args.half_precision:
			# Unscales gradients and calls or skips optimizer.step()
			self.scaler.step(self.optimizer)
			# Updates the scale for next iteration
			self.scaler.update()
		else:
			self.optimizer.step()

		return output, loss, loss_pack


if __name__ == '__main__':
	from fvcore.nn import FlopCountAnalysis
	from configs.cgh.imagenetcfg import ImageNetConfig

	model = Classifier("imagenet", "resnet50", 1000)
	print(model)

	x = torch.randn(16, 3, 224, 224)
	out = model(x)