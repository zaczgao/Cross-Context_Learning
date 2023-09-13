#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from classy_vision.generic.distributed_util import get_world_size, get_rank, is_distributed_training_run, \
	gather_from_all, get_cuda_device_index

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from utils.util import normalize_batch, denormalize_batch, gram_matrix


def calcFinalLoss(loss_pack, loss_weights_dict):
	loss_list = []
	for key in loss_pack.keys():
		loss_list.append(loss_weights_dict[key] * loss_pack[key])
	loss = torch.stack(loss_list, 0).sum()
	# loss = sum(loss_list)

	return loss


def calcMSE(input, target, mask=None):
	if mask is not None:
		assert mask.ndim == 4
		# divider = mask.mean((1, 2, 3))
		# diffMask = torch.pow(output - target, 2) * mask
		# diff = diffMask.mean((1, 2, 3)) / (divider + 1e-12)
		diff = torch.pow(input - target, 2) * mask
	else:
		diff = torch.pow(input - target, 2)

	loss = torch.mean(diff)
	return loss


def calcL1Loss(input, target, mask=None):
	if mask is not None:
		assert mask.ndim == 4
		diff = torch.abs(input - target) * mask
	else:
		diff = torch.abs(input - target)

	loss = torch.mean(diff)
	return loss


def calcGramLoss(input, target):
	gm_y = gram_matrix(input)
	gm_s = gram_matrix(target)
	loss = F.mse_loss(gm_y, gm_s)
	return loss


def calc_crossentropy(logits_q, logits_k):
	assert logits_k.dim() == 2
	# assert torch.allclose(logits_k.sum(dim=1).cpu(), torch.ones(logits_k.shape[0]), atol=1e-06)
	loss = torch.sum(-logits_k * F.log_softmax(logits_q, dim=1), dim=1).mean()
	return loss


def loss_mask(map, mask):
	mask = F.interpolate(mask, map.shape[2:], mode="bilinear", align_corners=True)
	return map * mask


def calcCosSim(x, y, clip=False, version='original'):
	if version == 'original':
		x = F.normalize(x, dim=-1, p=2)
		y = F.normalize(y, dim=-1, p=2)
		sim = (x * y).sum(dim=-1)
	elif version == 'simplified':
		sim = F.cosine_similarity(x, y, dim=-1)

	if clip:
		sim = torch.clamp(sim, min=0.0005, max=0.9995)

	return sim


class CGHLoss(nn.Module):
	def __init__(self, temp_t=0.04, temp_s=0.1):
		super().__init__()

		self.temp_t = temp_t
		self.temp_s = temp_s
		self.cl = nn.CrossEntropyLoss()

		version = torch.__version__.split(".")
		if int(version[0]) == 1 and int(version[1]) >= 10:
			self.ce = nn.CrossEntropyLoss()
		else:
			self.ce = calc_crossentropy

	def forward(self, logits_q, logits_k, logits_q_hyper, logits_k_hyper, logits, labels, logits_hyper, labels_hyper,
	            local_logits_q, local_logits_k):
		loss_pack = {}

		if logits is not None:
			loss_pack["cl"] = self.cl(logits, labels) + self.cl(logits_hyper, labels_hyper)
		else:
			loss_pack["cl"] = torch.tensor(0.0).cuda()

		temp_t = 0.04
		temp_s = 0.1
		loss_pack["cl_hyper"] = self.ce(logits_q_hyper / self.temp_s, F.softmax(logits_k.detach() / temp_t, dim=1))
		loss_pack["csist"] = self.ce(logits_q / temp_s, F.softmax(logits_k_hyper.detach() / self.temp_t, dim=1))

		loss_local = 0
		if local_logits_q is not None:
			for vid, (local_q, local_k) in enumerate(zip(local_logits_q, local_logits_k)):
				loss_local += self.ce(local_q / temp_s, F.softmax(local_k.detach() / self.temp_t, dim=1))
			loss_pack["csist"] += loss_local
			loss_pack["csist"] /= (len(local_logits_q) + 1)

		return loss_pack


class ClassifierLoss(nn.Module):
	def __init__(self):
		super().__init__()

		self.loss_fn = nn.CrossEntropyLoss()

	def forward(self, input, target):
		loss_pack = {}

		loss_pack["cls"] = self.loss_fn(input, target)

		return loss_pack
