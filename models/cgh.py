#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from classy_vision.generic.distributed_util import is_distributed_training_run

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from models.base_model import BaseModel
from models.network import buildEncoder, find_layer, desc_dim
from models.criterion import calcFinalLoss, CGHLoss
from utils.dist import concat_all_gather, get_rank


def normalize(v):
	if type(v) == list:
		return [normalize(vv) for vv in v]

	return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))


class FeatDistiller(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim=128, feat_dim=2048, layers=None, learn_kappa=False):
		super().__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim

		self.reg_conv = nn.Sequential(nn.Conv2d(in_dim, feat_dim, kernel_size=1, padding=0, bias=False),
		                              nn.BatchNorm2d(feat_dim),
		                              nn.ReLU())
		self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))

		self.projector = nn.Sequential(nn.Linear(feat_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim))

		# self.scale = nn.Parameter(
		#     torch.FloatTensor(num_levels).fill_(1),
		#     requires_grad=learn_kappa)

		inter_weight = {"layer1": 1.0 / 16, "layer2": 1.0 / 8, "layer3": 1.0 / 4, "layer4": 1.0}
		data = []
		for layer in layers:
			data.append(inter_weight[layer])
		self.scale = nn.Parameter(torch.tensor(data), requires_grad=learn_kappa)

	def downsample(self, features, output_shape):
		selected_features = nn.functional.interpolate(features, output_shape, mode='bilinear', align_corners=False)
		# selected_features = nn.functional.interpolate(features, output_shape, mode='bilinear', align_corners=True)

		return selected_features

	def get_hypercols(self, hypercols, output_shape):
		assert output_shape[-1] <= 7
		kappa = torch.clamp(self.scale, max=1.0)
		kappa = torch.split(kappa, 1, dim=0)

		for index, feat in enumerate(hypercols):
			hypercols[index] = self.downsample(feat, output_shape)
			# hypercols[index] = kappa[index] * hypercols[index]
			# hypercols[index] = normalize(hypercols[index])

		return torch.cat(hypercols, dim=1)

	def forward(self, hypercols, output_shape):
		hypercol_cat = self.get_hypercols(hypercols, output_shape)

		B, C, H, W = hypercol_cat.shape
		assert self.in_dim == C
		out = self.reg_conv(hypercol_cat)
		out = self.pool(out)
		out = self.projector(out)

		return out


# def get_mlp(inp_dim, hidden_dim, out_dim):
# 	mlp = nn.Sequential(
# 		nn.Linear(inp_dim, hidden_dim),
# 		nn.BatchNorm1d(hidden_dim),
# 		nn.ReLU(inplace=True),
# 		nn.Linear(hidden_dim, out_dim),
# 	)
# 	return mlp

def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True, encoder_global=None):
	"""
	https://github.com/zhihou7/BatchFormer
	"""
	mlp = []
	if encoder_global:
		mlp.append(encoder_global)
	for l in range(num_layers):
		dim1 = input_dim if l == 0 else mlp_dim
		dim2 = output_dim if l == num_layers - 1 else mlp_dim

		mlp.append(nn.Linear(dim1, dim2, bias=False))

		if l < num_layers - 1:
			mlp.append(nn.BatchNorm1d(dim2))
			mlp.append(nn.ReLU(inplace=True))
		elif last_bn:
			# follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
			# for simplicity, we further removed gamma in BN
			mlp.append(nn.BatchNorm1d(dim2, affine=False))

	return nn.Sequential(*mlp)


class CGH(nn.Module):
	"""
	Build a MoCo model with: a query encoder, a key encoder, and a queue
	https://arxiv.org/abs/1911.05722
	"""

	def __init__(self, dataset, arch, dim=128, hid_dim=4096, hyper_dim=256, layers=None, K=65536, m=0.999, T=0.07,
	             mlp=True, multi=False, norm_center=False):
		"""
		dim: feature dimension (default: 128)
		K: queue size; number of negative keys (default: 65536)
		m: moco momentum of updating key encoder (default: 0.999)
		T: softmax temperature (default: 0.07)
		"""
		super().__init__()

		self.K = K
		self.m = m
		self.T = T
		self.multi = multi

		# bn_splits = 1 if is_distributed_training_run() or not isTrain else 8
		bn_splits = 1 if is_distributed_training_run() else 8

		# create the encoders
		# num_classes is the output fc dimension
		self.encoder_q, self.ftr_dim = buildEncoder(dataset, arch=arch, feature_dim=dim, bn_splits=bn_splits)
		self.encoder_k, _ = buildEncoder(dataset, arch=arch, feature_dim=dim, bn_splits=bn_splits)

		if mlp:  # hack: brute-force replacement
			dim_mlp = self.encoder_q.fc.weight.shape[1]
			self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, hid_dim), nn.ReLU(), nn.Linear(hid_dim, dim))
			self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, hid_dim), nn.ReLU(), nn.Linear(hid_dim, dim))

		self.extract_layers = layers
		self.feature_register_hook(self.encoder_q, self.extract_layers)
		self.feature_register_hook(self.encoder_k, self.extract_layers)

		hyper_dim_in = 0
		for layer in self.extract_layers:
			hyper_dim_in += desc_dim[arch][layer]
		hyper_dim_hid = desc_dim[arch][self.extract_layers[-1]]
		self.hypercols_proj_q = FeatDistiller(hyper_dim_in, hid_dim, hyper_dim, dim_mlp, layers, False)
		self.hypercols_proj_k = FeatDistiller(hyper_dim_in, hid_dim, hyper_dim, dim_mlp, layers, False)

		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False  # not update by gradient

		for param_q, param_k in zip(self.hypercols_proj_q.parameters(), self.hypercols_proj_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False  # not update by gradient

		# create the queue
		self.register_buffer("queue", torch.randn(dim, K))
		self.queue = nn.functional.normalize(self.queue, dim=0)
		self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
		self.register_buffer('queue_targets', -1 * torch.ones(K).long())
		self.register_buffer("queue_index", torch.arange(0, K))

		self.register_buffer("queue_hyper", torch.randn(hyper_dim, K))
		self.queue_hyper = nn.functional.normalize(self.queue_hyper, dim=0)
		self.register_buffer("queue_hyper_ptr", torch.zeros(1, dtype=torch.long))

		self.norm_center = norm_center
		self.center_momentum = 0.9
		self.register_buffer("center", torch.zeros(1, K))
		self.register_buffer("center_hyper", torch.zeros(1, K))

		if is_distributed_training_run():
			self._batch_shuffle = self._batch_shuffle_ddp
			self._batch_unshuffle = self._batch_unshuffle_ddp
		else:
			self._batch_shuffle = self._batch_shuffle_single_gpu
			self._batch_unshuffle = self._batch_unshuffle_single_gpu

	def feature_register_hook(self, net, extract_layers):
		self.extracted_feats = []

		def _feature_extract_hook(module, input, output):
			self.extracted_feats.append(output)

		for layer_id in extract_layers:
			layer = find_layer(net, layer_id)
			assert layer is not None, f"intermediate layer ({layer}) not found"
			handle = layer.register_forward_hook(_feature_extract_hook)

	def get_encoder_feat(self, im):
		feat = self.encoder_q(im)
		self.extracted_feats = []
		return feat

	def get_feats(self, net, im):
		self.extracted_feats = []
		feat = net(im)
		extracted_feats = self.extracted_feats
		assert len(self.extract_layers) == len(extracted_feats)
		for i, layer in enumerate(self.extract_layers):
			assert extracted_feats[i] is not None, f"intermediate layer {layer} never emitted an output"

		return feat, extracted_feats

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		"""
		Momentum update of the key encoder
		"""
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

		for param_q, param_k in zip(self.hypercols_proj_q.parameters(), self.hypercols_proj_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys, targets, index=None):
		if is_distributed_training_run():
			# gather keys before updating queue
			keys = concat_all_gather(keys)
			targets = concat_all_gather(targets)
			if index is not None:
				index = concat_all_gather(index)

		batch_size = keys.shape[0]

		ptr = int(self.queue_ptr)
		assert self.K % batch_size == 0  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.queue[:, ptr:ptr + batch_size] = keys.T
		self.queue_targets[ptr:ptr + batch_size] = targets
		if index is not None:
			self.queue_index[ptr: ptr + batch_size] = index
		ptr = (ptr + batch_size) % self.K  # move pointer

		self.queue_ptr[0] = ptr

	@torch.no_grad()
	def _dequeue_and_enqueue_hyper(self, keys):
		if is_distributed_training_run():
			# gather keys before updating queue
			keys = concat_all_gather(keys)

		batch_size = keys.shape[0]

		ptr = int(self.queue_hyper_ptr)
		assert self.K % batch_size == 0  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.queue_hyper[:, ptr:ptr + batch_size] = keys.T
		ptr = (ptr + batch_size) % self.K  # move pointer

		self.queue_hyper_ptr[0] = ptr

	@torch.no_grad()
	def _batch_shuffle_single_gpu(self, x):
		"""
		Batch shuffle, for making use of BatchNorm.
		"""
		# random shuffle index
		idx_shuffle = torch.randperm(x.shape[0]).cuda()

		# index for restoring
		idx_unshuffle = torch.argsort(idx_shuffle)

		return x[idx_shuffle], idx_unshuffle

	@torch.no_grad()
	def _batch_shuffle_ddp(self, x):
		"""
		Batch shuffle, for making use of BatchNorm.
		*** Only support DistributedDataParallel (DDP) model. ***
		"""
		# gather from all gpus
		batch_size_this = x.shape[0]
		x_gather = concat_all_gather(x)
		batch_size_all = x_gather.shape[0]

		num_gpus = batch_size_all // batch_size_this

		# random shuffle index
		idx_shuffle = torch.randperm(batch_size_all).cuda()

		# broadcast to all gpus
		torch.distributed.broadcast(idx_shuffle, src=0)

		# index for restoring
		idx_unshuffle = torch.argsort(idx_shuffle)

		# shuffled index for this gpu
		gpu_idx = torch.distributed.get_rank()
		idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

		return x_gather[idx_this], idx_unshuffle

	@torch.no_grad()
	def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
		"""
		Undo batch shuffle.
		"""
		return x[idx_unshuffle]

	@torch.no_grad()
	def _batch_unshuffle_ddp(self, x, idx_unshuffle):
		"""
		Undo batch shuffle.
		*** Only support DistributedDataParallel (DDP) model. ***
		"""
		# gather from all gpus
		batch_size_this = x.shape[0]
		x_gather = concat_all_gather(x)
		batch_size_all = x_gather.shape[0]

		num_gpus = batch_size_all // batch_size_this

		# restored index for this gpu
		gpu_idx = torch.distributed.get_rank()
		idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

		return x_gather[idx_this]

	def get_contrastive_logits(self, q, k, queue):
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
		l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
		logits = torch.cat([l_pos, l_neg], dim=1)
		logits /= self.T
		labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

		return logits, labels

	def get_distribution_logits(self, q, k, queue):
		logits_q = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
		logits_k = torch.einsum('nc,ck->nk', [k, queue.clone().detach()])

		return logits_q, logits_k

	def compute_local_logits(self, logits_k, local_views):
		local_logits_q = []
		local_logits_k = []
		for local in local_views:
			logits_q = torch.einsum('nc,ck->nk', [local, self.queue.clone().detach()])
			local_logits_q.append(logits_q)
			local_logits_k.append(logits_k)

		return local_logits_q, local_logits_k

	@torch.no_grad()
	def update_center(self, teacher_output, teacher_output_hyper):
		"""
		Update center used for teacher output.
		"""
		batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
		batch_center_hyper = torch.sum(teacher_output_hyper, dim=0, keepdim=True)

		if is_distributed_training_run():
			dist.all_reduce(batch_center)
			batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

			dist.all_reduce(batch_center_hyper)
			batch_center_hyper = batch_center_hyper / (len(teacher_output_hyper) * dist.get_world_size())

		# ema update
		self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
		self.center_hyper = self.center_hyper * self.center_momentum + batch_center_hyper * (1 - self.center_momentum)

	def forward(self, images, targets, index=None, warmup=False):
		"""
		Input:
			im_q: a batch of query images
			im_k: a batch of key images
		Output:
			logits, targets
		"""
		im_q, im_k = images[1], images[0]

		# compute query features
		hyper_shape_id = -1  # 0/-1
		q, q_inter = self.get_feats(self.encoder_q, im_q)
		q_hyper = self.hypercols_proj_q(q_inter, q_inter[hyper_shape_id].shape[-2:])

		q = nn.functional.normalize(q, dim=1)
		q_hyper = nn.functional.normalize(q_hyper, dim=1)

		# compute key features
		with torch.no_grad():  # no gradient to keys
			self._momentum_update_key_encoder()  # update the key encoder

			# shuffle for making use of BN
			im_k, idx_unshuffle = self._batch_shuffle(im_k)

			k, k_inter = self.get_feats(self.encoder_k, im_k)
			k_hyper = self.hypercols_proj_k(k_inter, k_inter[hyper_shape_id].shape[-2:])

			k = nn.functional.normalize(k, dim=1)
			k_hyper = nn.functional.normalize(k_hyper, dim=1)

			# undo shuffle
			k = self._batch_unshuffle(k, idx_unshuffle)
			k_hyper = self._batch_unshuffle(k_hyper, idx_unshuffle)

		logits, labels, logits_hyper, labels_hyper = None, None, None, None
		if warmup:
			# compute logits for contrastive learning
			logits, labels = self.get_contrastive_logits(q, k, self.queue)
			logits_hyper, labels_hyper = self.get_contrastive_logits(q_hyper, k_hyper, self.queue_hyper)
			# logits_hyper, labels_hyper = self.get_contrastive_logits(q_hyper, k, self.queue)

		# compute logits for distribution similarity
		logits_q, logits_k = self.get_distribution_logits(q, k, self.queue)
		logits_q_hyper, logits_k_hyper = self.get_distribution_logits(q_hyper, k_hyper, self.queue_hyper)

		logits_k_ori = logits_k
		logits_k_hyper_ori = logits_k_hyper
		if self.norm_center:
			logits_k = logits_k - self.center
			logits_k_hyper = logits_k_hyper - self.center_hyper

		local_logits_q, local_logits_k = None, None
		if self.multi:
			local_views = list()
			for n, im_local in enumerate(images[2:]):
				local_q, _ = self.get_feats(self.encoder_q, im_local)
				local_q = nn.functional.normalize(local_q, dim=1)
				local_views.append(local_q)

			local_logits_q, local_logits_k = self.compute_local_logits(logits_k_hyper, local_views)

		# dequeue and enqueue
		self._dequeue_and_enqueue(k, targets, index)
		self._dequeue_and_enqueue_hyper(k_hyper)

		self.update_center(logits_k_ori, logits_k_hyper_ori)

		return logits_q, logits_k, logits_q_hyper, logits_k_hyper, logits, labels, logits_hyper, labels_hyper,\
		       local_logits_q, local_logits_k


class CGHModel(BaseModel):
	def __init__(self, args, cfg, world_size):
		super().__init__(args, cfg, world_size)

		self.loss_names = ["cl", "cl_hyper", "csist"]
		self.model_names = ["net"]

		# define networks
		self.net = CGH(cfg.DATA["type"], cfg.MODEL["arch"], dim=cfg.MODEL["feat_dim"], hid_dim=cfg.MODEL["hid_dim"],
		                hyper_dim=cfg.MODEL["hyper_dim"], layers=cfg.MODEL["layers"], K=cfg.MODEL["queue_len"],
		                m=cfg.MODEL["momentum"], T=cfg.MODEL["temp"], mlp=cfg.MODEL["mlp"],
		                multi=cfg.DATA["transform"]["multi"], norm_center=cfg.MODEL["norm_center"])
		self.init_net(syncBN=False)

		if self.isTrain:
			# define loss functions
			self.criterion = CGHLoss(cfg.MODEL["temp_t"], cfg.MODEL["temp_s"]).to(self.device)
			# initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
			if self.cfg.DATA["type"] == "imagenet":
				param_dict = {}
				for k, v in self.net.named_parameters():
					param_dict[k] = v

				bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
				rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

				self.optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, },
				                                  {'params': rest_params, 'weight_decay': 1e-4}],
				                                 lr=cfg.OPTIM["lr"], momentum=0.9, weight_decay=1e-4)
			else:
				self.optimizer = torch.optim.SGD(self.net.parameters(), lr=cfg.OPTIM["lr"],
				                                 momentum=cfg.OPTIM["momentum"], weight_decay=cfg.OPTIM["weight_decay"])
			self.optimizers.append(self.optimizer)

	def set_input(self, input, target, index=None):
		self.images = [img.cuda(non_blocking=True) for img in input]
		self.target = target.cuda(non_blocking=True)
		self.index = index.cuda(non_blocking=True)

	def forward(self):
		warm_up_loss = self.cfg.LOSS["warm_up_loss"]
		warmup = self.epoch < warm_up_loss

		# Casts operations to mixed precision
		if self.args.half_precision:
			with torch.cuda.amp.autocast():
				output = self.net(self.images, self.target, self.index, warmup)
		else:
			output = self.net(self.images, self.target, self.index, warmup)

		return output

	def backward(self, output):
		if self.args.half_precision:
			with torch.cuda.amp.autocast():
				loss_pack = self.criterion(*output)
				loss = calcFinalLoss(loss_pack, self.loss_weight)
			# Scales the loss, and calls backward() to create scaled gradients
			self.scaler.scale(loss).backward()
		else:
			loss_pack = self.criterion(*output)
			loss = calcFinalLoss(loss_pack, self.loss_weight)
			loss.backward()

		return loss, loss_pack

	def optimize_parameters(self):
		output = self.forward()

		self.optimizer.zero_grad()
		loss, loss_pack = self.backward(output)

		if self.args.half_precision:
			# Unscales gradients and calls or skips optimizer.step()
			self.scaler.step(self.optimizer)
			# Updates the scale for next iteration
			self.scaler.update()
		else:
			self.optimizer.step()

		return output, loss, loss_pack

	def adjust_loss_weight(self, epoch, i, iteration_per_epoch):
		warm_up_loss = self.cfg.LOSS["warm_up_loss"]
		T = epoch * iteration_per_epoch + i
		warmup_iters = warm_up_loss * iteration_per_epoch

		if epoch < warm_up_loss:
			alpha = T / warmup_iters
			self.loss_weight["cl"] = 1.0 - alpha
			self.loss_weight["cl_hyper"] = alpha * 0.5
			self.loss_weight["csist"] = alpha * 0.5
		else:
			self.loss_weight["cl"] = 0
			self.loss_weight["cl_hyper"] = 0.5
			self.loss_weight['csist'] = 0.5
