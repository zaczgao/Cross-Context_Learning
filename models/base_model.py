#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import math
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter
from classy_vision.generic.distributed_util import is_distributed_training_run

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from utils.util import save_checkpoint
from utils.dist import get_rank


class BaseModel(ABC):
	def __init__(self, args, cfg, world_size):
		self.args = args
		self.cfg = cfg
		self.gpu = args.gpu
		self.world_size = world_size
		self.isTrain = args.isTrain
		self.device = torch.device('cuda:{}'.format(args.gpu)) if args.gpu is not None else torch.device(
			'cpu')  # get device name: CPU or GPU

		self.batch_size_gpu = cfg.DATA["batch_size"]
		self.workers_gpu = cfg.DATA["workers"]
		self.start_epoch = 0  # manual epoch number (useful on restarts)
		self.global_step = 0
		self.best_acc1 = 0

		# self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
		self.time_str = ""
		self.exp_dir = args.exp_dir
		self.loss_names = []
		self.model_names = []
		self.visual_names = []
		self.optimizers = []
		self.image_paths = []
		self.metric = 0  # used for learning rate policy 'plateau'
		self.loss_weight = cfg.LOSS["weight"]
		self.writer = self.initTensorboardWriters()
		self.buffer_dict = dict()

	@abstractmethod
	def set_input(self, input, target, index=None):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): includes the data itself and its metadata information.
		"""
		pass

	@abstractmethod
	def forward(self):
		"""Run forward pass; called by both functions <optimize_parameters> and <test>."""
		pass

	@abstractmethod
	def optimize_parameters(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
		pass

	def init_net(self, syncBN=True):
		# load from pre-trained, before DistributedDataParallel constructor
		if self.args.pretrained:
			if os.path.isfile(self.args.pretrained):
				print("=> loading checkpoint '{}'".format(self.args.pretrained))
				checkpoint = torch.load(self.args.pretrained, map_location="cpu")

				# rename pre-trained keys
				state_dict = checkpoint['net']
				state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
				state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
				# for k in list(state_dict.keys()):
				# 	# retain only encoder up to before the embedding layer
				# 	if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
				# 		# remove prefix
				# 		state_dict[k[len("module.encoder_q."):]] = state_dict[k]
				# 	elif k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
				# 		state_dict[k[len("encoder_q."):]] = state_dict[k]
				# 	# delete renamed or unused k
				# 	del state_dict[k]

				self.start_epoch = 0
				msg = self.net.load_state_dict(state_dict, strict=False)
				assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

				print("=> loaded pre-trained model '{}' with msg: {}".format(self.args.pretrained, msg))
			else:
				print("=> no checkpoint found at '{}'".format(self.args.pretrained))

		if not torch.cuda.is_available():
			print('using CPU, this will be slow')
		elif is_distributed_training_run():
			# For multiprocessing distributed, DistributedDataParallel constructor
			# should always set the single device scope, otherwise,
			# DistributedDataParallel will use all available devices.
			if self.args.gpu is not None:
				torch.cuda.set_device(self.args.gpu)
				# When using a single GPU per process and per
				# DistributedDataParallel, we need to divide the batch size
				# ourselves based on the total number of GPUs we have
				self.batch_size_gpu = int(self.batch_size_gpu / self.world_size)
				self.workers_gpu = int((self.workers_gpu + self.world_size - 1) / self.world_size)
				for name in self.model_names:
					net = getattr(self, name)
					if syncBN:
						net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
					net.cuda(self.args.gpu)
					net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.args.gpu], find_unused_parameters=True)
					setattr(self, name, net)
			else:
				for name in self.model_names:
					net = getattr(self, name)
					if syncBN:
						net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
					net.cuda()
					# DistributedDataParallel will divide and allocate batch_size to all
					# available GPUs if device_ids are not set
					net = torch.nn.parallel.DistributedDataParallel(net)
					setattr(self, name, net)
		elif self.args.gpu is not None:
			torch.cuda.set_device(self.args.gpu)
			for name in self.model_names:
				net = getattr(self, name)
				net = net.cuda(self.args.gpu)
				setattr(self, name, net)
		else:
			# AllGather implementation (batch shuffle, queue update, etc.) in
			# this code only supports DistributedDataParallel.
			raise NotImplementedError("Only DistributedDataParallel is supported.")

	def setup(self):
		"""Load and print networks; create schedulers

		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		# if self.isTrain:
		# 	self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
		self.load_networks()
		self.print_networks(True)

		if self.args.half_precision:
			# Creates once at the beginning of training
			self.scaler = torch.cuda.amp.GradScaler()

	def set_epoch(self, epoch):
		self.epoch = epoch

	def start_of_training_epoch(self, **kargs):
		pass

	def end_of_training_epoch(self, **kargs):
		pass
	
	def adjust_loss_weight(self, epoch, i, iteration_per_epoch):
		pass
	
	def train(self):
		"""Make models train mode during train time"""
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, name)
				net.train()

	def eval(self):
		"""Make models eval mode during test time"""
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, name)
				net.eval()

	def test(self):
		"""Forward function used in test time.

		This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
		It also calls <compute_visuals> to produce additional visualization results
		"""
		with torch.no_grad():
			self.forward()
			self.compute_visuals()

	def compute_visuals(self):
		"""Calculate additional output images for visdom and HTML visualization"""
		pass

	def get_image_paths(self):
		""" Return image paths that are used to load current data"""
		return self.image_paths

	def adjust_learning_rate(self, optimizer, epoch, base_lr, iter, num_epoch, iteration_per_epoch, warm_up):
		T = epoch * iteration_per_epoch + iter
		warmup_iters = warm_up * iteration_per_epoch
		total_iters = (num_epoch - warm_up) * iteration_per_epoch
		total_epoch = num_epoch - warm_up

		if epoch < warm_up:
			lr = base_lr * 1.0 * T / warmup_iters
		else:
			if self.cfg.OPTIM["schedule_type"] == "cos":
				T = T - warmup_iters
				lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
			elif self.cfg.OPTIM["schedule_type"] == "cos_epoch":
				lr = 0.5 * base_lr * (1. + math.cos(math.pi * (epoch - warm_up) / total_epoch))
			elif self.cfg.OPTIM["schedule_type"] == "step_lr":
				lr = base_lr
				for milestone in self.cfg.OPTIM["schedule"]:
					lr *= 0.1 if epoch >= milestone else 1.

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def update_learning_rate(self, epoch, iter, iteration_per_epoch):
		"""Update learning rates for all the networks; called at the end of every epoch"""
		old_lr = self.optimizers[0].param_groups[0]['lr']
		for optimizer in self.optimizers:
			self.adjust_learning_rate(optimizer, epoch, self.cfg.OPTIM["lr"], iter, self.cfg.OPTIM["num_epoch"],
			                     iteration_per_epoch, self.cfg.OPTIM["warm_up"])

		if iter == 0:
			lr = self.optimizers[0].param_groups[0]['lr']
			print('learning rate %.7f -> %.7f' % (old_lr, lr))

	def get_current_visuals(self):
		"""Return visualization images. train.py will display these images with visdom, and save the images to a
		HTML"""
		visual_ret = OrderedDict()
		for name in self.visual_names:
			if isinstance(name, str):
				visual_ret[name] = getattr(self, name)
		return visual_ret

	def get_current_losses(self):
		"""Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
		errors_ret = OrderedDict()
		for name in self.loss_names:
			if isinstance(name, str):
				errors_ret[name] = float(
					getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
		return errors_ret

	def save_networks(self, epoch, is_best, save_ckpt=False):
		"""Save all the networks to the disk.

		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		state = {}
		state["epoch"] = epoch + 1
		state["arch"] = self.cfg.MODEL["arch"]
		state["best_acc1"] = self.best_acc1

		for name in self.model_names:
			net = getattr(self, name)
			state[name] = net.state_dict()

		for i, optimizer in enumerate(self.optimizers):
			state["optimizer" + str(i)] = optimizer.state_dict()

		# for i, scheduler in enumerate(self.schedulers):
		# 	state["scheduler" + str(i)] = scheduler.state_dict()

		save_checkpoint(state, is_best,
		                dir=os.path.join(self.exp_dir, "checkpoint_" + self.cfg.PHASE),
		                filename='{:s}_{:04d}.pth.tar'.format(self.cfg.NAME, epoch),
		                save_ckpt=save_ckpt)

	def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
		"""Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
		key = keys[i]
		if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
			if module.__class__.__name__.startswith('InstanceNorm') and \
					(key == 'running_mean' or key == 'running_var'):
				if getattr(module, key) is None:
					state_dict.pop('.'.join(keys))
			if module.__class__.__name__.startswith('InstanceNorm') and \
					(key == 'num_batches_tracked'):
				state_dict.pop('.'.join(keys))
		else:
			self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

	def load_networks(self):
		"""Load all the networks from the disk.

		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		# optionally resume from a checkpoint
		if self.args.resume:
			if os.path.isfile(self.args.resume):
				print("=> loading checkpoint '{}'".format(self.args.resume))
				if self.args.gpu is None:
					checkpoint = torch.load(self.args.resume)
				else:
					# Map model to be loaded to specified single gpu.
					loc = 'cuda:{}'.format(self.args.gpu)
					checkpoint = torch.load(self.args.resume, map_location=loc)
				self.start_epoch = checkpoint['epoch']
				self.best_acc1 = checkpoint['best_acc1']
				# if self.gpu is not None:
				# 	# best_acc1 may be from a checkpoint from a different GPU
				# 	self.best_acc1 = self.best_acc1.to(self.gpu)
				# load architecture params from checkpoint.

				if checkpoint['arch'] != self.cfg.MODEL['arch']:
					msg = ("Warning: Architecture configuration given in config file is"
					       " different from that of checkpoint."
					       " This may yield an exception while state_dict is being loaded.")
					print(msg)

				for name in self.model_names:
					net = getattr(self, name)
					net.load_state_dict(checkpoint[name])

				for i, optimizer in enumerate(self.optimizers):
					optimizer.load_state_dict(checkpoint["optimizer" + str(i)])

				# if self.isTrain:
				# 	for i, scheduler in enumerate(self.schedulers):
				# 		scheduler.load_state_dict(checkpoint["scheduler" + str(i)])

				print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))
			else:
				print("=> no checkpoint found at '{}'".format(self.args.resume))

	def print_networks(self, verbose):
		"""Print the total number of parameters in the network and (if verbose) network architecture

		Parameters:
			verbose (bool) -- if verbose: print the network architecture
		"""
		print('---------- Networks initialized -------------')
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, name)
				num_params = 0
				for param in net.parameters():
					num_params += param.numel()
				if verbose:
					print(net)
				print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
		print('-----------------------------------------------')

	def set_requires_grad(self, nets, requires_grad=False):
		"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
			nets (network list)   -- a list of networks
			requires_grad (bool)  -- whether the networks require gradients or not
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def __str__(self):
		"""
		Model prints with number of trainable parameters
		"""
		info = ""
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, name)

				model_parameters = filter(lambda p: p.requires_grad, net.parameters())
				num_params = sum([np.prod(p.size()) for p in model_parameters])

				info += net.__str__() + '\n[Network {}] Total number of trainable parameters : {}\n'.format(name,
				                                                                                            num_params)

		return info

	def initTensorboardWriters(self):
		writer = None
		if get_rank() == 0:
			log_dir = os.path.join(self.exp_dir, "log_" + self.cfg.PHASE)
			writer = SummaryWriter(log_dir=log_dir)
		return writer

	def logMetricsEpoch(self, epoch, metrics):
		for k, v in metrics.items():
			self.writer.add_scalar(k, np.array(v).mean(), epoch)


if __name__ == '__main__':
	pass
