#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import math
import random
import time
import warnings
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from classy_vision.generic.distributed_util import is_distributed_training_run

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from configs.options import LinclsOptions
from configs import createConfig
from samples import create_dataset
from models import create_model
from models.criterion import calcFinalLoss
from utils.util import AverageMeter, ProgressMeter, accuracy
from utils.dist import init_distributed_mode, all_reduce_mean, reduce_loss_dict, get_world_size, get_rank
# from utils.visualizer import dump_image


def train(args, cfg, world_size):
	model = create_model(args, cfg, world_size)
	model.setup()

	train_loader, memory_loader, val_loader, cluster_loader, train_sampler = create_dataset(args, cfg, model.batch_size_gpu,
	                                                                       model.workers_gpu)

	for epoch in range(model.start_epoch, cfg.OPTIM["num_epoch"]):
		if is_distributed_training_run():
			train_sampler.set_epoch(epoch)

		# train for one epoch
		metrics = trainEpoch(args, cfg, train_loader, cluster_loader, model, epoch)

		# evaluate on validation set
		acc1 = validate(val_loader, model.net, model.criterion, model.loss_weight, epoch)

		# remember best acc@1 and save checkpoint
		is_best = acc1 > model.best_acc1
		model.best_acc1 = max(acc1, model.best_acc1)

		if get_rank() == 0:
			save_ckpt = (epoch + 1) % cfg.EVAL["save_freq"] == 0
			model.save_networks(epoch, is_best, save_ckpt)

			if epoch == model.start_epoch:
				sanity_check(model.net.state_dict(), args.pretrained)

			metrics["Acc1/val"].append(acc1)
			model.logMetricsEpoch(epoch, metrics)

	print("best_acc1:", model.best_acc1)


def trainEpoch(args, cfg, train_loader, cluster_loader, model, epoch):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	lossterm = {}
	for name in model.loss_names:
		lossterm[name] = AverageMeter('Loss '+ name, ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(train_loader),
	                         [batch_time, data_time, losses, top1, top5] + list(lossterm.values()),
	                         prefix="Train Epoch: [{}]".format(epoch))
	metrics = defaultdict(list)

	model.set_epoch(epoch)
	model.start_of_training_epoch()

	"""
	Switch to eval mode:
	Under the protocol of linear classification on frozen features/models,
	it is not legitimate to change any part of the pre-trained model.
	BatchNorm in train mode may revise running mean/std (even if it receives
	no gradient), which are part of the model parameters too.
	"""
	model.eval()

	end = time.time()
	for i, (data, target, index) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(model.gpu, non_blocking=True)

		model.set_input(data, target, index)
		model.update_learning_rate(epoch, i, len(train_loader))
		model.adjust_loss_weight(epoch, i, len(train_loader))
		output, loss, loss_pack = model.optimize_parameters()

		# measure accuracy and record loss
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		batchsize = data.size(0)

		if is_distributed_training_run():
			# torch.distributed.barrier()
			loss_pack = reduce_loss_dict(loss_pack)
			loss = all_reduce_mean(loss)
			acc1 = all_reduce_mean(acc1)
			acc5 = all_reduce_mean(acc5)

		# measure elapsed time
		torch.cuda.synchronize()
		batch_time.update(time.time() - end)
		end = time.time()

		if get_rank() == 0:
			losses.update(loss.item(), batchsize)
			top1.update(acc1[0].item(), batchsize)
			top5.update(acc5[0].item(), batchsize)
			for name in model.loss_names:
				if name in loss_pack:
					lossterm[name].update(loss_pack[name].item(), batchsize)
			if i % cfg.EVAL["print_steps"] == 0:
				progress.display(i)

			model.writer.add_scalar("Loss/train_step", loss.item(), model.global_step)
			model.writer.add_scalar("Acc1/train_step", acc1[0].item(), model.global_step)
			metrics["Loss/train"].append(loss.item())
			metrics["Acc1/train"].append(acc1[0].item())
			for key, value in loss_pack.items():
				model.writer.add_scalar("Loss {}/train_step".format(key), value.item(), model.global_step)
				metrics["Loss {}/train".format(key)].append(value.item())

			# for key, img in data.items():
			# 	if "img" in key:
			# 		dir = os.path.join(model.exp_dir, "img_" + cfg.PHASE, str(i % 10))
			# 		img = dump_image(img, cfg.IMAGE_MEAN, cfg.IMAGE_STD, dir, key)
			# 		model.writer.add_image(key, img[0], global_step=model.global_step, dataformats='HWC')

			# if isinstance(output, dict):
			# 	for key, value in output.items():
			# 		if "img" in key:
			# 			# dir = os.path.join(model.exp_dir, "img_" + cfg.PHASE, str(i % 10))
			# 			img = dump_image(value.detach(), cfg.IMAGE_MEAN, cfg.IMAGE_STD)
			# 			model.writer.add_image(key, img[0], global_step=model.global_step, dataformats='HWC')
		model.global_step += 1

	return metrics


def main(**kargs):
	args = LinclsOptions().parse()
	init_distributed_mode(args)

	if not is_distributed_training_run():
		warnings.warn('You have chosen a specific GPU. This will completely '
		              'disable data parallelism.')

	cudnn.benchmark = True
	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		cudnn.deterministic = True
		cudnn.benchmark = False
		warnings.warn('You have chosen to seed training. '
		              'This will turn on the CUDNN deterministic setting, '
		              'which can slow down your training considerably! '
		              'You may see unexpected behavior when restarting '
		              'from checkpoints.')

	config = createConfig(args)
	assert config.PHASE == "lincls"

	world_size = get_world_size()
	train(args, config, world_size)


def sanity_check(state_dict, pretrained_weights):
	"""
	Linear classifier should not change any weights other than the linear layer.
	This sanity check asserts nothing wrong happens (e.g., BN stats updated).
	"""
	print("=> loading '{}' for sanity check".format(pretrained_weights))
	checkpoint = torch.load(pretrained_weights, map_location="cpu")
	state_dict_pre = checkpoint['net']

	for k in list(state_dict.keys()):
		# only ignore fc layer
		if 'fc.weight' in k or 'fc.bias' in k:
			continue

		# name in pretrained model
		k_pre = 'module.encoder_q.' + k[len('module.'):] if k.startswith('module.') else 'module.encoder_q.' + k
		if k_pre not in state_dict_pre:
			k_pre = k_pre[len('module.'):]

		assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
			'{} is changed in linear classifier training.'.format(k)

	print("=> sanity check passed.")


def validate(val_loader, model, criterion, loss_weights_dict, epoch):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test Epoch: [{}]'.format(
		epoch))

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (images, target, _) in enumerate(val_loader):
			images = images.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

			# compute output
			output = model(images)
			loss_pack = criterion(output, target)
			loss = calcFinalLoss(loss_pack, loss_weights_dict)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))

			if is_distributed_training_run():
				# torch.distributed.barrier()
				loss = all_reduce_mean(loss)
				acc1 = all_reduce_mean(acc1)
				acc5 = all_reduce_mean(acc5)

			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0].item(), images.size(0))
			top5.update(acc5[0].item(), images.size(0))

			# measure elapsed time
			torch.cuda.synchronize()
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				progress.display(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

	return top1.avg


if __name__ == '__main__':
	main()
