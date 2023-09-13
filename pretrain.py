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
import copy
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

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from configs.options import PretrainOptions
from configs import createConfig
from samples import create_dataset
from models import create_model
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
		acc1 = 0.
		if cfg.EVAL["enabled"] and ((epoch + 1) % cfg.EVAL["val_epoch"] == 0 or epoch == 0):
			dim_mlp = cfg.MODEL["feat_dim"]
			# enc = copy.deepcopy(enc)
			# enc.fc = nn.Identity()
			acc1 = validateKNN(memory_loader, val_loader, model.net, epoch, dim_mlp, cfg.EVAL["knn_k"], cfg.EVAL["knn_t"])

		# remember best acc@1 and save checkpoint
		is_best = acc1 > model.best_acc1
		model.best_acc1 = max(acc1, model.best_acc1)

		if get_rank() == 0:
			save_ckpt = (epoch + 1) % cfg.EVAL["save_freq"] == 0
			model.save_networks(epoch, is_best, save_ckpt)

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
	model.start_of_training_epoch(loader=cluster_loader)

	# if cfg.MODEL["type"].lower() == "rlca":
	# 	model.get_cluster(epoch, cluster_loader)

	# switch to train mode
	model.train()

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
		# acc1, acc5 = accuracy(output, target, topk=(1, 5))
		acc1 = torch.tensor([0.0]).cuda()
		acc5 = torch.tensor([0.0]).cuda()
		batchsize = data[0].size(0)

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

	model.end_of_training_epoch()

	return metrics


def main(**kargs):
	args = PretrainOptions().parse()
	init_distributed_mode(args)

	if not is_distributed_training_run():
		warnings.warn('You have chosen a specific GPU. This will completely '
		              'disable data parallelism.')

	cudnn.benchmark = True
	if args.seed is not None:
		random.seed(args.seed)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.cuda.manual_seed_all(args.seed)
		cudnn.deterministic = True
		cudnn.benchmark = False
		warnings.warn('You have chosen to seed training. '
		              'This will turn on the CUDNN deterministic setting, '
		              'which can slow down your training considerably! '
		              'You may see unexpected behavior when restarting '
		              'from checkpoints.')
		args.half_precision = False

	config = createConfig(args)
	assert config.PHASE == "pretrain"

	world_size = get_world_size()
	train(args, config, world_size)


# test using a knn monitor
def validateKNN(memory_loader, val_loader, net, epoch, dim_mlp, knn_k, knn_t):
	batch_time = AverageMeter('Time', ':6.3f')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test Epoch: [{}]".format(epoch))

	feature_bank = []
	classes = len(memory_loader.dataset.classes)

	# switch to evaluate mode
	net.eval()

	with torch.no_grad():
		end = time.time()

		# generate feature bank
		# for images, target, _ in memory_loader:
		# 	images = images.cuda(non_blocking=True)
		# 	feature = model(images)
		# 	feature = F.normalize(feature, dim=1)
		# 	feature_bank.append(feature)
		# # [D, N]
		# feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
		feature_bank = torch.zeros(len(memory_loader.dataset), dim_mlp).cuda()
		with torch.cuda.amp.autocast(enabled=True):
			for i, (images, _, index) in enumerate(memory_loader):
				images = images.cuda(non_blocking=True)
				feat = net.module.get_encoder_feat(images)
				feat = F.normalize(feat, dim=1)
				feature_bank[index] = feat.float()
		if is_distributed_training_run():
			print("dist training...")
			dist.barrier()
			dist.all_reduce(feature_bank, op=dist.ReduceOp.SUM)
		feature_bank = feature_bank.t()

		# [N]
		feature_labels = torch.tensor(memory_loader.dataset.targets, device=feature_bank.device)
		# loop test data to predict the label by weighted knn search

		for i, (images, target, _) in enumerate(val_loader):
			images = images.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

			# compute output
			feature = net.module.get_encoder_feat(images)
			feature = F.normalize(feature, dim=1)

			prediction = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

			# measure accuracy and record loss
			total_num = images.size(0)
			total_top1 = (prediction[:, 0] == target).float().sum()
			acc1 = total_top1 / total_num * 100

			if is_distributed_training_run():
				# torch.distributed.barrier()
				acc1 = all_reduce_mean(acc1)

			top1.update(acc1.item(), images.size(0))

			# measure elapsed time
			torch.cuda.synchronize()
			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				progress.display(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

	return top1.avg


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
	# compute cos similarity between each feature vector and feature bank ---> [B, N]
	sim_matrix = torch.mm(feature, feature_bank)
	# [B, K]
	sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
	# [B, K]
	sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
	sim_weight = (sim_weight / knn_t).exp()

	# counts for each class
	one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
	# [B*K, C]
	one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
	# weighted score ---> [B, C]
	pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

	pred_labels = pred_scores.argsort(dim=-1, descending=True)
	return pred_labels


if __name__ == '__main__':
	main()
