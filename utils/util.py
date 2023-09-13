#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import shutil
import time
import datetime
import logging
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict, deque
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchvision.models as models

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from utils.dist import is_dist_avail_and_initialized


def get_logger(file_path):
	""" Make python logger """
	# [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
	logger = logging.getLogger('USNet')
	log_format = '%(asctime)s | %(message)s'
	formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
	file_handler = logging.FileHandler(file_path)
	file_handler.setFormatter(formatter)
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	logger.setLevel(logging.INFO)

	return logger


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            datetime.timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        # file_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    try:
        if dump_params:
            pickle.dump(params, open(os.path.join(params.exp_dir, "params.pkl"), "wb"))
    except:
        print("error when dumping params")

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.exp_dir, "checkpoints")
    try:
        if not params.rank and not os.path.isdir(params.dump_checkpoints):
            os.makedirs(params.dump_checkpoints)
    except:
        print("error when dumping checkpoints")

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.exp_dir, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.exp_dir, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.exp_dir)
    logger.info("")
    return logger, training_stats


def set_modelgrad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag


def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar', save_ckpt=False):
	os.makedirs(dir, exist_ok=True)
	torch.save(state, os.path.join(dir, "checkpoint.pth.tar"))
	if save_ckpt:
		file_path = os.path.join(dir, filename)
		torch.save(state, file_path)
	# if is_best:
	# 	best_path = os.path.join(dir, 'model_best.pth.tar')
	# 	shutil.copyfile(file_path, best_path)


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries), flush=True)

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def accuracy(output, target, topk=(1,), score=True):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		if score:
			_, pred = output.topk(maxk, 1, True, True)
		else:
			pred = output[:, :maxk]
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def normalize_batch(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
	# normalize using mean and std
	dtype = batch.dtype
	mean = torch.as_tensor(mean, dtype=dtype, device=batch.device)
	std = torch.as_tensor(std, dtype=dtype, device=batch.device)
	mean = mean.view(-1, 1, 1)
	std = std.view(-1, 1, 1)
	batch = (batch - mean) / std
	return batch


def denormalize_batch(batch, mean, std):
	"""denormalize for visualization"""
	dtype = batch.dtype
	mean = torch.as_tensor(mean, dtype=dtype, device=batch.device)
	std = torch.as_tensor(std, dtype=dtype, device=batch.device)
	mean = mean.view(-1, 1, 1)
	std = std.view(-1, 1, 1)
	batch = batch * std + mean
	return batch


def gram_matrix(y):
	if y.ndim == 4:
		(b, ch, h, w) = y.size()
	elif y.ndim == 2:
		(b, ch) = y.size()
		h = w = 1
	else:
		raise NotImplementedError("Wrong dim")

	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram


def worker_init_fn(id):
	# https://github.com/pytorch/pytorch/issues/5059
	# np.random.seed((id + torch.initial_seed()) % np.iinfo(np.int32).max)

	process_seed = torch.initial_seed()
	# Back out the base_seed so we can use all the bits.
	base_seed = process_seed - id
	ss = np.random.SeedSequence([id, base_seed])
	# More than 128 bits (4 32-bit words) would be overkill.
	np.random.seed(ss.generate_state(4))


def is_custom_kernel_supported():
	version_str = str(torch.version.cuda).split(".")
	major = version_str[0]
	minor = version_str[1]
	return int(major) >= 10 and int(minor) >= 1


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, encoder_key, model_name=None):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("{}.".format(encoder_key), ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        raise Exception


def load_netowrk(model, path, checkpoint_key="net"):
	if os.path.isfile(path):
		print("=> loading checkpoint '{}'".format(path))
		checkpoint = torch.load(path, map_location="cpu")

		# rename pre-trained keys
		state_dict = checkpoint[checkpoint_key]
		state_dict_new = {}
		for k in list(state_dict.keys()):
			if k.startswith('module'):
				# remove prefix
				state_dict_new[k[len("module."):]] = state_dict[k]
			else:
				state_dict_new[k] = state_dict[k]

		msg = model.load_state_dict(state_dict_new)
		assert set(msg.missing_keys) == set()

		print("=> loaded pre-trained model '{}'".format(path))
	else:
		print("=> no checkpoint found at '{}'".format(path))


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


class data_prefetcher():
	""" https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
	"""

	def __init__(self, gpu, loader):
		self.gpu = gpu
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		# self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
		# self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
		# With Amp, it isn't necessary to manually convert data to half.
		# if args.fp16:
		#     self.mean = self.mean.half()
		#     self.std = self.std.half()
		self.preload()

	def preload(self):
		try:
			self.next_input, self.next_target = next(self.loader)
		except StopIteration:
			self.next_input = None
			self.next_target = None
			return
		# if record_stream() doesn't work, another option is to make sure device inputs are created
		# on the main stream.
		# self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
		# self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
		# Need to make sure the memory allocated for next_* is not still in use by the main stream
		# at the time we start copying to next_*:
		# self.stream.wait_stream(torch.cuda.current_stream())
		with torch.cuda.stream(self.stream):
			self.next_input = self.next_input.cuda(self.gpu, non_blocking=True)
			self.next_target = self.next_target.cuda(self.gpu, non_blocking=True)
		# more code for the alternative if record_stream() doesn't work:
		# copy_ will record the use of the pinned source tensor in this side stream.
		# self.next_input_gpu.copy_(self.next_input, non_blocking=True)
		# self.next_target_gpu.copy_(self.next_target, non_blocking=True)
		# self.next_input = self.next_input_gpu
		# self.next_target = self.next_target_gpu

		# With Amp, it isn't necessary to manually convert data to half.
		# if args.fp16:
		#     self.next_input = self.next_input.half()
		# else:
		# self.next_input = self.next_input.float()
		# self.next_input = self.next_input.sub_(self.mean).div_(self.std)

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		input = self.next_input
		target = self.next_target
		if input is not None:
			input.record_stream(torch.cuda.current_stream())
		if target is not None:
			target.record_stream(torch.cuda.current_stream())
		self.preload()
		return input, target


class ActivationsAndGradients:
	""" Class for extracting activations and
	registering gradients from targetted intermediate layers """

	def __init__(self, model, target_layers, reshape_transform=None):
		self.model = model
		self.gradients = []
		self.activations = []
		self.reshape_transform = reshape_transform
		self.handles = []
		for target_layer in target_layers:
			self.handles.append(
				target_layer.register_forward_hook(self.save_activation))
			# Because of https://github.com/pytorch/pytorch/issues/61519,
			# we don't use backward hook to record gradients.
			self.handles.append(
				target_layer.register_forward_hook(self.save_gradient))

	def save_activation(self, module, input, output):
		activation = output

		if self.reshape_transform is not None:
			activation = self.reshape_transform(activation)
		self.activations.append(activation.detach())

	def save_gradient(self, module, input, output):
		if not hasattr(output, "requires_grad") or not output.requires_grad:
			# You can only register hooks on tensor requires grad.
			return

		# Gradients are computed in reverse order
		def _store_grad(grad):
			if self.reshape_transform is not None:
				grad = self.reshape_transform(grad)
			self.gradients = [grad.detach()] + self.gradients

		output.register_hook(_store_grad)

	def __call__(self, x):
		self.gradients = []
		self.activations = []
		return self.model(x)

	def release(self):
		for handle in self.handles:
			handle.remove()


def decode_imagenet(imagenet_root: str, wordnet_is_a_txt_path: str, words_txt_path: str, superclass_path: str="", superclass_names_path: str=""):
	dataset = torchvision.datasets.ImageFolder(imagenet_root)
	imagenet_wnids = dataset.classes

	with open(wordnet_is_a_txt_path, 'r') as f:
		wn_lines = f.readlines()
	with open(words_txt_path, 'r') as f:
		w_lines = f.readlines()
	child_to_parent_wnid = {}
	for wn_line in wn_lines:
		parent_wnid, child_wnid = wn_line.split()
		child_to_parent_wnid[child_wnid] = parent_wnid
	wnid_to_name: Dict[str, str] = {}
	for w_line in w_lines:
		wnid, name = w_line.split('\t')
		wnid_to_name[wnid] = name.rstrip('\n')

	imagenet_parent_wnid_set = set()
	for child_wnid in imagenet_wnids:
		parent_wnid = child_to_parent_wnid[child_wnid]
		imagenet_parent_wnid_set.add(parent_wnid)
	imagenet_parent_wnid_list = list(imagenet_parent_wnid_set)
	imagenet_parent_wnid_list.sort()
	print('the number of parent classes', len(imagenet_parent_wnid_list))

	parent_idx_list: List[str] = []
	parent_name_list: List[str] = []
	imagenet_child_parent_idx = []
	for child_wnid in imagenet_wnids:
		parent_wnid = child_to_parent_wnid[child_wnid]
		parent_name = wnid_to_name[parent_wnid]
		parent_idx = imagenet_parent_wnid_list.index(parent_wnid)
		parent_idx_str = str(parent_idx)
		parent_idx_list.append(parent_idx_str)
		parent_name_list.append(parent_name)
		imagenet_child_parent_idx.append(parent_idx)

	assert len(parent_idx_list) == 1000 and len(parent_name_list) == 1000
	# parent_idx_list_txt = '\n'.join(parent_idx_list)
	# parent_name_list_txt = '\n'.join(parent_name_list)
	# with open(superclass_path, 'w') as f:
	# 	f.write(parent_idx_list_txt)
	# with open(superclass_names_path, 'w') as f:
	# 	f.write(parent_name_list_txt)

	return wnid_to_name, imagenet_child_parent_idx


def global_pooling(x, type):
    assert x.dim() == 4
    if type == 'max':
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif type == 'avg':
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError(
            f"Unknown pooling type '{type}'. Supported types: ('avg', 'max').")


def calc_params(net, verbose=False):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	if verbose:
		print(net)
	print('Total number of parameters : %.3f M' % (num_params / 1e6))

	return num_params


if __name__ == '__main__':
	net_func = getattr(models, "resnet50")
	net = net_func(num_classes=1)
	calc_params(net)
