import os

import torch
from classy_vision.generic.distributed_util import is_distributed_training_run

from .loader import ImageFolderInstance, CIFAR10Instance, CIFAR100Instance, ImageFolderSubset
from .tinyimagenet import TinyImageNet
from .stl10 import STL10
from .transforms import get_trasnform
from utils.util import worker_init_fn


# https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py


def create_dataset(args, cfg, batch_size_gpu, workers_gpu):
	# Data loading code
	traindir = os.path.join(cfg.DATA["root"], 'train')
	valdir = os.path.join(cfg.DATA["root"], 'val')

	if cfg.DATA["type"].lower() == "cifar10":
		train_dataset = CIFAR10Instance(cfg.DATA["root"], train=True, transform=get_trasnform(True, cfg),
		                                 download=True)
		memory_dataset = CIFAR10Instance(cfg.DATA["root"], train=True, transform=get_trasnform(False, cfg),
		                                  download=True)
		val_dataset = CIFAR10Instance(cfg.DATA["root"], train=False, transform=get_trasnform(False, cfg),
		                               download=True)
		cluster_dataset = CIFAR10Instance(cfg.DATA["root"], train=True, transform=get_trasnform(False, cfg),
		                                  download=True)
	elif cfg.DATA["type"].lower() == "cifar100":
		train_dataset = CIFAR100Instance(cfg.DATA["root"], train=True, transform=get_trasnform(True, cfg),
		                                 download=True)
		memory_dataset = CIFAR100Instance(cfg.DATA["root"], train=True, transform=get_trasnform(False, cfg),
		                                  download=True)
		val_dataset = CIFAR100Instance(cfg.DATA["root"], train=False, transform=get_trasnform(False, cfg),
		                               download=True)
		cluster_dataset = CIFAR100Instance(cfg.DATA["root"], train=True, transform=get_trasnform(False, cfg),
		                                  download=True)
	elif cfg.DATA["type"].lower() == "tinyimagenet":
		train_dataset = TinyImageNet(cfg.DATA["root"], split="train", transform=get_trasnform(True, cfg), in_memory=False)
		memory_dataset = TinyImageNet(cfg.DATA["root"], split="train", transform=get_trasnform(False, cfg), in_memory=False)
		val_dataset = TinyImageNet(cfg.DATA["root"], split="val", transform=get_trasnform(False, cfg))
		cluster_dataset = TinyImageNet(cfg.DATA["root"], split="train", transform=get_trasnform(False, cfg), in_memory=False)
	elif cfg.DATA["type"].lower() == "stl10":
		if cfg.PHASE.lower() == "pretrain":
			train_split = "train+unlabeled"
		else:
			train_split = "train"
		train_dataset = STL10(root=cfg.DATA["root"], split=train_split, download=True, transform=get_trasnform(True, cfg))
		memory_dataset = STL10(root=cfg.DATA["root"], split=train_split, download=True, transform=get_trasnform(False, cfg))
		val_dataset = STL10(root=cfg.DATA["root"], split="test", download=True, transform=get_trasnform(False, cfg))
		cluster_dataset = STL10(root=cfg.DATA["root"], split=train_split, download=True, transform=get_trasnform(False, cfg))
		# if cfg.PHASE.lower() == "pretrain":
		# 	unlabeled_dataset = STL10(root=cfg.DATA["root"], split="unlabeled", download=True,
		# 	                          transform=get_trasnform(True, cfg))
		# 	train_dataset = torch.utils.data.ConcatDataset([train_dataset, unlabeled_dataset])
	elif cfg.DATA["type"].lower() == "imagenet100":
		train_dataset = ImageFolderSubset(cfg.DATA["class_path"], traindir, transform=get_trasnform(True, cfg))
		memory_dataset = ImageFolderSubset(cfg.DATA["class_path"], traindir, transform=get_trasnform(False, cfg))
		# val_dataset = ImageFolderSubset(cfg.DATA["class_path"], valdir, transform=get_trasnform(False, cfg))
		val_dataset = ImageFolderSubset(cfg.DATA["class_path"], cfg.DATA["root_val"], transform=get_trasnform(False, cfg))
		cluster_dataset = ImageFolderSubset(cfg.DATA["class_path"], traindir, transform=get_trasnform(False, cfg))
	elif cfg.DATA["type"].lower() == "imagenet":
		train_dataset = ImageFolderInstance(traindir, transform=get_trasnform(True, cfg))
		memory_dataset = ImageFolderInstance(traindir, transform=get_trasnform(False, cfg))
		val_dataset = ImageFolderInstance(cfg.DATA["root_val"], transform=get_trasnform(False, cfg))
		# val_dataset = ImageFolderInstance("/homes/zg002/Dataset/ImageNet/val/", transform=get_trasnform(False, cfg))
		cluster_dataset = ImageFolderInstance(traindir, transform=get_trasnform(False, cfg))

	if is_distributed_training_run():
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		memory_sampler = torch.utils.data.distributed.DistributedSampler(memory_dataset, shuffle=False)
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
		cluster_sampler = torch.utils.data.distributed.DistributedSampler(cluster_dataset, shuffle=False)
	else:
		train_sampler = None
		memory_sampler = None
		val_sampler = None
		cluster_sampler = None

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_gpu,
	                                           shuffle=(train_sampler is None), num_workers=workers_gpu,
	                                           pin_memory=True, sampler=train_sampler, drop_last=True)
	memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=batch_size_gpu, shuffle=False,
	                                            sampler=memory_sampler, num_workers=workers_gpu, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_gpu, shuffle=False,
	                                         num_workers=workers_gpu, pin_memory=True, sampler=val_sampler)

	cluster_loader = torch.utils.data.DataLoader(
		cluster_dataset, batch_size=batch_size_gpu * 5, shuffle=False,
		sampler=cluster_sampler, num_workers=workers_gpu, pin_memory=True)

	return train_loader, memory_loader, val_loader, cluster_loader, train_sampler