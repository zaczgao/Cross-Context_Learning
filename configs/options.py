#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import argparse

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


class BaseOptions():
	"""This class defines options used during both training and test time.

	It also implements several helper functions such as parsing, printing, and saving the options.
	It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and
	model class.
	"""

	def __init__(self):
		"""Reset the class; indicates the class hasn't been initailized"""
		self.initialized = False

	def initialize(self, parser):
		"""Define the common options that are used in both training and test."""
		parser.add_argument('--debug', action='store_true')
		parser.add_argument('--name', type=str, default='experiment_name',
		                    help='name of the experiment. It decides where to store samples and models')
		parser.add_argument('--resume', default='', type=str, metavar='PATH',
		                    help='path to latest checkpoint (default: none)')
		parser.add_argument("--exp_dir", type=str, default="./exp_dir",
		                    help="experiment dump path for checkpoints and log")
		parser.add_argument('--half_precision', action='store_true', help='use half precision')
		parser.add_argument('--use_wandb', action='store_true', help='use wandb')

		# dist
		parser.add_argument("--world_size", default=-1, type=int, help="""
		                    number of processes (gpus for all nodes): it is set automatically and
		                    should not be passed as argument""")
		parser.add_argument("--rank", default=-1, type=int, help="""global rank of this process:
		                    it is set automatically and should not be passed as argument""")
		parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
		parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
		parser.add_argument("--dist_url", default="env://", type=str,
		                    help="""url used to set up distributed training; """)
		parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
		parser.add_argument('--multiprocessing_distributed', action='store_true',
		                    help='Use multi-processing distributed training to launch '
		                         'N processes per node, which has N GPUs. This is the '
		                         'fastest way to use PyTorch for either single node or '
		                         'multi node data parallel training')

		# model parameters
		parser.add_argument('--arch', metavar='ARCH', default='resnet50',
		                    help='model architecture: ' + ' (default: resnet50)')
		parser.add_argument('--model', type=str, default='MoCo', help="chooses which model to use. [MoCo | ]")
		# dataset parameters
		parser.add_argument('--dataset', default='cifar10', help='dataset name')
		parser.add_argument('--preprocess', type=str, default='resize_and_crop',
		                    help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | '
		                         'scale_width_and_crop | none]')
		# additional parameters
		parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
		parser.add_argument('--prefix', default='', help="customized prefix")
		parser.add_argument('--suffix', default='', type=str, help='customized suffix')
		self.initialized = True
		return parser

	def gather_options(self):
		"""Initialize our parser with basic options(only once).
		Add additional model-specific and dataset-specific options.
		These options are defined in the <modify_commandline_options> function
		in model and dataset classes.
		"""
		if not self.initialized:  # check if it has been initialized
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		# get the basic options
		opt, _ = parser.parse_known_args()

		# save and return the parser
		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		"""Print and save options

		It will print both current options and default values(if different).
		It will save options into a text file / [checkpoints_dir] / opt.txt
		"""
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)

		# save to the disk
		# expr_dir = os.path.join(opt.exp_dir, opt.name)
		expr_dir = opt.exp_dir
		os.makedirs(expr_dir, exist_ok=True)
		file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

	def parse(self):
		"""Parse our options, create checkpoints directory suffix, and set up gpu device."""
		opt = self.gather_options()
		opt.isTrain = self.isTrain  # train or test

		# process opt.suffix
		if opt.suffix:
			suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
			opt.name = opt.name + suffix

		self.print_options(opt)

		self.opt = opt
		return self.opt


class PretrainOptions(BaseOptions):
	"""This class includes training options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('--phase', type=str, default='pretrain', choices=["pretrain", "lincls", "knn", "semi"], help='stage')
		parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
		parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

		self.isTrain = True
		return parser


class LinclsOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)  # define shared options
		parser.add_argument('--phase', type=str, default='lincls', choices=["pretrain", "lincls", "knn", "semi"], help='stage')
		parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
		parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

		self.isTrain = True
		return parser


class KNNOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)  # define shared options
		parser.add_argument('--phase', type=str, default='knn', choices=["pretrain", "lincls", "knn", "semi"], help='stage')
		parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
		parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

		parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
		parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
		parser.add_argument('--use_cuda', default=1, type=int,
		                    help="Store features in GPU.")
		parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
		                    help='Number of NN to use. 20 is usually working the best.')
		parser.add_argument('--temperature', default=0.07, type=float,
		                    help='Temperature used in the voting coefficient')
		parser.add_argument("--checkpoint_key", default="state_dict", type=str,
		                    help='Key to use in the checkpoint')
		parser.add_argument("--encoder_key", default="encoder", type=str,
		                    help='Key to use for the model encoder')
		parser.add_argument('--dump_features', default=None,
		                    help='Path where to save computed features, empty for no saving')
		parser.add_argument('--load_features', default=None, help="""If the features have
		    already been computed, where to find them.""")
		parser.add_argument('--data', type=str)

		self.isTrain = True
		return parser


class ClusterOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)  # define shared options
		parser.add_argument('--phase', type=str, default='cluster', choices=["pretrain", "lincls", "knn", "cluster", "semi"], help='stage')
		parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
		parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

		parser.add_argument("--backbone_prefix", type=str, default="backbone")
		parser.add_argument("--model_prefix", type=str, default="model")
		parser.add_argument("--num_classes", type=str, default="1000")
		parser.add_argument('--rel_pos_emb', default=False, action='store_true',
		                    help='Use relative position embedding in ViT.')
		parser.add_argument("--batch_size", "-bs", type=int, default=64)
		parser.add_argument('--data', type=str, default="./datasets/Imagenet1K/ILSVRC/Data/CLS-LOC")
		parser.add_argument('--use_super_class', default=False, action='store_true',
		                    help='use_super_class.')

		self.isTrain = True
		return parser


class SemiOptions(BaseOptions):
	"""This class includes test options.

	It also includes shared options defined in BaseOptions.
	"""

	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)  # define shared options
		parser.add_argument('--phase', type=str, default='semi', choices=["pretrain", "lincls", "knn", "semi"], help='stage')
		parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
		parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

		parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
		                    help="fine-tune on either 1% or 10% of labels")
		parser.add_argument("--data_path", type=str, default="",
		                    help="path to imagenet")
		parser.add_argument("--workers", default=10, type=int,
		                    help="number of data loading workers")
		parser.add_argument("--epochs", default=70, type=int,
		                    help="number of total epochs to run")
		parser.add_argument("--batch_size", default=256, type=int,
		                    help="batch size in total")
		parser.add_argument("--lr", default=0.005, type=float, help="initial learning rate - trunk")
		parser.add_argument("--lr_last_layer", default=0.02, type=float, help="initial learning rate - head")
		parser.add_argument("--decay_epochs", type=int, nargs="+", default=[30, 60],
		                    help="Epochs at which to decay learning rate.")
		parser.add_argument("--gamma", type=float, default=0.1, help="lr decay factor")


		self.isTrain = True
		return parser
