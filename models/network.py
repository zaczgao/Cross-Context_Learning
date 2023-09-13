#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
import functools
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions.beta import Beta

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from utils import util

desc_dim = {
	"resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
	"resnet50": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
	"resnet101": {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
}


def find_layer(net, layer):
	if type(layer) == str:
		modules = dict([*net.named_modules()])
		return modules.get(layer, None)
	elif type(layer) == int:
		children = [*net.children()]
		return children[layer]
	return None


def get_norm_layer(norm_type='instance'):
	"""Return a normalization layer

	Parameters:
		norm_type (str) -- the name of the normalization layer: batch | instance | none

	For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
	For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
	"""
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
	elif norm_type == 'none':
		def norm_layer(x):
			return nn.Identity()
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
	def __init__(self, num_features, num_splits, **kw):
		super().__init__(num_features, **kw)
		self.num_splits = num_splits

	def forward(self, input):
		N, C, H, W = input.shape
		if self.training or not self.track_running_stats:
			running_mean_split = self.running_mean.repeat(self.num_splits)
			running_var_split = self.running_var.repeat(self.num_splits)
			outcome = nn.functional.batch_norm(
				input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
				self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
				True, self.momentum, self.eps).view(N, C, H, W)
			self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
			self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
			return outcome
		else:
			return nn.functional.batch_norm(
				input, self.running_mean, self.running_var,
				self.weight, self.bias, False, self.momentum, self.eps)


class ResNetCIFAR(nn.Module):
	"""
	Common CIFAR ResNet recipe.
	Comparing with ImageNet ResNet recipe, it:
	(i) replaces conv1 with kernel=3, str=1
	(ii) removes pool1
	"""

	def __init__(self, arch=None, feature_dim=128, bn_splits=16):
		super().__init__()

		if isinstance(arch, str):
			# use split batchnorm
			norm_layer = functools.partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
			resnet_arch = getattr(models, arch)
			net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)
		else:
			net = arch

		for name, module in net.named_children():
			if name == 'conv1':
				module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if isinstance(module, nn.MaxPool2d):
				continue
			if isinstance(module, nn.Linear):
				self.add_module("flatten", nn.Flatten(1))
			self.add_module(name, module)

	def forward(self, x):
		for name, block in self._modules.items():
			x = block(x)
		return x


def buildEncoder(dataset, arch, feature_dim=128, bn_splits=8):
	# use split batchnorm
	norm_layer = functools.partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
	resnet_arch = getattr(models, arch)
	net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

	if dataset.lower() == "cifar10" or dataset.lower() == "cifar100":
		net = ResNetCIFAR(arch=net, feature_dim=feature_dim, bn_splits=bn_splits)

	ftr_dim = net.fc.weight.shape[1]

	return net, ftr_dim


class UnetGenerator(nn.Module):
	"""Create a Unet-based generator"""

	def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
		"""Construct a Unet generator
		Parameters:
			input_nc (int)  -- the number of channels in input images
			output_nc (int) -- the number of channels in output images
			num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
								image of size 128x128 will become of size 1x1 # at the bottleneck
			ngf (int)       -- the number of filters in the last conv layer
			norm_layer      -- normalization layer

		We construct the U-Net from the innermost layer to the outermost layer.
		It is a recursive process.
		"""
		super(UnetGenerator, self).__init__()
		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
		                                     innermost=True)  # add the innermost layer
		for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
			                                     norm_layer=norm_layer, use_dropout=use_dropout)
		# gradually reduce the number of filters from ngf * 8 to ngf
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
		                                     norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
		                                     norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
		                                     norm_layer=norm_layer)  # add the outermost layer

	def forward(self, input):
		"""Standard forward"""
		return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
	"""Defines the Unet submodule with skip connection.
		X -------------------identity----------------------
		|-- downsampling -- |submodule| -- upsampling --|
	"""

	def __init__(self, outer_nc, inner_nc, input_nc=None,
	             submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		"""Construct a Unet submodule with skip connections.

		Parameters:
			outer_nc (int) -- the number of filters in the outer conv layer
			inner_nc (int) -- the number of filters in the inner conv layer
			input_nc (int) -- the number of channels in input images/features
			submodule (UnetSkipConnectionBlock) -- previously defined submodules
			outermost (bool)    -- if this module is the outermost module
			innermost (bool)    -- if this module is the innermost module
			norm_layer          -- normalization layer
			use_dropout (bool)  -- if use dropout layers.
		"""
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
		                     stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:  # add skip connections
			return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
	"""Defines a PatchGAN discriminator"""

	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
		"""Construct a PatchGAN discriminator

		Parameters:
			input_nc (int)  -- the number of channels in input images
			ndf (int)       -- the number of filters in the last conv layer
			n_layers (int)  -- the number of conv layers in the discriminator
			norm_layer      -- normalization layer
		"""
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		kw = 4
		padw = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # gradually increase the number of filters
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [
			nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		"""Standard forward."""
		return self.model(input)


def conv_ReLU(in_channels, out_channels, kernel_size, stride=1, padding=0,
              use_norm=True, norm=nn.InstanceNorm2d):
	"""Returns a 2D Conv followed by a ReLU
	"""
	if use_norm:
		return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
		                               stride, padding),
		                     norm(out_channels),
		                     nn.ReLU(inplace=True))
	else:
		return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
		                               stride, padding),
		                     nn.ReLU(inplace=True))


class MyUpsample(nn.Module):
	def __init__(self, scale_factor, mode='nearest'):
		super().__init__()
		self.upsample = nn.functional.interpolate
		self.scale_factor = scale_factor
		self.mode = mode

	def forward(self, x):
		x = x.float()
		if self.mode == 'bilinear':
			x = self.upsample(x, scale_factor=self.scale_factor, mode=self.mode,
			                  align_corners=True)
		else:
			x = self.upsample(x, scale_factor=self.scale_factor, mode=self.mode)
		return x


def decoder_block(in_filters, out_filters, transpose=False, norm=nn.InstanceNorm2d):
	if transpose:
		return nn.Sequential(
			nn.ConvTranspose2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
			norm(out_filters),
			nn.ReLU(inplace=True))

	else:
		return nn.Sequential(conv_ReLU(in_filters, out_filters, 3, stride=1, padding=1, use_norm=True, norm=norm),
		                     MyUpsample(scale_factor=2, mode='bilinear'),
		                     conv_ReLU(out_filters, out_filters, 3, stride=1, padding=1, use_norm=True, norm=norm))


def encoder_block(in_filters, out_filters, norm=nn.InstanceNorm2d):
	"""helper function to return two 3x3 convs with the 1st being stride 2
	"""
	return nn.Sequential(conv_ReLU(in_filters, out_filters, 3, stride=2, padding=1, use_norm=True, norm=norm),
	                     conv_ReLU(out_filters, out_filters, 3, stride=1, padding=1, use_norm=True, norm=norm))


def expand_target(target, prediction):
	"""Expands the target in case of BoW predictions from multiple crops."""
	assert prediction.size(1) == target.size(1)
	batch_size_x_num_crops, num_words = prediction.size()
	batch_size = target.size(0)
	assert batch_size_x_num_crops % batch_size == 0
	num_crops = batch_size_x_num_crops // batch_size

	if num_crops > 1:
		target = target.unsqueeze(1).repeat(1, num_crops, 1).view(-1, num_words)

	return target


def get_template(feat_map, mode="local_average"):
	assert feat_map.dim() == 4
	N, Cf, Hf, Wf = feat_map.shape

	if mode == "local_average":
		features_local = F.avg_pool2d(feat_map, kernel_size=5, stride=1, padding=0)
		heatmap = features_local.sum(1).view(N, -1)  # (N, Hf*Wf)
		_, indices = heatmap.max(dim=1, keepdim=True)
		feat_tmpl = get_feat_crop(features_local, indices)
		feat_tmpl = feat_tmpl.squeeze(dim=1)
	elif mode == "global_average":
		feat_tmpl = util.global_pooling(feat_map, type="avg").flatten(1)

	return feat_tmpl


class BoxGenerator(object):
	"""To generate box based on heatmap in the randomly augmented views.
	"""

	def __init__(self, input_size, min_size, num_patches_per_image, iou_threshold, alpha=1.0):
		self.input_size = input_size
		self.min_size = min_size
		self.num_patches_per_image = num_patches_per_image
		self.iou_threshold = iou_threshold
		self.max_iter = 50

		# a == b == 1.0 is uniform distribution
		self.beta = Beta(alpha, alpha)

	def generate(self, rois):
		spatial_boxes = []
		width, height = self.input_size, self.input_size
		n_samples = rois.size(0)

		for batch_idx, roi_box in enumerate(rois):
			h0, w0, h1, w1 = roi_box
			ch0 = max(int(height * h0), self.min_size // 2)
			ch1 = min(int(height * h1), height - self.min_size // 2)
			cw0 = max(int(width * w0), self.min_size // 2)
			cw1 = min(int(width * w1), width - self.min_size // 2)

			spatial_boxes_image = []
			if ch0 > ch1 or cw0 > cw1:
				spatial_box1 = [batch_idx, 0, 0, self.input_size, self.input_size]
				spatial_boxes_image.append(clip_box(spatial_box1, self.input_size))
				spatial_boxes.append(torch.tensor(spatial_boxes_image))
				continue

			for i in range(self.num_patches_per_image):
				for cnt in range(self.max_iter):  # try 50 times untill IoU condition meets
					ch = ch0 + int((ch1 - ch0) * self.beta.sample())
					cw = cw0 + int((cw1 - cw0) * self.beta.sample())
					h_2 = np.random.randint(self.min_size // 2, min(ch, height - ch) + 1)
					w_2 = np.random.randint(self.min_size // 2, min(cw, width - cw) + 1)

					box1_l = cw - w_2
					box1_r = cw + w_2
					box1_t = ch - h_2
					box1_b = ch + h_2

					if i == 0 or self.iou_threshold == 1.0:
						break

					# reject patches if the generated patch overlaps more than
					# the specific IoU threshold
					max_iou = 0.
					for box in spatial_boxes_image[-i:]:
						iou = bbox_iou([box1_l, box1_t, box1_r, box1_b],
						               [box[1], box[2], box[3], box[4]])
						max_iou = max(max_iou, iou)
						if max_iou > self.iou_threshold:
							break

					if max_iou < self.iou_threshold:
						break

				if cnt == (self.max_iter - 1) and max_iou > self.iou_threshold:
					continue

				# append a spatial box
				spatial_box1 = [batch_idx, box1_l, box1_t, box1_r, box1_b]
				spatial_boxes_image.append(clip_box(spatial_box1, self.input_size))

			spatial_boxes.append(torch.tensor(spatial_boxes_image))

		return spatial_boxes


def bbox_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if interArea == 0:
		return 0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
	boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou


def clip_box(box_with_inds, input_size):
	box_with_inds[1] = float(max(0, box_with_inds[1]))
	box_with_inds[2] = float(max(0, box_with_inds[2]))
	box_with_inds[3] = float(min(input_size, box_with_inds[3]))
	box_with_inds[4] = float(min(input_size, box_with_inds[4]))

	return box_with_inds


if __name__ == '__main__':
	pass
