#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys
from PIL import ImageFilter, Image

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torchvision.transforms as transforms

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)

from samples.transforms.utils_transforms import GaussianBlur


class MultiCropsTransform(object):
    """
    The code is modified from
    https://github.com/maple-research-lab/AdCo/blob/b8f749db3e8e075f77ec17f859e1b2793844f5d3/data_processing/MultiCrop_Transform.py
    """
    def __init__(
            self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,normalize,init_size=224):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #image_k
        weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(weak)
        trans_weak=[]

        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )


            weak=transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
            trans_weak.extend([weak]*nmb_crops[i])

        trans.extend(trans_weak)
        self.trans=trans
        print("in total we have %d transforms"%(len(self.trans)))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


class ssl_transforms(object):
	"""
	A stochastic data augmentation module that transforms any given data example randomly
	resulting in two correlated views of the same example,
	denoted x ̃i and x ̃j, which we consider as a positive pair.
	"""

	def __init__(self, isTrain, img_size, mean, std, p_blur=0.5, multi=False):
		self.isTrain = isTrain
		normalize = transforms.Normalize(mean=mean, std=std)

		if isTrain:
			# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
			self.train_transform = transforms.Compose([
				transforms.RandomResizedCrop(img_size, scale=(0.2, 1.)),
				transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=p_blur),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize
			])
		else:
			if img_size < 224:
				self.test_transform = transforms.Compose([
					transforms.ToTensor(),
					normalize
				])
			else:
				self.test_transform = transforms.Compose([
					transforms.Resize(int(img_size * (8. / 7))),
					transforms.CenterCrop(img_size),
					transforms.ToTensor(),
					normalize
				])

	def __call__(self, x):
		if self.isTrain:
			img1 = self.train_transform(x)
			img2 = self.train_transform(x)
			return [img1, img2]
		else:
			return self.test_transform(x)


class ReSSL_Multi_Transform(object):
    def __init__(
            self,
            size_crops=[224, 192, 160, 128, 96],
            nmb_crops=[1, 1, 1, 1, 1],
            min_scale_crops=[0.2, 0.172, 0.143, 0.114, 0.086],
            max_scale_crops=[1.0, 0.86, 0.715, 0.571, 0.429],
            init_size=224,
            strong=False):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]

        self.strong = strong

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #image_k
        weak = transforms.Compose([
            transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        trans.append(weak)


        trans_weak=[]
        if strong:
            min_scale_crops=[0.08, 0.08, 0.08, 0.08, 0.08]
            jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        else:
            jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)


        for i in range(len(size_crops)):
            aug_list = [
                transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i])
                ),
                transforms.RandomApply([jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
            ]

            if self.strong:
                aug_list.append(RandAugment(5, 10))

            aug_list.extend([
                transforms.ToTensor(),
                normalize
            ])

            aug = transforms.Compose(aug_list)
            trans_weak.extend([aug]*nmb_crops[i])

        trans.extend(trans_weak)
        self.trans=trans
        print("in total we have %d transforms"%(len(self.trans)))
    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops


class ressl_transforms(object):
	"""
	A stochastic data augmentation module that transforms any given data example randomly
	resulting in two correlated views of the same example,
	denoted x ̃i and x ̃j, which we consider as a positive pair.
	"""

	def __init__(self, isTrain, img_size, mean, std, p_blur=0.5, multi=False):
		self.isTrain = isTrain
		self.multi = multi
		normalize = transforms.Normalize(mean=mean, std=std)

		if isTrain:
			if self.multi:
				self.multi_transform = ReSSL_Multi_Transform()

			# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
			self.train_transform = transforms.Compose([
				transforms.RandomResizedCrop(img_size, scale=(0.2, 1.)),
				transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([GaussianBlur([.1, 2.])], p=p_blur),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize
			])

			self.weak_transform = transforms.Compose([
				transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1)),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			])
		else:
			if img_size < 224:
				self.test_transform = transforms.Compose([
					transforms.ToTensor(),
					normalize
				])
			else:
				self.test_transform = transforms.Compose([
					transforms.Resize(int(img_size * (8. / 7))),
					transforms.CenterCrop(img_size),
					transforms.ToTensor(),
					normalize
				])

	def __call__(self, x):
		if self.isTrain:
			if self.multi:
				return self.multi_transform(x)
			else:
				img1 = self.train_transform(x)
				img2 = self.weak_transform(x)
				return [img2, img1]
		else:
			return self.test_transform(x)


class lincls_transforms(object):
	"""
	A stochastic data augmentation module that transforms any given data example randomly
	resulting in two correlated views of the same example,
	denoted x ̃i and x ̃j, which we consider as a positive pair.
	"""

	def __init__(self, isTrain, img_size, mean, std, multi=False):
		self.isTrain = isTrain
		normalize = transforms.Normalize(mean=mean, std=std)

		if isTrain:
			self.train_transform = transforms.Compose([
				transforms.RandomResizedCrop(img_size),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize
			])
		else:
			if img_size < 224:
				self.test_transform = transforms.Compose([
					transforms.ToTensor(),
					normalize
				])
			else:
				self.test_transform = transforms.Compose([
					transforms.Resize(int(img_size * (8. / 7))),
					transforms.CenterCrop(img_size),
					transforms.ToTensor(),
					normalize
				])

	def __call__(self, x):
		if self.isTrain:
			img = self.train_transform(x)
			return img
		else:
			return self.test_transform(x)


if __name__ == '__main__':
	from PIL import ImageFilter, Image
	from utils.visualizer import dump_image

	img = Image.open("./ILSVRC2012_val_00002457.JPEG")

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	augment = ressl_transforms(True, 224, mean, std)
	img = augment(img)

	filedir = "./img"
	os.makedirs(filedir, exist_ok=True)

	filepath = os.path.join(filedir, "{}.png".format("imgweak"))
	dump_image(img[0], mean, std, filepath=filepath)
	filepath = os.path.join(filedir, "{}.png".format("imgstrong"))
	dump_image(img[1], mean, std, filepath=filepath)

	# import torchvision.datasets as datasets
	# from configs.disclr.cifar10cfg import Cifar10Config
	# from configs.disclr.tinyimagenetcfg import TinyImageNetConfig
	# from samples.tinyimagenet import TinyImageNet
	# from utils.visualizer import dump_image
	#
	# # cfg = Cifar10Config("pretrain")
	# cfg = TinyImageNetConfig("pretrain")
	# cfg.display()
	#
	# if cfg.DATASET.lower() == "cifar10":
	# 	dataset = datasets.CIFAR10(cfg.DATA_ROOT, train=True, transform=getTransform(True, cfg), download=True)
	# elif cfg.DATASET.lower() == "tinyimagenet":
	# 	dataset = TinyImageNet(cfg.DATA_ROOT, split="train", transform=getTransform(True, cfg))
	# loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
	#
	# img = Image.open("./imagenet_sample_image.png")
	# preprocess = getTransform(True, cfg)
	# img = preprocess(img)
	# dump_image(img["imgOri"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, dir="./img", name="imgOri")
	# dump_image(img["imgAug1"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, dir="./img", name="imgAug1")
	# dump_image(img["imgAug2"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, dir="./img", name="imgAug2")
	# dump_image(img["imgApp"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, dir="./img", name="imgApp")
	# dump_image(img["imgDef"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, dir="./img", name="imgDef")
	#
	# isshow = True
	# for i, (images, target) in enumerate(loader):
	# 	dump_image(images["imgOri"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, show=isshow)
	# 	dump_image(images["imgAug1"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, show=isshow)
	# 	dump_image(images["imgAug2"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, show=isshow)
	# 	dump_image(images["imgApp"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, show=isshow)
	# 	dump_image(images["imgDef"], cfg.IMAGE_MEAN, cfg.IMAGE_STD, show=isshow)
