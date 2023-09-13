#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
"""

__author__ = "GZ"

import os
import sys

import torchvision.datasets as datasets

# Root directory of the project
try:
	abspath = os.path.abspath(__file__)
except NameError:
	abspath = os.getcwd()
ROOT_DIR = os.path.dirname(abspath)


class CIFAR10Instance(datasets.CIFAR10):
    def __init__(self, root="./", train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index


class CIFAR100Instance(datasets.CIFAR100):
    def __init__(self, root="./", train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, index


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class ImageFolderSubset(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)
    """
    def __init__(self, class_path, root, transform, **kwargs):
        super().__init__(root, transform, **kwargs)
        self.class_path = class_path
        new_samples, sorted_classes = self.get_class_samples()
        self.imgs = self.samples = new_samples  # len=126689
        self.classes = sorted_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
        self.targets = [s[1] for s in self.samples]

    def get_class_samples(self):
        classes = open(self.class_path).readlines()
        classes = [m.strip() for m in classes]
        classes = set(classes)
        class_to_sample = [[os.path.basename(os.path.dirname(m[0])), m] for m in self.imgs]
        selected_samples = [m[1] for m in class_to_sample if m[0] in classes]

        sorted_classes = sorted(list(classes))
        target_mapping = {self.class_to_idx[k]: j for j, k in enumerate(sorted_classes)}

        valid_pairs = [[m[0], target_mapping[m[1]]] for m in selected_samples]
        return valid_pairs, sorted_classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class COCOInstance(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)

        self.samples = []
        for index in range(len(self.ids)):
            id = self.ids[index]
            path = self.coco.loadImgs(id)[0]["file_name"]
            self.samples.append([path, id])

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, index


if __name__ == '__main__':
    import torchvision.transforms as transforms

    class_path = "./data/ImageNet/imagenet100.txt"
    root = "../data/ImageNet/val"

    img_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(int(img_size * (8. / 7))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])

    dataset = ImageFolderSubset(class_path, root, transform)
