import os
import json
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import cv2 as cv

from . import augment

CLASS_NAMES = ['lace', 'fiber']

FLIP_HORIZONTAL = 'flip_horizontal'
FLIP_VERTICAL = 'flip_vertical'

class TextileDatasetStage1(Dataset):
    def __init__(self, data_dir, class_name, split, **kwargs):
        assert class_name in CLASS_NAMES, ('class_name: {}, should be in {}'.format(class_name, CLASS_NAMES))
        self.data_dir = os.path.join(data_dir, class_name)
        self.class_name = class_name
        self.split = split

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.img_paths = []
        self.labels = []
        img_types = os.listdir(os.path.join(self.data_dir, split))
        for img_type in img_types:
            if img_type != 'good' and self.split == 'train':
                continue
            img_names = os.listdir(os.path.join(self.data_dir, split, img_type))
            for img_name in img_names:
                if img_type == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
                self.img_paths.append(os.path.join(self.data_dir, split, img_type, img_name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv.imread(self.img_paths[idx])
        label = self.labels[idx]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label


class TextileDatasetStage2(Dataset):
    def __init__(self, data_dir, class_name, split, augments=[], **kwargs):
        assert class_name in CLASS_NAMES, ('class_name: {}, should be in {}'.format(class_name, CLASS_NAMES))
        self.data_dir = os.path.join(data_dir, class_name)
        self.class_name = class_name
        self.split = split
        self.augments = augments

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.img_paths = []
        self.labels = []
        img_types = os.listdir(os.path.join(self.data_dir, split))
        for img_type in img_types:
            img_names = os.listdir(os.path.join(self.data_dir, split, img_type))
            for img_name in img_names:
                if img_type == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
                self.img_paths.append(os.path.join(self.data_dir, split, img_type, img_name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv.imread(self.img_paths[idx])
        label = self.labels[idx]
        if self.split == 'train':
            if FLIP_HORIZONTAL in self.augments and random.random() < 0.5:
                img = cv.flip(img, 1)
            if FLIP_VERTICAL in self.augments and random.random() < 0.5:
                img = cv.flip(img, 0)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label
