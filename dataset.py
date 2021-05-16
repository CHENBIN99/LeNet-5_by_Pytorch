# -*- coding: utf-8 -*-
import os
from torch.utils.data import Dataset
from PIL import Image


class Cifar10Dataset(Dataset):
    def __init__(self, img_path, transform=None, target_transform=None):
        self.imgs = []
        img_list = os.listdir(img_path)
        for i in img_list:
            label = i.split('_')[0]
            self.imgs.append({'label': label, "img": os.path.join(img_path, i)})
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.imgs[index]
        # label = [0.0]*10
        # label[int(data['label'])] = 1.0
        label = float(data['label'])
        img_path = data['img']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return label, img

    def __len__(self):
        return len(self.imgs)
