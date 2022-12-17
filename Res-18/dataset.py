import numpy as np
import cv2
import matplotlib
import math
import os
import torch.nn.functional as F
import torch
import json
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

MAPS = ['map3','map4']
Scales = [0.9, 1.1]
MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class Gamma(object):
    def __init__(self, gamma_range, prob):
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, img):
        img = np.array(img)
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            gamma_table = [np.power(x / 255.0, 1 / gamma) * 255 for x in range(256)]
            gamma_table = np.array(gamma_table).astype(np.uint8)
            img = cv2.LUT(img, gamma_table).astype(np.uint8)
        img = Image.fromarray(img)
        return img


class FSCTransform(object):
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        Normalize = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


class FSCTransformWithGT(object):
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        Normalize = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)

        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)

        gamma_range = [0.8, 1.25]
        prob = 0.5
        gamma_fn = Gamma(gamma_range, prob)
        resized_image = gamma_fn(resized_image)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample


class FSCDataset(Dataset):
    def __init__(self, img_dir, density_dir, meta_file, train=True):
        super(FSCDataset, self).__init__()
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.meta_file = meta_file
        self.train = train

        # Construct metas
        if isinstance(meta_file, str):
            meta_file = [meta_file]
        self.metas = []
        for _meta_file in meta_file:
            with open(_meta_file, "r+") as f_r:
                for line in f_r:
                    meta = json.loads(line)
                    self.metas.append(meta)
                    
    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        Transform = transforms.Compose([FSCTransform(MAX_HW)])
        TransformTrain = transforms.Compose([FSCTransformWithGT(MAX_HW)])

        # Read image
        img_name = meta["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Read density map
        density_name = meta["density"]
        density_path = os.path.join(self.density_dir, density_name)
        density = np.load(density_path)

        # Get boxes, h, w
        boxes = meta["boxes"]


        # Transform
        sample = {
            'image': image,
            'lines_boxes': boxes,
            'gt_density': density
        }

        if self.train:
            sample = TransformTrain(sample)
        else:
            sample = Transform(sample)
        image, boxes, density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()

        dataset = {
            "filename": img_name,
            "image": image,
            "density": density,
            "boxes": boxes,
        }
        return dataset