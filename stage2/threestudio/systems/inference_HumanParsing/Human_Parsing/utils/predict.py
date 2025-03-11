import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import cv2
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from utils.transforms import get_affine_transform
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.dataset_settings import *
import warnings
warnings.filterwarnings("ignore")

class HumanParsing():
    def __init__(self, dataset='mhp', weight=None):
        self.dataset = dataset
        self.weight = weight
        self.input_size = dataset_settings[dataset]['input_size']
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.num_classes = dataset_settings[dataset]['num_classes']
        self.model = networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None)
        self.path_pretrained = self.weight

        state_dict = torch.load(self.path_pretrained)['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def preprocessing(self, img_path):
        img = self.check_type(img_path)
        self.img_copy = img.copy()
        h, w, _ = img.shape
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        input = torch.unsqueeze(input, 0)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta

    def run(self, img_path):
        image, meta = self.preprocessing(img_path)

        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']

        output = self.model(image.cuda())
        upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
        parsing_result = np.argmax(logits_result, axis=2)
        return parsing_result