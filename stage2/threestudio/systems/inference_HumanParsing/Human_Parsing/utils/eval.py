import os
import argparse
import numpy as np
import torch

from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import networks
from datasets.datasets import LIPDataValSet
from utils.miou import compute_mean_ioU
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing
import time
import warnings
warnings.filterwarnings("ignore")


class Eval():
    def __init__(self, arch='resnet101', data_dir='./MHP', batch_size=1, input_size='473, 473', num_classes=8,
                 ignore_label=255, log_dir='./log', model_restore='./log/checkpoint_last.pth.tar', gpu='0', multi_scales='1',
                 flip=False):
        self.arch = arch
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.log_dir = log_dir
        self.model_restore = model_restore
        self.gpu = gpu
        self.multi_scales = multi_scales
        self.flip = flip

    def multi_scale_testing(self, model, batch_input_im, crop_size=[473, 473], flip=True, multi_scales=[1]):
        flipped_idx = (15, 14, 17, 16, 19, 18)
        if len(batch_input_im.shape) > 4:
            batch_input_im = batch_input_im.squeeze()
        if len(batch_input_im.shape) == 3:
            batch_input_im = batch_input_im.unsqueeze(0)

        interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
        ms_outputs = []
        for s in multi_scales:
            interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
            scaled_im = interp_im(batch_input_im)
            parsing_output = model(scaled_im)
            parsing_output = parsing_output[0][-1]
            output = parsing_output[0]
            if flip:
                flipped_output = parsing_output[1]
                flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
                output += flipped_output.flip(dims=[-1])
                output *= 0.5
            output = interp(output.unsqueeze(0))
            ms_outputs.append(output[0])
        ms_fused_parsing_output = torch.stack(ms_outputs)
        ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
        ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC
        parsing = torch.argmax(ms_fused_parsing_output, dim=2)
        parsing = parsing.data.cpu().numpy()
        ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
        return parsing, ms_fused_parsing_output

    def run(self):
        multi_scales = [float(i) for i in self.multi_scales.split(',')]
        gpus = [int(i) for i in self.gpu.split(',')]
        assert len(gpus) == 1
        if not self.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu

        cudnn.benchmark = True
        cudnn.enabled = True

        h, w = map(int, self.input_size.split(','))
        input_size = [h, w]

        model = networks.init_model(self.arch, num_classes=self.num_classes, pretrained=None)

        IMAGE_MEAN = model.mean
        IMAGE_STD = model.std
        INPUT_SPACE = model.input_space

        if INPUT_SPACE == 'BGR':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN,
                                     std=IMAGE_STD),
            ])
        if INPUT_SPACE == 'RGB':
            transform = transforms.Compose([
                transforms.ToTensor(),
                BGR2RGB_transform(),
                transforms.Normalize(mean=IMAGE_MEAN,
                                     std=IMAGE_STD),
            ])

        # Data loader
        lip_test_dataset = LIPDataValSet(self.data_dir, 'val', crop_size=input_size, transform=transform,
                                         flip=self.flip)
        num_samples = len(lip_test_dataset)
        testloader = data.DataLoader(lip_test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        # Load model weight
        state_dict = torch.load(self.model_restore)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            pass

        model.cuda()
        model.eval()

        sp_results_dir = os.path.join(self.log_dir, 'sp_results')
        if not os.path.exists(sp_results_dir):
            os.makedirs(sp_results_dir)

        parsing_preds = []
        scales = np.zeros((num_samples, 2), dtype=np.float32)
        centers = np.zeros((num_samples, 2), dtype=np.int32)
        with torch.no_grad():
            total_time = 0
            for idx, batch in enumerate(tqdm(testloader)):
                image, meta = batch
                if (len(image.shape) > 4):
                    image = image.squeeze()
                im_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]
                scales[idx, :] = s
                centers[idx, :] = c

                parsing, logits = self.multi_scale_testing(model, image.cuda(), crop_size=input_size, flip=self.flip,
                                                           multi_scales=multi_scales)

                parsing_preds.append(parsing)
        mIoU = compute_mean_ioU(parsing_preds, scales, centers, self.num_classes, self.data_dir, input_size)
        # print(mIoU)
        return mIoU
