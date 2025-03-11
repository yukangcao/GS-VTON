#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import timeit
import argparse
import cv2
import albumentations as A

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils import data

import networks
from datasets.datasets import LIPDataSet
from datasets.target_generation import generate_edge_tensor
import utils.schp as schp
from utils.transforms import BGR2RGB_transform
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler

from utils.eval import Eval
from utils.predict import HumanParsing
import csv
import warnings

warnings.filterwarnings("ignore")
import wandb

os.environ["WANDB_API_KEY"] = 'e7ed558aefc5cddf29d04c3037a712507b253521'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=str, default='473, 473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0,1,2')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str,
                        default='./pretrain_model/resnet101-imagenet.pth')  # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint_last.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    parser.add_argument("--project-name", type=str, default='HumanParsing', help='name of project log in wandb')
    parser.add_argument("--num-worker", type=int, default=8, help='num-worker < or = < !cat /proc/cpuinfo | grep processor | wc -l >')
    parser.add_argument("--is-first", action="store_true")
    return parser.parse_args()

def log2wandb(lr, loss, project, epoch, data_dir, log_dir):
    wandb.init(project=project, entity='khanghn')
    wandb.log({"Lr: ": lr, "Loss: ": loss.data.cpu()})

    root = os.path.join(data_dir, 'val_images')
    path_image_val = [name for name in os.listdir(root)]
    path = os.path.join(root, path_image_val[0])

    class_labels = {0: 'Background', 1: 'Face', 2: 'Hair', 3: 'Left-leg', 4: 'Right-leg',
                    5: 'Left-arm', 6: 'Right-arm', 7: 'Torso-skin'}

    def log(weight=os.path.join(log_dir, "model_parsing_best.pth.tar"), mode='Best', data_dir=None):
        mIoU = eval(weight, data_dir)
        for key, value in mIoU.items():
            wandb.log({f"{key} [{mode}]": value})

        # save to csv
        mIoU.update({"Lr: ": lr, "Loss: ": loss.data.cpu(), "Epoch": epoch})
        values = []
        for _, value in mIoU.items():
            values.append(value)


        with open(os.path.join(log_dir, f"log_{mode}.csv"), "a") as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(values)

        Human_parsing_predictor = HumanParsing(dataset='mhp', weight=weight)

        log_image = cv2.imread(path)
        log_mask = Human_parsing_predictor.run(log_image)
        mask_img = wandb.Image(log_image[:, :, ::-1],
                               caption=f"Prediction {mode}",
                               masks={"predictions":
                                          {"mask_data": log_mask,
                                           "class_labels": class_labels}})
        wandb.log({f'mask-{mode}': mask_img})
    log(weight=os.path.join(log_dir, "model_parsing_best.pth.tar"), mode='Best', data_dir=data_dir)
    log(weight=os.path.join(log_dir, "checkpoint_last.pth.tar"), mode='Last', data_dir=data_dir)


def eval(model_checkpoint='./log/model_parsing_best.pth.tar', data_dir=None):
    eval_model = Eval(model_restore=model_checkpoint, data_dir=data_dir)
    mIoU = eval_model.run()
    return mIoU

def init_log_csv(log_dir):
    keys = ['Background',
            'Face',
            'Hair',
            'Left-leg',
            'Right-leg',
            'Left-arm',
            'Right-arm',
            'Torso-skin',
            'Pixel accuracy',
            'Mean accuracy',
            'Mean IU',
            'Lr',
            'Loss',
            'Epoch']

    mode = ['Best', 'Last']
    for i in mode:
        with open(os.path.join(log_dir, f"log_{i}.csv"), "a") as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(keys)


def main():
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0
    mIU_max = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model Initialization
    AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    model = DataParallelModel(AugmentCE2P)
    model.cuda()

    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))

    restore_from = args.model_restore
    if os.path.exists(restore_from):
        checkpoint = torch.load(restore_from)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print('Resume training from {}, start at epoch: {}'.format(restore_from, start_epoch))

    SCHP_AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    schp_model.cuda()

    if os.path.exists(args.schp_restore):
        print('Resuming schp checkpoint from {}'.format(args.schp_restore))
        schp_checkpoint = torch.load(args.schp_restore)
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = schp_checkpoint['cycle_n']
        schp_model.load_state_dict(schp_model_state_dict)

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # Data Loader
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')

        augment = A.Compose([
            A.GaussNoise(p=0.3),
            A.MedianBlur(p=0.3),
            A.Blur(p=.3),
            A.CLAHE(p=.2),
            A.ChannelShuffle(p=.2),
            A.HueSaturationValue(p=.2),
            A.InvertImg(p=.1),
            A.RGBShift(p=.2)
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')

        augment = A.Compose([
            A.GaussNoise(p=0.3),
            A.MedianBlur(p=0.3),
            A.Blur(p=.3),
            A.CLAHE(p=.2),
            A.ChannelShuffle(p=.2),
            A.HueSaturationValue(p=.2),
            A.InvertImg(p=.1),
            A.RGBShift(p=.2)
        ])

        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # Dataloader
    train_dataset = LIPDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform, augment=augment)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size * len(gpus),
                                   num_workers=args.num_worker, shuffle=True, pin_memory=True, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                 eta_min=args.learning_rate / 100, warmup_epoch=10,
                                 start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / 2,
                                 cyclical_epoch=args.cycle_epochs)

    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()

    if args.is_first:
        init_log_csv(args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step(epoch=epoch)
        lr = lr_scheduler.get_lr()[0]

        model.train()
        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader) * epoch

            images, labels, _ = batch
            labels = labels.cuda(non_blocking=True)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)

            preds = model(images)

            # Online Self Correction Cycle with Label Refinement
            """
                cycle_n = (epoch - schp_start(100)) // 10
            """
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds = schp_model(images)
                    soft_parsing = []
                    soft_edge = []

                    # ------ Author's code ------
                    # for soft_pred in soft_preds:
                    #     soft_parsing.append(soft_pred[0][-1])
                    #     soft_edge.append(soft_pred[1][-1])      # <- BUGS: IndexError: list index out of range
                    # soft_preds = torch.cat(soft_parsing, dim=0)
                    # soft_edges = torch.cat(soft_edge, dim=0)

                    # ------ Khang's code ------
                    """
                        Output model: [[parsing result1, parsing result2],[edge result]]
                        Viết lại code để shape phù hợp với shape của def parsing_loss() cần
                        Size ảnh sau khi qua model sẽ còn 119 * 119, tuy nhiên các hàm xử lý phía sau sẽ
                            resize bằng F.interpolate về size ban đầu
                    """
                    soft_edges = soft_preds[1][0]  # (2, 2, 119, 119) # edge
                    for soft_pred in soft_preds[0]:
                        soft_parsing.append(torch.unsqueeze(soft_pred[-1], 0))

                    soft_preds = torch.cat(soft_parsing, dim=0)  # (2, 8, 119, 119) # parsing

            else:
                soft_preds = None
                soft_edges = None

            loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                print('iter = {} of {} completed, lr = {}, loss = {}'.format(i_iter, total_iters, lr,
                                                                             loss.data.cpu().numpy()))
        """
            Giả sử eval_epochs = 10, thì cứ sau 10 epoch sẽ lưu weight checkpoint 1 lần
        """
        if (epoch + 1) % (args.eval_epochs) == 0:

            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': loss,
                'mIU_max': mIU_max
            }

            schp.save_schp_checkpoint(states, args.log_dir, filename='checkpoint_last.pth.tar')

            # save best
            result_metrics = eval(model_checkpoint=os.path.join(args.log_dir, 'checkpoint_last.pth.tar'), data_dir=args.data_dir)
            mIU = result_metrics['Mean IU']

            status = f'\n------Saved best weight. mIU_max = {mIU_max} at Epoch: {epoch + 1}------\n' if mIU > mIU_max else f'\n------Save failed------\n'
            print(status)

            if mIU > mIU_max:
                mIU_max = mIU
                best_save_path = os.path.join(args.log_dir, 'model_parsing_best.pth.tar')
                if os.path.exists(best_save_path):
                    os.remove(best_save_path)
                torch.save(states, best_save_path)

            log2wandb(lr, loss, args.project_name, epoch + 1, args.data_dir, args.log_dir)

        # Self Correction Cycle with Model Aggregation
        """
            giả sử cho schp_start = 100, và cycle_epoch = 10 thì từ epoch 100 cứ sau 10 epoch sẽ bắt đầu lưu weight schp 1 lần
        """
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print('Self-correction cycle number {}'.format(cycle_n))
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))  # <=== cập nhật param, cycle_n càng lớn thì param của model schp càng nhiều hơn model gốc
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)  # <=== Dùng weight vừa được cập nhật ở trên để training train_loader
            schp.save_schp_checkpoint({
                'state_dict': schp_model.state_dict(),
                'cycle_n': cycle_n,
            }, False, args.log_dir, filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                             (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
