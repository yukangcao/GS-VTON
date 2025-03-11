# Self Correction for Human Parsing

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
## Requirements

```
conda env create schp 
conda activate schp
pip3 install -r requirements.txt
```

**Pretrain ResNet101** ([resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth))
```
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O pretrain_model/resnet101-imagenet.pth
```


Please download the [LIP](http://sysu-hcp.net/lip/) dataset following the below structure.

```commandline
data/LIP
|--- train_imgaes # 30462 training single person images
|--- val_images # 10000 validation single person images
|--- train_segmentations # 30462 training annotations
|--- val_segmentations # 10000 training annotations
|--- train_id.txt # training image list
|--- val_id.txt # validation image list
```
## Training MHP dataset (without schp)
```bash
git clone https://github.com/KudoKhang/Human-Parsing
cd Human-Parsing
```
Download ATR.pth pretrained
```bash
sudo -H pip3 install gdown
mkdir pretrain_model
gdown https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP -O pretrain_model/atr.pth
```

```
!python train.py --num-classes 8 --batch-size 2 --gpu '0' --schp-start 1000 --data-dir './MHP' --eval-epochs 1 --imagenet-pretrain './pretrain_model/atr.pth'
```

Or:
```bash
bash train.sh
```

## Evaluation
```
python evaluate.py --model-restore [CHECKPOINT_PATH]
```
CHECKPOINT_PATH should be the path of trained model.

## Citation

Please cite our work if you find this repo useful in your research.

```latex
@article{li2020self,
  title={Self-Correction for Human Parsing}, 
  author={Li, Peike and Xu, Yunqiu and Wei, Yunchao and Yang, Yi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2020},
  doi={10.1109/TPAMI.2020.3048039}}
```

## Visualization

* Source Image.
![demo](./demo/demo.jpg)
* LIP Parsing Result.
![demo-lip](./demo/demo_lip.png)
* ATR Parsing Result.
![demo-atr](./demo/demo_atr.png)
* Pascal-Person-Part Parsing Result.
![demo-pascal](./demo/demo_pascal.png)
