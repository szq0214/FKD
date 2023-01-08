
## Preparation

- Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). This repo has minimal modifications on that code. 

- Download our soft label and unzip it. We provide multiple types of [soft labels](http://zhiqiangshen.com/projects/FKD/index.html), and we recommend to use [Marginal Smoothing Top-5 (500-crop)](https://drive.google.com/file/d/14leI6xGfnyxHPsBxo0PpCmOq71gWt008/view?usp=sharing).


## FKD Training on ViT/DeiT and SReT

To train a ViT model, run `train_ViT_FKD.py` with the desired model architecture and the path to the soft label and ImageNet dataset:

```
python train_ViT_FKD.py \
--dist-url 'tcp://127.0.0.1:10001' \
--dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 \
-a SReT_LT --lr 0.002 --wd 0.05 \
--num_crops 4 -b 1024 --cos \
--temp 1.0 \
--softlabel_path [soft label path, e.g., ./FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet] \
[imagenet-folder with train and val folders]
```

For the instructions of `SReT_LT` model, please refer to [SReT](https://github.com/szq0214/SReT) for details.

## Evaluation

```
python train_ViT_FKD.py -a SReT_LT -e --resume [model path] [imagenet-folder with train and val folders]
```

### Trained Models

| Model    | FLOPs| #params | accuracy (Top-1) |weights  |configurations |
|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [`DeiT-T-distill`](https://github.com/facebookresearch/deit) | 1.3B  | 5.7M | 74.5  |-- |  -- |
| `FKD ViT/DeiT-T` | 1.3B  | 5.7M | **75.2**  |[link](https://drive.google.com/file/d/1m33c1wHdCV7ePETO_HvWNaboSd_W4nfC/view?usp=sharing) |  [Table 13 of paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf) |
| [`SReT-LT-distill`](https://github.com/szq0214/SReT)  |  1.2B | 5.0M | 77.7  |-- |  --  |
| `FKD SReT-LT`    |  1.2B | 5.0M | **78.7**  |[link](https://drive.google.com/file/d/1mmdPXKutHM9Li8xo5nGG6TB0aAXA9PFR/view?usp=sharing) |  [Table 13 of paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)  |


