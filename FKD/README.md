## 🚀 FKD: A Fast Knowledge Distillation Framework for Visual Recognition

Official PyTorch implementation of paper [**A Fast Knowledge Distillation Framework for Visual Recognition**](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf) (ECCV 2022, [ECCV paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf), [arXiv](https://arxiv.org/abs/2112.01528)), Zhiqiang Shen and Eric Xing.


<div align=center>
<img width=55% src="FKD.png"/>
</div>

### Abstract

Knowledge Distillation (KD) has been recognized as a useful tool in many visual tasks, such as the **supervised classification** and **self-supervised representation learning**. While the main drawback of a vanilla KD framework lies in its mechanism that most of the computational overhead is consumed on forwarding through the giant teacher networks, which makes the whole learning procedure a low-efficient and costly manner. 

**🚀 Fast Knowledge Distillation (FKD)** is a novel framework that addresses the low-efficiency drawback, simulates the distillation training phase, and generates soft labels following the multi-crop KD procedure, meanwhile enjoying a faster training speed than other methods. FKD is even more efficient than the conventional classification framework when employing multi-crop in the same image for data loading. It achieves **80.1%** (SGD) and **80.5%** (AdamW) using ResNet-50 on ImageNet-1K with plain training settings. This work also demonstrates the efficiency advantage of FKD on the self-supervised learning task.

## Citation

	@article{shen2021afast,
	      title={A Fast Knowledge Distillation Framework for Visual Recognition}, 
	      author={Zhiqiang Shen and Eric Xing},
	      year={2021},
	      journal={arXiv preprint arXiv:2112.01528}
	}

## What's New
* Please refer to our work [here](https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L/relabel#make-fkd-compatible-with-mixup-and-cutmix) if you would like to utilize mixture-based data augmentations (Mixup, CutMix, etc.) during the soft label generation and model training.
* Includes [code of soft label generation](FKD_SLG) for customization. We will also set up a [soft label zoo and baselines](FKD_SLG) with multiple soft labels from various teachers. 
* FKD with AdamW on ResNet-50 achieves **80.5%** using a plain training scheme. Pre-trained model is available [here](https://drive.google.com/file/d/14HgpE-9SMOFUN3cb7gT9OjURqq7s7q2_/view?usp=sharing).



## Supervised Training

### Preparation

- Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). This repo has minimal modifications on that code. 

- Download our soft label and unzip it. We provide multiple types of [soft labels](http://zhiqiangshen.com/projects/FKD/index.html), and we recommend to use [Marginal Smoothing Top-5 (500-crop)](https://drive.google.com/file/d/14leI6xGfnyxHPsBxo0PpCmOq71gWt008/view?usp=sharing). 

- [Optional] Generate customized soft labels using [./FKD_SLG](FKD_SLG).


### FKD Training on CNNs

To train a model, run `train_FKD.py` with the desired model architecture and the path to the soft label and ImageNet dataset:


```
python train_FKD.py -a resnet50 --lr 0.1 --num_crops 4 -b 1024 --cos --temp 1.0 --softlabel_path [soft label path] [imagenet-folder with train and val folders]
```

Add `--mixup_cutmix` to enable Mixup and Cutmix augmentations. For `--softlabel_path`, use format as `./FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet`.

Multi-processing distributed training on a single node with multiple GPUs:

```
python train_FKD.py \
--dist-url 'tcp://127.0.0.1:10001' \
--dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 \
-a resnet50 --lr 0.1 --num_crops 4 -b 1024 \
--temp 1.0 --cos -j 32 \
--save_checkpoint_path ./FKD_nc_4_res50_plain \
--softlabel_path [soft label path, e.g., ./FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet] \
[imagenet-folder with train and val folders]
```


**For multiple nodes multi-processing distributed training, please refer to [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet) for details.**


### Evaluation

```
python train_FKD.py -a resnet50 -e --resume [model path] [imagenet-folder with train and val folders]
```

### Training Speed Comparison

The training speed of each epoch is tested on HPC/CIAI cluster at MBZUAI with 8 NVIDIA V100 GPUs. The batch size is 1024 for all three methods: **(i)** regular/vanilla classification framework, **(ii)** Relabel and **(iii)** FKD. For `Vanilla` and `ReLabel`, we use the average of 10 epochs after the speed is stable. For FKD, we perform `num_crops = 4` to calculate the average of (4 $\times$ 10) epochs, note that using 8 will give faster training speed. All other settings are the same for the comparison.

| Method |  Network  | Training time per-epoch |
|:-------:|:--------:|:--------:|
| Vanilla | ResNet-50 |  579.36 sec/epoch | 
| ReLabel | ResNet-50   | 762.11 sec/epoch |  
| FKD (Ours) | ResNet-50 |  486.77 sec/epoch | 

### Trained Models

| Method |  Network  | accuracy (Top-1)  |weights  |configurations |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| [`ReLabel`](https://github.com/naver-ai/relabel_imagenet) | ResNet-50 | 78.9 | -- |  -- |
| `FKD`| ResNet-50 | &emsp; &emsp;   **80.1<sup>+1.2%</sup>** | [link](https://drive.google.com/file/d/1qQK3kae4pXBZOldegnZqw7j_aJWtbPgV/view?usp=sharing) | same as ReLabel while initial lr = 0.1 $\times$ $batch size \over 512$ |
| | | |
| `FKD`<sub>(Plain)</sub>| ResNet-50 |  **79.8** | [link](https://drive.google.com/file/d/1s6Tx5xmXnAseMZJBwaa4bnuvzZZGjMdk/view?usp=sharing) |  [Table 12 in paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)<br><sub>(w/o warmup&colorJ )</sub>  |
| `FKD`<sub>(AdamW)</sub> | ResNet-50 | **80.5** | [link](https://drive.google.com/file/d/14HgpE-9SMOFUN3cb7gT9OjURqq7s7q2_/view?usp=sharing) |  [Table 13 in paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)<br><sub>(same as our settings on ViT and SReT)</sub> |
| | | |
| [`ReLabel`](https://github.com/naver-ai/relabel_imagenet) | ResNet-101 | 80.7  | -- |  -- | 
| `FKD` | ResNet-101 | &emsp; &emsp; **81.9<sup>+1.2%</sup>** | [link](TBA) |  [Table 12 in paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)  |
| | | |
| `FKD`<sub>(Plain)</sub>| ResNet-101 | **81.7**  | [link](https://drive.google.com/file/d/13bVpHpTykCaYYXIAbWHa2W2C2tSxZlW5/view?usp=sharing) |  [Table 12 in paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)<br><sub>(w/o warmup&colorJ )</sub>  |  

### Mobile-level Efficient Networks

| Method |  Network  | FLOPs | accuracy (Top-1)  |weights  |
|:-------:|:--------:|:--------:|:--------:|:--------:|
| [`FBNet`](https://arxiv.org/abs/1812.03443)| FBNet-c100 | 375M | 75.12% | -- | 
| `FKD`| FBNet-c100 | 375M | &emsp; &emsp;  **77.13%<sup>+2.01%</sup>** | [link](https://drive.google.com/file/d/1s2pnIedXgwYAPpY2GBT3OC24ZP-0vfWe/view?usp=sharing) |  
| | | |
| [`EfficientNetv2`](https://arxiv.org/abs/2104.00298)| EfficientNetv2-B0 | 700M | 78.35% | -- | 
| `FKD`| EfficientNetv2-B0 | 700M | &emsp; &emsp;   **79.94%<sup>+1.59%</sup>** | [link](https://drive.google.com/file/d/1qL21XOnTRWt6CvZLvUY5IpULISESEfZm/view?usp=sharing) |  

The training protocol is the same as we used for ViT/SReT:

```
# Use the same settings as on ViT and SReT
cd train_ViT
# Train the model
python -u train_ViT_FKD.py \
--dist-url 'tcp://127.0.0.1:10001' \
--dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 \
-a tf_efficientnetv2_b0 \
--lr 0.002 --wd 0.05 \
--epochs 300 --cos -j 32 \
--num_classes 1000 --temp 1.0 \
-b 1024 --num_crops 4 \
--save_checkpoint_path ./FKD_nc_4_224_efficientnetv2_b0 \
--soft_label_type marginal_smoothing_k5  \
--softlabel_path [soft label path] \
[imagenet-folder with train and val folders]
```

### FKD Training on ViT/DeiT and SReT

To train a ViT model, run `train_ViT_FKD.py` with the desired model architecture and the path to the soft label and ImageNet dataset:

```
cd train_ViT
python train_ViT_FKD.py \
--dist-url 'tcp://127.0.0.1:10001' \
--dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 \
-a SReT_LT --lr 0.002 --wd 0.05 --num_crops 4 \
--temp 1.0 -b 1024 --cos \
--softlabel_path [soft label path] \
[imagenet-folder with train and val folders]
```

For the instructions of `SReT_LT` model, please refer to [SReT](https://github.com/szq0214/SReT) for details.

### Evaluation

```
python train_ViT_FKD.py -a SReT_LT -e --resume [model path] [imagenet-folder with train and val folders]
```

### Trained Models

| Model    | FLOPs| #params | accuracy (Top-1) |weights  |configurations |
|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [`DeiT-T-distill`](https://github.com/facebookresearch/deit) | 1.3B  | 5.7M | 74.5  |-- |  -- |
| `FKD ViT/DeiT-T` | 1.3B  | 5.7M | **75.2**  |[link](https://drive.google.com/file/d/1m33c1wHdCV7ePETO_HvWNaboSd_W4nfC/view?usp=sharing) |  [Table 13 in paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf) |
| [`SReT-LT-distill`](https://github.com/szq0214/SReT)  |  1.2B | 5.0M | 77.7  |-- |  --  |
| `FKD SReT-LT`    |  1.2B | 5.0M | **78.7**  |[link](https://drive.google.com/file/d/1mmdPXKutHM9Li8xo5nGG6TB0aAXA9PFR/view?usp=sharing) |  [Table 13 in paper](http://zhiqiangshen.com/projects/FKD/FKD_camera-ready.pdf)  |

## Fast MEAL V2

Please see [MEAL V2](https://github.com/szq0214/MEAL-V2) for the instructions to run FKD with MEAL V2.

## Self-supervised Representation Learning Using FKD

Please see [FKD-SSL](https://github.com/szq0214/FKD/tree/main/FKD_SSL) for the instructions to run FKD for SSL task.


## Contact

Zhiqiang Shen (zhiqiangshen0214 at gmail.com or zhiqians at andrew.cmu.edu) 

