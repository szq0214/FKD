## FKD: A Fast Knowledge Distillation Framework for Visual Recognition


Official PyTorch implementation of paper [**A Fast Knowledge Distillation Framework for Visual Recognition**](https://arxiv.org/abs/2112.01528). Zhiqiang Shen and Eric Xing from CMU and MUZUAI.

<div align=center>
<img width=60% src="FKD.png"/>
</div>

### Abstract

Knowledge Distillation (KD) has been recognized as a useful tool in many visual tasks, such as the supervised classification and self-supervised representation learning, while the main drawback of a vanilla KD framework lies in its mechanism that most of the computational overhead is consumed on forwarding through the giant teacher networks, which makes the whole learning procedure in a low-efficient and costly manner. In this work, we propose a Fast Knowledge Distillation (FKD) framework that simulates the distillation training phase and generates soft labels following the multi-crop KD procedure, meanwhile enjoying the faster training speed than ReLabel as we have no post-processes like RoI align and softmax operations. Our FKD is even more efficient than the conventional classification framework when employing multi-crop in the same image for data loading. We achieve 79.8% using ResNet-50 on ImageNet-1K, outperforming ReLabel by ~1.0% while being faster. We also demonstrate the efficiency advantage of FKD on the self-supervised learning task.

## Supervised Training

### Preparation

- Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). This repo has minimal modifications on that code. 

- Download our soft label. We provide multiple types of [soft labels](http://zhiqiangshen.com/projects/FKD/index.html), and we recommend to use [Marginal Smoothing Top-5 (500-crop)](https://drive.google.com/file/d/14leI6xGfnyxHPsBxo0PpCmOq71gWt008/view?usp=sharing).


### FKD Training on CNNs

To train a model, run `train_FKD.py` with the desired model architecture and the path to the soft label and ImageNet dataset:


```
python train_FKD.py -a resnet50 --lr 0.1 --num_crops 4 -b 1024 --cos --softlabel_path [soft label path] [imagenet-folder with train and val folders]
```

For `--softlabel_path`, simply use format as `./FKD_soft_label_500_crops_marginal_smoothing_k_5`

**Multi-processing distributed training is supported, please refer to [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet) for details.**


### Evaluation

```
python train_FKD.py -a resnet50 -e --resume [model path] [imagenet-folder with train and val folders]
```

### Trained Models

| Model    | accuracy (Top-1)  |weights  |configurations |
|:-------:|:--------:|:--------:|:--------:|
| [`ReLabel ResNet-50`](https://github.com/naver-ai/relabel_imagenet) |  78.9 | -- |  -- |
| `FKD ResNet-50` |  **79.8** | [link](https://drive.google.com/file/d/1Dj0gZKyWtfSh3Dq70gbIbErJx7siKBsx/view?usp=sharing) |  [Table 10 in paper](https://arxiv.org/pdf/2112.01528.pdf) |
| | | |
| [`ReLabel ResNet-101`](https://github.com/naver-ai/relabel_imagenet)| 80.7  | -- |  -- | 
| `FKD ResNet-101`| **81.7**  | [link](https://drive.google.com/file/d/13bVpHpTykCaYYXIAbWHa2W2C2tSxZlW5/view?usp=sharing) |  [Table 10 in paper](https://arxiv.org/pdf/2112.01528.pdf) |  

### FKD Training on ViT/DeiT and SReT

To train a ViT model, run `train_ViT_FKD.py` with the desired model architecture and the path to the soft label and ImageNet dataset:

```
cd train_ViT
python train_ViT_FKD.py -a SReT_LT --lr 0.002 --wd 0.05 --num_crops 4 -b 1024 --cos --softlabel_path [soft label path] [imagenet-folder with train and val folders]
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
| `FKD ViT/DeiT-T` | 1.3B  | 5.7M | **75.2**  |[link](https://drive.google.com/file/d/1m33c1wHdCV7ePETO_HvWNaboSd_W4nfC/view?usp=sharing) |  [Table 11 in paper](https://arxiv.org/pdf/2112.01528.pdf) |
| [`SReT-LT-distill`](https://github.com/szq0214/SReT)  |  1.2B | 5.0M | 77.7  |-- |  --  |
| `FKD SReT-LT`    |  1.2B | 5.0M | **78.7**  |[link](https://drive.google.com/file/d/1mmdPXKutHM9Li8xo5nGG6TB0aAXA9PFR/view?usp=sharing) |  [Table 11 in paper](https://arxiv.org/pdf/2112.01528.pdf)  |

## Fast MEAL V2

Please see [MEAL V2](https://github.com/szq0214/MEAL-V2) for the instructions to run FKD with MEAL V2.

## Self-supervised Representation Learning Using FKD

Please see [FKD-SSL](https://github.com/szq0214/FKD/tree/main/FKD_SSL) for the instructions to run FKD code for SSL task.

## Citation

	@article{shen2021afast,
	      title={A Fast Knowledge Distillation Framework for Visual Recognition}, 
	      author={Zhiqiang Shen and Eric Xing},
	      year={2021},
	      journal={arXiv preprint arXiv:2112.01528}
	}

## Contact

Zhiqiang Shen (zhiqians at andrew.cmu.edu or zhiqiangshen0214 at gmail.com) 

