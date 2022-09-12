
## Soft Label Generation (SLG)

### Preparation

- Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).
- Install `timm` using:

	```
	pip install git+https://github.com/rwightman/pytorch-image-models.git
	```


### Generating Soft Labels from Supervised Teachers

FKD flags:

- `--num_crops `: number of crops in each image to generate soft labels. Default: 500.
- `--num_seg `: true number of batch-size on GPUs during generating. Make sure `--num_crops` is divisible by `--num_seg`. Default: 50.
- `--label_type `: type of generated soft labels. Default: `marginal_smoothing_k5`.

Path flags:

- `--save_path `: specify the folder to save soft labels.
- `--reference_path`: specify the path to existing soft labels as the reference of crop locations. This is used for soft label ensemble in [FKD MEAL V2](https://github.com/szq0214/MEAL-V2).
- [imagenet-folder with train and val folders]: ImageNet data folder.

Model flag:

- `--arch `: specify which model to use as the teacher network.
- `--input_size `: input size of teacher network.
- `--teacher_source	`: source of teachers. Currently, it supports models from (1) `pytorch`; (2) `timm`; and (3) private pre-trained models.

Some important notes:

- Modify `normalize` values according to the training settings of teacher networks.

	```
    # EfficientNet_V2_L, BEIT, etc.
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])            
    # ResNet, efficientnet_l2_ns_475, etc.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ```

- Modify `--min_scale_crops` and `--max_scale_crops` according to the training setting of teacher networks. For example, `tf_efficientnet_l2_ns_475` in `timm` has the scale=(0.08, 0.936).
- `--num_seg` is the true number of batch-size, thus `-b` can be set to a relatively small value. It will not slow down the training.
- `Resume` is supported by simply restart. You can also launch multiple experiments parallelly to speed up generating, it will automatically skip the existing files.

**Important:** Test your teacher models using `--evaluate` to check whether the accuracy is correct before starting to generate soft labels.

An example of the command line for generating soft labels from `tf_efficientnet_l2_ns_475 `:

```
python generate_soft_label.py \
-a tf_efficientnet_l2_ns_475 \
--input_size 475 \
--min_scale_crops 0.08 \
--max_scale_crops 0.936 \
--num_crops 500 \
--num_seg 50 \
-b 4 \
--label_type marginal_smoothing_k5 \
--save_path FKD_efficientnet_l2_ns_475_marginal_smoothing_k5 \
[imagenet-folder with train and val folders]
```

Soft label generation from self-supervised teachers will be available soon.

