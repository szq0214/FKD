# SReT (Sliced Recursive Transformer: https://arxiv.org/abs/2111.05297)
# Zhiqiang Shen 
# CMU & MBZUAI

# PiT (Rethinking Spatial Dimensions of Vision Transformers)
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

# Timm (https://github.com/rwightman/pytorch-image-models)
# Ross Wightman
# Apache License v2.0

import torch
from einops import rearrange
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple, lecun_normal_
from timm.models.registry import register_model


__all__ = [
    "SReT",
    "SReT_T",
    "SReT_LT",
    "SReT_S",
]


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Non_proj(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()

    def forward(self, x, recursive_index):
        x = self.coefficient1(x) + self.coefficient2(self.mlp(self.norm1(x)))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_groups1=8, num_groups2=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups1 = num_groups1
        self.num_groups2 = num_groups2
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, recursive_index):
        B, N, C = x.shape
        if recursive_index == False:
            num_groups = self.num_groups1
        else:
            num_groups = self.num_groups2
            if num_groups != 1:
                idx = torch.randperm(N)
                x = x[:,idx,:]
                inverse = torch.argsort(idx)
        qkv = self.qkv(x).reshape(B, num_groups, N // num_groups, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)  
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_groups, N // num_groups, C)
        x = x.permute(0, 3, 1, 2).reshape(B, C, N).transpose(1, 2)
        if recursive_index == True and num_groups != 1:
            x = x[:,inverse,:]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Transformer_Block(nn.Module):

    def __init__(self, dim, num_groups1, num_groups2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_groups1=num_groups1, num_groups2=num_groups2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()

    def forward(self, x, recursive_index):
        x = self.coefficient1(x) + self.coefficient2(self.drop_path(self.attn(self.norm1(x),recursive_index)))
        
        x = self.coefficient3(x) + self.coefficient4(self.drop_path(self.mlp(self.norm2(x))))
        return x

class Transformer(nn.Module):
    def __init__(self, base_dim, depth, recursive_num, groups1, groups2, heads, mlp_ratio, np_mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        blocks = [
            Transformer_Block(
                dim=embed_dim,
                num_groups1=groups1,
                num_groups2=groups2,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(recursive_num)]

        recursive_loops = int(depth/recursive_num)
        non_projs = [
            Non_proj(
                dim=embed_dim, num_heads=heads, mlp_ratio=np_mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=drop_path_prob[i], norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
            for i in range(depth)]
        RT = []
        for rn in range(recursive_num):
            for rl in range(recursive_loops):
                RT.append(blocks[rn])
                RT.append(non_projs[rn*recursive_loops+rl])

        self.blocks = nn.ModuleList(RT)

    def forward(self, x):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        for i, blk in enumerate(self.blocks):
            if (i+2)%4 == 0: # mark the recursive layers
                recursive_index = True
            else:
                recursive_index = False
            x = blk(x, recursive_index)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):

        x = self.conv(x)

        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=3,
                              stride=2, padding=1, bias=True)
        self.bn1 = norm_layer(int(out_channels/2))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(out_channels/2), out_channels, kernel_size=3,
                              stride=2, padding=1, bias=True)
        self.bn2 = norm_layer(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=2, padding=1, bias=True)
        self.bn3 = norm_layer(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


class SReT(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, recursive_num, groups1, groups2, heads,
                 mlp_ratio, np_mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super(SReT, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = int(image_size/8)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], recursive_num[stage], groups1[stage], groups2[stage], heads[stage],
                            mlp_ratio, np_mlp_ratio, 
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)

        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Distilled_SReT(SReT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.forward_features(x)
        x_cls = self.head(x)
        # `x_cls, x_cls` is used to make it compatible with DeiT codebase, while SReT uses global_average pooling, and soft label only for knowledge distillation
        # so `x_cls` is enough
        if self.training:
            # return x_cls, x_cls
            return x_cls
        else:
            return x_cls


@register_model
def SReT_T(pretrained=False, **kwargs):
    model = SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2,5,3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=3.6,
        np_mlp_ratio=1,
        drop_path_rate=0.1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_T.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

@register_model
def SReT_LT(pretrained=False, **kwargs):
    model = SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=4.0,
        np_mlp_ratio=1,
        drop_path_rate=0.1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_LT.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

def SReT_S(pretrained=False, **kwargs):
    model = SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[42, 42, 42],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[3, 6, 12],
        groups1=[8, 4, 1], 
        groups2=[2, 1, 1],
        mlp_ratio=3.0,
        np_mlp_ratio=2,
        drop_path_rate=0.2,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_S.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

# Knowledge Distillation
@register_model
def SReT_T_distill(pretrained=False, **kwargs):
    model = Distilled_SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=3.6,
        np_mlp_ratio=1,
        drop_path_rate=0.1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_T_distill.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

@register_model
def SReT_LT_distill(pretrained=False, **kwargs):
    model = Distilled_SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=4.0,
        np_mlp_ratio=1,
        drop_path_rate=0.1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_LT_distill.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

def SReT_S_distill(pretrained=False, **kwargs):
    model = Distilled_SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[42, 42, 42],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[3, 6, 12],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=3.0,
        np_mlp_ratio=2,
        drop_path_rate=0.2,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_S_distill.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model