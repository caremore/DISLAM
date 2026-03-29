# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer

from .layers import ConvBlock, Conv3x3, upsample
from collections import OrderedDict

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc  # [256, 512, 1024, 1024]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # i=4 → i=0 共5层

        self.convs = OrderedDict()

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0 and i - 1 < len(num_ch_enc):
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        x = input_features[-1]  # [1, 1024, 16, 16]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            if self.use_skips and i > 0 and i - 1 < len(input_features):
                skip = input_features[i - 1]
                if skip.shape[2:] != x[0].shape[2:]:#F.interpolate(x, scale_factor=2, mode="nearest")
                    skip = F.interpolate(skip, size=x[0].shape[2:], mode='nearest')
                x += [skip]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                disp = self.sigmoid(self.convs[("dispconv", i)](x))
                if i == 0:
                    disp = F.interpolate(disp, size=(384, 384), mode="nearest")#size=(512, 512),(192, 640) (384, 384)
                self.outputs[("disp", i)] = disp

        return self.outputs
class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x):
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)

            y = x.transpose(1, 2)
            B, C, L = y.shape

            H = W = int(L ** 0.5)
            # if L==120:
            #     H=6
            #     W=20
            # elif L==480:
            #     H = 12
            #     W = 40
            # else:
            #     H = 24
            #     W = 80

            y = y.reshape(B, C, H, W)

            features.append(y)
        #x = self.norm(x)

        return features

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}




class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.num_ch_enc = np.array([256, 512, 1024, 1024])
        self.decoder = DepthDecoder(self.num_ch_enc)

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x_aug):

        x_aug = (x_aug - 0.45) / 0.225
        z = self.encoder(x_aug)
        depth = self.decoder(z)



        return depth

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_simmim(config,pretrained=True):
    model_type = config['model_type']
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_chans=config['in_chans'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config['qkv_bias'],
            qk_scale=config['qk_scale'],
            drop_rate=config['drop_rate'],
            drop_path_rate=config['drop_path_rate'],
            ape=config['ape'],
            patch_norm=config['patch_norm'],
            use_checkpoint=config['use_checkpoint'])
        encoder_stride = 32
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    # 导入 ImageNet 的预训练权重
    if pretrained:
        weights = '/autodl-fs/data/pythoncode/VIODE_atd/swinv2_small_patch4_window8_256.pth'
        pretrained_dict = torch.load(weights, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #print(model)
        # pretrained_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        # model.load_state_dict(pretrained_model.state_dict(), strict=False)

    return model


# import torch
#
# # 引入你写的模型构建函数
#
# # ---------- Step 1: 配置模型参数 ----------
# config = {
#     'model_type': 'swin',
#     'img_size':  512,
#     'patch_size': 4,
#     'in_chans': 3,
#     'num_classes': 0,
#     'embed_dim': 128,
#     'depths': [2, 2, 18, 2],
#     'num_heads': [4, 8, 16, 32],
#     'window_size': 4,
#     'mlp_ratio': 4.0,
#     'qkv_bias': True,
#     'qk_scale': None,
#     'drop_rate': 0.0,
#     'drop_path_rate': 0.1,
#     'ape': False,
#     'patch_norm': True,
#     'use_checkpoint': False
# }
#
# # ---------- Step 2: 构建模型并加载预训练 ----------
# model = build_simmim(config, pretrained=False)
#
# # ---------- Step 3: 构造随机输入 ----------
# B, C, H, W = 1, 3, 512, 512  # batch size, channel, height, width
# x_aug = torch.rand(B, C, H, W)  # 模拟遮挡图像
#
#
# # ---------- Step 4: 前向推理 ----------
#
# output = model(x_aug)

# ---------- Step 5: 打印结果 ----------
# print("✅ 模型运行成功！")
# print("输入图像形状:", x_aug.shape)
# print("输出图像形状:", output.shape)  # 应为 [1, 3, 192, 192]

# B, H, W = 1, 512, 512
# num_ch_enc = [256, 512, 1024, 1024]  # encoder 每层输出通道np.array([256, 512, 1024, 1024])
#
# # 构造多尺度 encoder 输出
# input_features = [
#     torch.randn(B, 256, H//8, W//8),           # 512x512
#     torch.randn(B, 512, H//16, W//16),     # 256x256
#     torch.randn(B, 1024, H//32, W//32),    # 128x128
#     torch.randn(B, 1024, H//32, W//32),    # 64x64
# ]
#
# # 初始化解码器
# decoder = DepthDecoder(num_ch_enc=num_ch_enc)
#
# # 前向传播
# outputs = decoder(input_features)
#
# # 打印输出结果 shape
# for s in outputs:
#     print(f"{s}: {outputs[s].shape}")
