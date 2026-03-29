import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock, Conv3x3, upsample


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)

            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


# B, H, W = 1, 512, 512
# num_ch_enc = [64, 64, 128, 256, 512]  # encoder 每层输出通道
#
# # 构造多尺度 encoder 输出
# input_features = [
#     torch.randn(B, 64, H//2, W//2),           # 512x512
#     torch.randn(B, 64, H//4, W//4),     # 256x256
#     torch.randn(B, 128, H//8, W//8),    # 128x128
#     torch.randn(B, 256, H//16, W//16),    # 64x64
#     torch.randn(B, 512, H//32, W//32),  # 32x32
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