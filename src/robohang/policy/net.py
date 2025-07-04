# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
""" Parts of the U-Net model """

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        assert in_channels > out_channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels - out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Kernel_size=1 Conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels_list: List[int],
        output_channel: int,
    ):
        super().__init__()
        self.input_channel = int(input_channel)
        self.channels_list = [int(x) for x in channels_list] # [64, 128, 256, 512, 1024]
        self.output_channel = int(output_channel)
        self.channels_list_len = len(self.channels_list)

        # self.positional_encoding = bool(positional_encoding)
        # self.inc = DoubleConv(self.input_channel + (0 if not self.positional_encoding else 2), self.channels_list[0])
        self.inc = DoubleConv(self.input_channel, self.channels_list[0])
        self.outc = OutConv(self.channels_list[0], self.output_channel)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for idx in range(1, self.channels_list_len):
            self.downs.append(Down(self.channels_list[idx - 1], self.channels_list[idx]))
            self.ups.append(Up(self.channels_list[self.channels_list_len - idx], self.channels_list[self.channels_list_len - idx - 1]))

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W]"""
        xs = [self.inc(x)] + [None] * (self.channels_list_len - 1)
        for idx, down in enumerate(self.downs):
            xs[idx + 1] = down(xs[idx])

        x = xs[-1]
        for idx, up in enumerate(self.ups):
            x = up(x, xs[self.channels_list_len - idx - 2])
        
        out = self.outc(x)
        return out


class UNetMultiHead(nn.Module):
    def __init__(
        self,
        input_channel: int,
        input_height: int,
        input_width: int,
        channels_list: List[int],
        segment_output_channels: List[int],
        classif_output_mlp_dims: List[List[int]],
        conv_out_dim: Optional[int],
    ):
        super().__init__()
        self.input_channel = int(input_channel)
        self.channels_list = [int(x) for x in channels_list] # [64, 128, 256, 512, 1024]
        self.segment_output_channels = [int(x) for x in segment_output_channels]
        self.classif_output_mlp_dims = [[int(_) for _ in x] for x in classif_output_mlp_dims]
        self.channels_list_len = len(self.channels_list)
        
        output_height = input_height
        output_width = input_width
        self.inc = DoubleConv(self.input_channel, self.channels_list[0])
        self.downs = nn.ModuleList()
        for idx in range(1, self.channels_list_len):
            self.downs.append(Down(self.channels_list[idx - 1], self.channels_list[idx]))
            output_height //= 2
            output_width //= 2
        if conv_out_dim is not None:
            assert conv_out_dim <= output_height * output_width * self.channels_list[-1], \
                "{} {} {} {}".format(conv_out_dim, output_height, output_width, self.channels_list[-1])
            self.conv_out_dim = conv_out_dim
        else:
            self.conv_out_dim = output_height * output_width * self.channels_list[-1]
        
        # segment
        self.outc = nn.ModuleList()
        self.ups = nn.ModuleList()
        for soc in self.segment_output_channels:
            self.outc.append(OutConv(self.channels_list[0], soc))
            ups = nn.ModuleList()
            for idx in range(1, self.channels_list_len):
                ups.append(Up(self.channels_list[self.channels_list_len - idx], self.channels_list[self.channels_list_len - idx - 1]))
            self.ups.append(ups)
        
        # classify
        self.mlps = nn.ModuleList()
        for mlp_dims in self.classif_output_mlp_dims:
            self.mlps.append(torch.nn.Sequential(
                torch.nn.Linear(self.conv_out_dim, mlp_dims[0]), torch.nn.ReLU(inplace=True),
                *[torch.nn.Sequential(
                    torch.nn.Linear(mlp_dims[i], mlp_dims[i + 1]), 
                    torch.nn.ReLU(inplace=True)) for i in range(len(mlp_dims) - 2)
                ],
                torch.nn.Linear(mlp_dims[-2], mlp_dims[-1]),
            ))
    
    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        xs = [self.inc(x)] + [None] * (self.channels_list_len - 1)
        for idx, down in enumerate(self.downs):
            xs[idx + 1] = down(xs[idx])

        global_feature = xs[-1]
        return global_feature, xs

    def encode(self, x: torch.Tensor):
        return self._encode(x)

    def decode_seg(self, global_feature: torch.Tensor, xs: List[torch.Tensor]):
        B, C, H, W = global_feature.shape

        seg_list = []
        for head_idx in range(len(self.segment_output_channels)):
            x = global_feature
            for idx, up in enumerate(self.ups[head_idx]):
                x = up(x, xs[self.channels_list_len - idx - 2])
            seg_list.append(self.outc[head_idx](x))
        if len(seg_list) > 0:
            seg = torch.concat(seg_list, dim=1)
        else:
            seg = None
        
        return seg

    def decode_cls(self, global_feature: torch.Tensor):
        B, C, H, W = global_feature.shape

        cls_list = []
        for head_idx in range(len(self.classif_output_mlp_dims)):
            x = global_feature
            cls_list.append(self.mlps[head_idx](x.view(B, -1)[:, :self.conv_out_dim]))
        if len(cls_list) > 0:
            cls = torch.concat(cls_list, dim=1)
        else:
            cls = None
        
        return cls


class CNN(nn.Module):
    def __init__(
        self,
        input_channel: int,
        input_height: int,
        input_width: int,
        output_channel: int,
        channels_list: List[int],
        output_mlp_hidden: List[int],
    ):
        super().__init__()
        self.input_channel = int(input_channel)
        self.input_height = input_height
        self.input_width = input_width
        self.channels_list = [int(x) for x in channels_list]
        self.channels_list_len = len(self.channels_list)
        self.output_mlp_hidden = [int(x) for x in output_mlp_hidden]

        self.inc = DoubleConv(self.input_channel, self.channels_list[0])
        self.downs = nn.ModuleList()
        output_height = input_height
        output_width = input_width
        for idx in range(1, self.channels_list_len):
            self.downs.append(Down(self.channels_list[idx - 1], self.channels_list[idx]))
            output_height //= 2
            output_width //= 2
        conv_out_dim = output_height * output_width * self.channels_list[-1]
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, self.output_mlp_hidden[0]), torch.nn.ReLU(inplace=True),
            *[torch.nn.Sequential(
                torch.nn.Linear(self.output_mlp_hidden[i], self.output_mlp_hidden[i + 1]), 
                torch.nn.ReLU(inplace=True)) for i in range(len(self.output_mlp_hidden) - 1)],
            torch.nn.Linear(self.output_mlp_hidden[-1], output_channel),
        )

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        xs = [self.inc(x)] + [None] * (self.channels_list_len - 1)
        for idx, down in enumerate(self.downs):
            xs[idx + 1] = down(xs[idx])

        x = xs[-1]
        return self.out_mlp(x.view(B, -1))


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List[int], last_no_activate=True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp = torch.nn.Sequential(
            *[torch.nn.Linear(input_dim, self.hidden_dim[0]), torch.nn.ReLU(inplace=True)],
            *[torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]), 
                torch.nn.ReLU(inplace=True),
            ) for i in range(len(self.hidden_dim) - 1)],
            *([torch.nn.Linear(self.hidden_dim[-1], output_dim)] + ([] if last_no_activate else [torch.nn.ReLU(inplace=True)])),
        )
    
    def forward(self, x):
        return self.mlp(x)
        