from typing import List
import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


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


class CNN(nn.Module):
    def __init__(
        self,
        input_channel: int,
        input_height: int,
        input_width: int,
        output_channel: int,

        channels_list: List[int],
        output_mlp_hidden: List[int],
        last_activate: bool
    ):
        super().__init__()
        self.input_channel = int(input_channel)
        self.input_height = input_height
        self.input_width = input_width
        self.output_channel = output_channel

        self.channels_list = [int(x) for x in channels_list]
        self.channels_list_len = len(self.channels_list)
        self.output_mlp_hidden = [int(x) for x in output_mlp_hidden]
        self.last_activate = last_activate

        self.inc = DoubleConv(self.input_channel, self.channels_list[0])
        self.downs = nn.ModuleList()
        output_height = input_height
        output_width = input_width
        for idx in range(1, self.channels_list_len):
            self.downs.append(Down(self.channels_list[idx - 1], self.channels_list[idx]))
            output_height //= 2
            output_width //= 2
        conv_out_dim = output_height * output_width * self.channels_list[-1]
        self.out_mlp = nn.Sequential(
            *[nn.Linear(conv_out_dim, self.output_mlp_hidden[0]), nn.ReLU(inplace=True)],
            *[nn.Sequential(
                nn.Linear(self.output_mlp_hidden[i], self.output_mlp_hidden[i + 1]), 
                nn.ReLU(inplace=True)
            ) for i in range(len(self.output_mlp_hidden) - 1)],
            *([nn.Linear(self.output_mlp_hidden[-1], output_channel)] + ([nn.ReLU(inplace=True)] if last_activate else [])),
        )

    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        xs = [self.inc(x)] + [None] * (self.channels_list_len - 1)
        for idx, down in enumerate(self.downs):
            xs[idx + 1] = down(xs[idx])

        x = xs[-1]
        return self.out_mlp(x.view(B, -1))


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class CNNBackBone(nn.Module):
    def __init__(
        self,
        input_channel: int,
        input_height: int,
        input_width: int,
        channels_list: List[int],
        output_dim: int,
        last_activate: bool,
    ):
        super().__init__()
        self.input_channel = int(input_channel)
        self.input_height = input_height
        self.input_width = input_width

        self.channels_list = [int(x) for x in channels_list]
        self.channels_list_len = len(self.channels_list)

        self.inc = DoubleConv(self.input_channel, self.channels_list[0])
        self.downs = nn.ModuleList()
        output_height = input_height
        output_width = input_width
        for idx in range(1, self.channels_list_len):
            self.downs.append(Down(self.channels_list[idx - 1], self.channels_list[idx]))
            output_height //= 2
            output_width //= 2
        self.FH, self.FW, self.FD = output_height, output_width, self.channels_list[-1]

        self.output_dim = output_dim
        self.last_activate = last_activate
        self.projector = torch.nn.Conv2d(in_channels=self.channels_list[-1], out_channels=output_dim, kernel_size=(1, 1))
        self.xype = PositionEmbeddingSine(num_pos_feats=output_dim // 2, normalize=True)
        
    def forward(self, x: torch.Tensor):
        """x: [B, C, H, W]"""
        B, C, H, W = x.shape
        assert H == self.input_height and W == self.input_width, x.shape
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        assert x.shape == (B, self.FD, self.FH, self.FW)

        x = self.projector(x)
        x = x + self.xype(x)
        if self.last_activate:
            x = torch.nn.functional.relu(x)
        return x


class ResnetBackBone(nn.Module):
    def __init__(self, output_dim: int, last_activate: bool):
        super().__init__()

        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return_layers = {'layer3': 'feat3'}
        self.feature_extractor = IntermediateLayerGetter(self.net, return_layers=return_layers)

        self.last_activate = last_activate
        self.output_dim = output_dim
        self.projector = torch.nn.Conv2d(in_channels=256, out_channels=output_dim, kernel_size=(1, 1))
        self.xype = PositionEmbeddingSine(num_pos_feats=output_dim // 2, normalize=True)
    
    def forward(self, x):
        x = self.feature_extractor(x)["feat3"] # [B, C, H, W]

        x = self.projector(x) # [B, C', H, W]
        x = x + self.xype(x)
        if self.last_activate:
            x = torch.nn.functional.relu(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List[int], last_activate: bool) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            *[nn.Linear(input_dim, self.hidden_dim[0]), nn.ReLU(inplace=True)],
            *[nn.Sequential(
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]), 
                nn.ReLU(inplace=True),
            ) for i in range(len(self.hidden_dim) - 1)],
            *([nn.Linear(self.hidden_dim[-1], output_dim)] + ([nn.ReLU(inplace=True)] if last_activate else [])),
        )
    
    def forward(self, x):
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        self.pe.to(device=x.device)
        x = x + self.pe[:x.shape[-2], :]
        return x