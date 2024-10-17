# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

"""
This code is modified from pytorch ERFNET implementation:
https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ERFNet-CULane-PyTorch/models/erfnet.py
https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection/blob/master/networks/erfnet.py
"""
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class DownsamplerBlock (nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv3d(ninput, noutput-ninput, 3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool3d(2, stride=2)
        self.bn = nn.BatchNorm3d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):

    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1x1_1 = nn.Conv3d(chann, chann, (3,1,1), stride=1, padding=(1,0,0), bias=True)
        self.conv1x3x1_1 = nn.Conv3d(chann, chann, (1,3,1), stride=1, padding=(0,1,0), bias=True)
        self.conv1x1x3_1 = nn.Conv3d(chann, chann, (1,1,3), stride=1, padding=(0,0,1), bias=True)

        self.bn1 = nn.BatchNorm3d(chann, eps=1e-03)

        self.conv3x1x1_2 = nn.Conv3d(chann, chann, (3,1,1), stride=1, padding=(1*dilated,0,0), bias=True, dilation=(dilated,1,1))
        self.conv1x3x1_2 = nn.Conv3d(chann, chann, (1,3,1), stride=1, padding=(0,1*dilated,0), bias=True, dilation=(1,dilated,1))
        self.conv1x1x3_2 = nn.Conv3d(chann, chann, (1,1,3), stride=1, padding=(0,0,1*dilated), bias=True, dilation=(1,1,dilated))

        self.bn2 = nn.BatchNorm3d(chann, eps=1e-03)

        self.dropout = nn.Dropout3d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1x1_1(input)
        output = F.relu(output)
        output = self.conv1x3x1_1(output)
        output = F.relu(output)
        output = self.conv1x1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1x1_2(output)
        output = F.relu(output)
        output = self.conv1x3x1_2(output)
        output = F.relu(output)
        output = self.conv1x1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output + input)


class Encoder(nn.Module):

    def __init__(self, in_channels: int, num_filters: int):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, num_filters)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(num_filters, 4 * num_filters))

        for x in range(5):    #5 times
           self.layers.append(non_bottleneck_1d(4 * num_filters, 0.1, 1))  

        self.layers.append(DownsamplerBlock(4 * num_filters, 8 * num_filters))

        for x in range(2):    #2 times
            self.layers.append(non_bottleneck_1d(8 * num_filters, 0.1, 2))
            self.layers.append(non_bottleneck_1d(8 * num_filters, 0.1, 4))
            self.layers.append(non_bottleneck_1d(8 * num_filters, 0.1, 8))
            self.layers.append(non_bottleneck_1d(8 * num_filters, 0.1, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock (nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose3d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm3d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)
    

class Decoder (nn.Module):

    def __init__(self, out_channels: Optional[int], num_filters: int):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(8 * num_filters, 4 * num_filters))
        self.layers.append(non_bottleneck_1d(4 * num_filters, 0, 1))
        self.layers.append(non_bottleneck_1d(4 * num_filters, 0, 1))

        self.layers.append(UpsamplerBlock(4 * num_filters, num_filters))
        self.layers.append(non_bottleneck_1d(num_filters, 0, 1))
        self.layers.append(non_bottleneck_1d(num_filters, 0, 1))

        self.use_last_conv = out_channels is not None
        if self.use_last_conv:
            self.output_conv = nn.ConvTranspose3d(num_filters, out_channels, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        if self.use_last_conv:
            output = self.output_conv(output)

        return output
    

class ERFNet(nn.Module):

    def __init__(            
        self,
        in_channels=1,
        out_channels: Union[List[int], int]=1,
        num_filters: int=32,
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, num_filters)

        self.decoders = nn.ModuleList()
        out_channels = [out_channels] if not isinstance(out_channels, list) else out_channels
        for channels in out_channels:
            decoder = Decoder(channels, num_filters)
            self.decoders.append(decoder)

    def forward(self, input):
        """
        Forward function of 3D ERFNet

        Returns (Tensor or list[Tensor]):
            One output or multiple outputs depending on number of decoders.
        """
        output = self.encoder(input)
        
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(output))

        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
    

class Identification(nn.Module):

    def __init__(
        self,
        num_features: int,
        out_channels: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_features, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, out_channels, bias=True),
        )

        self.num_features = num_features
        self.out_channels = out_channels

    def forward(
        self,
        features: TensorType['B', 'C', 'd', 'h', 'w', torch.float32],
        instances: TensorType['B', 'D', 'H', 'W', torch.int64],
    ):
        # apply instance masks to get prototypes
        prototypes = []
        for b in range(instances.shape[0]):
            for id in instances[b].unique()[1:]:
                in_mask = instances[b] == id
                in_mask = F.interpolate(
                    input=in_mask[None, None].float().detach(),
                    size=tuple(features.shape[-3:]),
                    mode='trilinear',
                )[0, 0] >= 0.5
                in_features = features[b, :, in_mask]
                prototypes.append(in_features if in_mask.sum() else None)

        # note when instance is removed due to interpolation
        keep = torch.tensor([proto is not None for proto in prototypes])
        keep_idxs = torch.nonzero(keep)[:, 0].to(features.device)

        # perform average pooling to get mean embeddings
        if keep_idxs.shape[0] > 0:
            embeddings = torch.stack([prototypes[idx].mean(1) for idx in keep_idxs])
        else:
            embeddings = torch.zeros(0, features.shape[1]).to(features)

        # process prototypes for prediction
        out = torch.full((len(prototypes), self.out_channels), torch.nan).to(features)
        out[keep_idxs] = self.mlp(embeddings)

        return prototypes, out
