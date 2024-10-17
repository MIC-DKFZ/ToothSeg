from typing import List, Union

from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    
    def __init__(
        self,
        in_channels=1,
        latent_channels=0,
        encoder_channels: List[int]=[16, 32, 32, 64, 64, 128, 128, 256, 256],
        decoder_channels: List[int]=[128, 128, 64, 64, 32, 32, 16, 16],
        encoder_depths: List[int]=[1, 2, 3, 3, 3],
        decoder_depths: List[int]=[3, 3, 2, 1],
        out_channels: Union[List[int], int]=1,
        normalization='batchnorm',
        encoder_dropout=False,
        decoder_dropout=False,
        out_indices: List[int]=None,
        shared_blocks: int=0,
    ):
        super(VNet, self).__init__()

        # encoder blocks
        self.block_one = ConvBlock(encoder_depths[0], in_channels, encoder_channels[0], normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(encoder_channels[0], encoder_channels[1], normalization=normalization)

        self.block_two = ConvBlock(encoder_depths[1], encoder_channels[1], encoder_channels[2], normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(encoder_channels[2], encoder_channels[3], normalization=normalization)

        self.block_three = ConvBlock(encoder_depths[2], encoder_channels[3], encoder_channels[4], normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(encoder_channels[4], encoder_channels[5], normalization=normalization)

        self.block_four = ConvBlock(encoder_depths[3], encoder_channels[5], encoder_channels[6], normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(encoder_channels[6], encoder_channels[7], normalization=normalization)

        self.block_five_a = ConvBlock(encoder_depths[4] - 1, encoder_channels[7], encoder_channels[8], normalization=normalization)
        self.block_five_b = ConvBlock(1, encoder_channels[8], encoder_channels[8], normalization=normalization)

        # latent blocks
        self.has_latent = latent_channels > 0
        if self.has_latent:
            channels = encoder_channels[-1]
            self.latent_layers = nn.Sequential(
                nn.Linear(channels, channels // 2, bias=False),
                nn.BatchNorm1d(channels // 2),
                nn.ReLU(),
                nn.Linear(channels // 2, channels // 4, bias=False),
                nn.BatchNorm1d(channels // 4),
                nn.ReLU(),
                nn.Linear(channels // 4, latent_channels),
            )

        # decoder blocks
        self.decoder_blocks = nn.ModuleList()
        out_channels = [out_channels] if not isinstance(out_channels, list) else out_channels
        decoder_channels = encoder_channels[-1:] + decoder_channels
        out_indices = [4]*len(out_channels) if out_indices is None else out_indices
        shared_indices = [0] + [shared_blocks]*(len(out_channels) - 1)
        for start_idx, shared_idx, out_channels in zip(out_indices, shared_indices, out_channels):
            self.decoder_blocks.append(nn.ModuleList())

            self.decoder_blocks[-1].extend(self.decoder_blocks[0][:max(2 * shared_idx - 1, 0)])

            for i in range(shared_idx + 4 - start_idx, 4):
                channels = decoder_channels[2*i:2*i + 3]

                if shared_idx > 0 and i == shared_idx:
                    block = ConvBlock(decoder_depths[i - 1], decoder_channels[2*i - 1], channels[0], normalization=normalization)
                    self.decoder_blocks[-1].append(block)

                block_up = UpsamplingDeconvBlock(channels[0], channels[1], normalization=normalization)
                self.decoder_blocks[-1].append(block_up)

                if decoder_depths[i] == 0:
                    continue

                block = ConvBlock(decoder_depths[i], channels[1], channels[2], normalization=normalization)
                self.decoder_blocks[-1].append(block)
                

            out_conv = nn.Conv3d(channels[-2], out_channels, 1, padding=0)
            self.decoder_blocks[-1].append(out_conv)
        
        # dropout
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5a = self.block_five_a(x4_dw)
        x5 = self.block_five_b(x5a)
        if self.encoder_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return x5a, res
    
    def attributes(self, features):
        if not self.has_latent:
            return ()
        
        embeddings = F.max_pool3d(
            input=features,
            kernel_size=features.shape[-1],
        )
        embeddings = embeddings.reshape(features.shape[:-3])
        logits = self.latent_layers(embeddings)

        return (logits,)
    
    def decoder(self, features, blocks):
        features = features[:len(blocks) // 2 + 1][::-1]
        x = features[0]
        for i in range(0, len(blocks) - 1, 2):
            if i > 0:
                x = blocks[i - 1](x)
            
            x = blocks[i](x)
            if (i // 2 + 1) < (len(features) - 1):
                x = x + features[i // 2 + 1]

        if len(blocks) % 2 == 1:
            x = x + features[-1]
            x = blocks[-2](x)

        if self.decoder_dropout:
            x = self.dropout(x)

        out = blocks[-1](x)
        
        return out        

    def forward(self, input):
        # convolutional encoder
        latents, features = self.encoder(input)

        # convolutional decoder (multiple branches)
        outs = ()
        for blocks in self.decoder_blocks:
            out = self.decoder(features, blocks)
            outs += (out,)

        outs += self.attributes(latents)

        if len(outs) == 1:
            return outs[0]
        
        return outs
