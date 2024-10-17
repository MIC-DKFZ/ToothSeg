import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.relu  = nn.ReLU()

        num_groups = int(out_ch / 8)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
    
    def forward(self, x):
        return self.norm2(self.relu(self.conv2(self.norm1(self.relu(self.conv1(x))))))


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256,512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            x        = torch.cat([x, encoder_features[i]], dim=1)
            x        = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    
    def __init__(
        self,
        in_channels=1,
        enc_chs=(64, 128, 256, 512),
        dec_chs=(512, 256, 128, 64),
        out_channels=1,
    ):
        super().__init__()

        self.encoder     = Encoder((in_channels,) + enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv3d(dec_chs[-1], out_channels, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[-1], enc_ftrs[::-1][1:])
        out      = self.head(out)
        
        return out
