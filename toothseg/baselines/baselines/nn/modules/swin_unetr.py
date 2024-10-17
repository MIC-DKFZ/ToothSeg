from typing import Tuple

from monai.networks.nets.swin_unetr import SwinUNETR as MonaiSwinUNETR


class SwinUNETR(MonaiSwinUNETR):

    def __init__(
        self,
        img_size: Tuple[int, int, int],
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],            
        )
