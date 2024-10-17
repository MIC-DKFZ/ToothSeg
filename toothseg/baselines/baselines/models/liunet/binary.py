from baselines.models.cuinet.roi import ToothSegmentationNet
import baselines.nn as nn


class LiuToothSegmentationNet(ToothSegmentationNet):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(architecture={}, **kwargs)

        self.model = nn.UNet(out_channels=1)
        self.seg_criterion = nn.BinarySegmentationLoss(
            bce_weight=1.0, dice_weight=1.0,
        )
