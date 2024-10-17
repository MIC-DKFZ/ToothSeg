from baselines.models.relunet.single_tooth import SingleToothSegmentationNet
import baselines.nn as nn


class FocalSingleToothSegmentationNet(SingleToothSegmentationNet):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.seg_criterion = nn.BinarySegmentationLoss(
            bce_weight=0.0, dice_weight=1.0, focal_weight=1.0,
        )
