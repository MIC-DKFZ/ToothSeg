import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import os, sys
sys.path.append(os.getcwd())

from baselines.datamodules import (
    CuiNetFovCropSegDataModule,
    CuiNetToothSkeletonsCropSegDataModule,
    CuiNetToothPatchKeypointSegDataModule,
)
from baselines.models import (
    ToothSegmentationNet,
    ToothSegmentationOffsetsNet,
    SingleToothPredictionNet,
)


def train(stage: str, devices: int, checkpoint: Optional[str]):
    with open('baselines/config/cuinet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule'].update(config['datamodule'][stage])
    if stage == 'roi':
        dm = CuiNetFovCropSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage in ['centroids', 'skeletons']:
        dm = CuiNetToothSkeletonsCropSegDataModule(seed=config['seed'], **config['datamodule'])
    if stage == 'single_tooth':
        dm = CuiNetToothPatchKeypointSegDataModule(seed=config['seed'], **config['datamodule'])
    
    config['model'][stage].pop('pretrained')
    if stage == 'roi':
        model = ToothSegmentationNet(**config['model'][stage])
    elif stage in ['centroids', 'skeletons']:
        model = ToothSegmentationOffsetsNet(**config['model'][stage])
    if stage == 'single_tooth':
        model = SingleToothPredictionNet(**config['model'][stage])

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'cuinet_{stage}_{config["version"]}',
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='epoch',
        mode='max',
        filename='weights-{epoch:02d}',
    )
    loss_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='loss/val',
        filename='weights-{epoch:02d}',
    )
    metric_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='dice/val',
        mode='max',
        filename='weights-{epoch:02d}',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        max_epochs=config['model'][stage]['epochs'],
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            metric_checkpoint_callback,
            LearningRateMonitor(),
        ],
        sync_batchnorm=True,
        accumulate_grad_batches=3 // devices,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['roi', 'centroids', 'skeletons', 'single_tooth'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    args = parser.parse_args()

    train(args.stage, args.devices, args.checkpoint)
