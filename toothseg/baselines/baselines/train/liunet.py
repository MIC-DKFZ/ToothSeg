import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import os, sys
sys.path.append(os.getcwd())

from baselines.datamodules import (
    LiuNetFovCropSegDataModule,
    LiuNetInstancesFovCropSegDataModule,
)
from baselines.models import (
    LiuToothSegmentationNet,
    LiuToothInstanceSegmentationNet,
)


def train(stage: str, devices: int, checkpoint: Optional[str]):
    with open('baselines/config/liunet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule'].update(config['datamodule'][stage])
    if stage == 'binary':
        dm = LiuNetFovCropSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'instances':
        dm = LiuNetInstancesFovCropSegDataModule(seed=config['seed'], **config['datamodule'])
    
    config['model'][stage].pop('pretrained')
    if stage == 'binary':
        model = LiuToothSegmentationNet(**config['model'])
    elif stage == 'instances':
        model = LiuToothInstanceSegmentationNet(**config['model'])

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'liunet_{stage}_{config["version"]}',
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
        max_epochs=config['model']['epochs'],
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            metric_checkpoint_callback,
            LearningRateMonitor(),
        ],
        sync_batchnorm=True,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['binary', 'instances'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    args = parser.parse_args()

    train(args.stage, args.devices, args.checkpoint)
