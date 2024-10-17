import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import os, sys
sys.path.append(os.getcwd())

from baselines.datamodules import (
    WangNetFovCropSegDataModule,
    WangNetToothPatchSegDataModule,
)
from baselines.models import (
    DentalNet,
    FocalSingleToothSegmentationNet,
)


def train(stage: str, devices: int, checkpoint: Optional[str]):
    with open('baselines/config/wangnet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule'].update(config['datamodule'][stage])
    if stage == 'instances':
        dm = WangNetFovCropSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'single_tooth':
        dm = WangNetToothPatchSegDataModule(seed=config['seed'], **config['datamodule'])

    config['model'][stage].pop('pretrained')
    if stage == 'instances':
        model = DentalNet(**config['model'][stage])
    elif stage == 'single_tooth':
        model = FocalSingleToothSegmentationNet(**config['model'][stage])

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'wangnet_{stage}_{config["version"]}',
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
        monitor='f1/val' if stage == 'instances' else 'dice/val',
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
    parser.add_argument('stage', choices=['instances', 'single_tooth'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    args = parser.parse_args()

    train(args.stage, args.devices, args.checkpoint)
